"""Train the learned make/miss model (app/outcome_model.json) on the dev labels.

Replays every labeled dev run with the current scoring code and the model off, keeps the
outcome measurements of each predicted shot matched to a made/missed label (shots with
a rim known at arrival only; eval/fixed.csv too, each video a group), then:

- leave-one-game-out (LOGO): p(made) for each shot from a model that never saw its game;
- call thresholds by nested LOGO: inside each outer fold, out-of-fold p on the other
  games (grouped folds) picks made at p >= t_hi, missed at p <= t_lo, else unknown,
  maximizing right - 4 x wrong calls with no more misses called made than the rules;
- the shipped model: trained on all shots, thresholds picked the same way on the LOGO p.

    python scripts/train_outcome_model.py --runs eval/runs --jobs data/jobs

Rerun after any change to the shot or outcome rules; --rows caches the replayed
measurements (--reuse reads them back instead of replaying).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BALLFORM_OUTCOME_MODEL"] = "0"  # measure the rules' shots, never a previous model's calls
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

from app import evaluation as EV  # noqa: E402
from app.outcome_model import FEATURES, GBM, MODEL_PATH, OutcomeModel  # noqa: E402

FRESH = {"BKN@HOU 2025-10-27", "WAS@CHI 2025-11-22", "ATL@OKC 2025-12-29", "POR@SAS 2026-01-03",
         "PHX@MIA 2026-01-13", "LAL@DEN 2026-01-20", "PHX@POR 2026-02-03", "CLE@SAC 2026-02-07",
         "LAC@HOU 2026-02-11", "DET@NYK 2026-02-19", "IND@MIL 2026-03-15", "WAS@LAL 2026-03-30",
         "ORL@CHI 2026-04-10", "PHI@NYK 2026-05-06", "DET@CLE 2026-05-11", "NYK@SAS 2026-06-13"}
CALL = {"made": 1, "likely made": 1, "missed": 0}
PARAMS = {"n_estimators": 150, "learning_rate": .05, "max_depth": 3, "min_leaf": 8, "lam": 1.}
WRONG_COST = 4  # a call is worth making above 80% right
HI = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95, .975]
LO = [.5, .45, .4, .35, .3, .25, .2, .15, .1, .05, .025]


def _one_thread() -> None:
    import cv2
    cv2.setNumThreads(1)


def replay_job(job_id: str, jobs_dir: Path, runs_dir: Path, court_landmarks: Path | None) -> list[dict]:
    """``evaluation.replay`` that also returns each shot's outcome measurements."""
    from app.analyzer import SCORING_INPUTS, fit_calibration, load_scoring_inputs, score_clip

    job_dir = jobs_dir / job_id
    saved = runs_dir / job_id / SCORING_INPUTS
    inputs = load_scoring_inputs(saved)
    landmarks = EV._court_landmarks(job_id, EV._settings(job_dir), court_landmarks)
    if landmarks and inputs["game_mode"]:
        inputs["court_landmarks"], inputs["calibration"] = fit_calibration(landmarks, inputs["width"], inputs["height"])
    source = next(iter(sorted(job_dir.glob("input.*"))), None)
    cache = None
    if inputs["calibration"] is not None and source is not None:
        cache = runs_dir / job_id / f"court_map-{EV._court_map_key(inputs['court_landmarks'], saved, source)}.pkl"
    scored = score_clip(inputs, None, source, court_map_cache=cache)
    features = scored.get("outcome_features") or [None] * len(scored["shots"])
    return [{"release_s": shot.release_s, "outcome": shot.outcome, "features": f}
            for shot, f in zip(scored["shots"], features)]


def collect(label_paths: list[Path], jobs_dir: Path, runs_dir: Path, workers: int) -> list[dict]:
    rows = []
    court_landmarks = ROOT / "eval" / "court_landmarks.json"
    for path in label_paths:
        assert "holdout" not in path.name, "never train on a holdout set"
        by_job: dict[str, list] = {}
        for label in EV.read_labels(path):
            by_job.setdefault(label.job_id, []).append(label)
        started = time.time()
        with ProcessPoolExecutor(workers, initializer=_one_thread) as pool:
            futures = {job: pool.submit(replay_job, job, jobs_dir, runs_dir, court_landmarks) for job in by_job}
            for job, labels in by_job.items():
                try:
                    shots = futures[job].result()
                except Exception as exc:  # a run that cannot be replayed is left out, as in evaluation
                    print(f"  {job}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                game = next((EV.game_of(label.notes) for label in labels if EV.game_of(label.notes)), job)
                for match in EV.match_shots(labels, shots, .75):
                    if match.label is None or match.shot is None or match.shot["features"] is None:
                        continue
                    if match.label.outcome not in ("made", "missed"):
                        continue
                    rows.append({"set": path.stem, "job_id": job, "game": game, "label_release_s": match.label.release_s,
                                 "made": int(match.label.outcome == "made"),
                                 "rules": CALL.get(match.shot["outcome"], -1), "features": match.shot["features"]})
        print(f"{path.name}: {len(by_job)} clips replayed in {time.time() - started:.0f} s", file=sys.stderr)
    return rows


def _fit_predict(args):
    X, y, train, test = args
    return GBM(**PARAMS).fit(X[train], y[train]).predict_proba(X[test])[:, 1]


def grouped_oof(X, y, groups, mask, pool, folds: int | None = None) -> np.ndarray:
    """Out-of-fold p(made) over ``mask``: leave one group out, or ``folds`` folds of whole groups."""
    names = sorted(set(groups[mask]))
    parts = [[g] for g in names] if folds is None else [names[i::folds] for i in range(folds)]
    jobs = [(X, y, mask & ~np.isin(groups, part), mask & np.isin(groups, part)) for part in parts]
    p = np.full(len(y), np.nan)
    for (_, _, _, test), pred in zip(jobs, pool.map(_fit_predict, jobs)):
        p[test] = pred
    return p


def score_calls(p, y, rules, lo: float, hi: float) -> tuple[int, int, int]:
    made, missed = p >= hi, (p <= lo) & (p < hi)
    right = int((made & (y == 1)).sum() + (missed & (y == 0)).sum())
    wrong = int((made & (y == 0)).sum() + (missed & (y == 1)).sum())
    return right, wrong, int((made & (y == 0)).sum())


def choose_thresholds(p, y, rules) -> tuple[float, float]:
    """Most right - WRONG_COST x wrong calls, with no more misses called made than the rules;
    ties go to the wider band of unknowns."""
    rules_mcm = int(((rules == 1) & (y == 0)).sum())
    best, choice = None, (.05, .95)
    for hi in reversed(HI):
        for lo in reversed(LO):
            if lo > hi:
                continue
            right, wrong, mcm = score_calls(p, y, rules, lo, hi)
            if mcm > rules_mcm:
                continue
            value = right - WRONG_COST * wrong
            if best is None or value > best:
                best, choice = value, (lo, hi)
    return choice


def nested_thresholds(X, y, rules, groups, mask, pool) -> tuple[np.ndarray, np.ndarray]:
    """Per shot, the thresholds chosen without its game (inner grouped 5-fold out-of-fold p)."""
    lo, hi = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for g in sorted(set(groups[mask])):
        inner = mask & (groups != g)
        p_inner = grouped_oof(X, y, groups, inner, pool, folds=5)
        t = choose_thresholds(p_inner[inner], y[inner], rules[inner])
        lo[groups == g], hi[groups == g] = t
    return lo, hi


def auc(y, p) -> float:
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    for value in np.unique(p):  # average tied ranks
        tied = p == value
        ranks[tied] = ranks[tied].mean()
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def line(name: str, mask, y, rules, p=None, lo=None, hi=None) -> str:
    n = int(mask.sum())
    if p is None:
        called = mask & (rules >= 0)
        right = int((called & (rules == y)).sum())
        mcm = int((called & (rules == 1) & (y == 0)).sum())
        mcmk = int((called & (rules == 0) & (y == 1)).sum())
        return f"  {name:<34} n {n:3d} called {int(called.sum()):3d} right {right:3d} MCM {mcm} makes-called-missed {mcmk}"
    made, missed = mask & (p >= hi), mask & (p <= lo) & (p < hi)
    right = int((made & (y == 1)).sum() + (missed & (y == 0)).sum())
    mcm, mcmk = int((made & (y == 0)).sum()), int((missed & (y == 1)).sum())
    ok = mask & ~np.isnan(p)
    return (f"  {name:<34} n {n:3d} called {int((made | missed).sum()):3d} right {right:3d} MCM {mcm} "
            f"makes-called-missed {mcmk}  (AUC {auc(y[ok], p[ok]):.3f})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=ROOT / "eval" / "runs")
    parser.add_argument("--jobs", type=Path, default=ROOT / "data" / "jobs")
    parser.add_argument("--labels", type=Path, default=ROOT / "eval" / "labels.csv")
    parser.add_argument("--fixed", type=Path, default=ROOT / "eval" / "fixed.csv",
                        help="fixed-camera labels, trained on too (grouped by video, as games)")
    parser.add_argument("--dev-only", action="store_true",
                        help="train on --labels only; --fixed is then reported as transfer")
    parser.add_argument("--rows", type=Path, help="cache of the replayed measurements")
    parser.add_argument("--reuse", action="store_true", help="read --rows instead of replaying")
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args(argv)
    paths = [args.labels] + ([args.fixed] if args.fixed and args.fixed.exists() else [])
    if args.reuse and args.rows and args.rows.exists():
        rows = json.loads(args.rows.read_text())
    else:
        rows = collect(paths, args.jobs, args.runs, args.workers)
        if args.rows:
            args.rows.write_text(json.dumps(rows))
    X = np.array([[row["features"].get(name, np.nan) for name in FEATURES] for row in rows], float)
    y = np.array([row["made"] for row in rows])
    rules = np.array([row["rules"] for row in rows])
    groups = np.array([row["game"] for row in rows])
    dev = np.array([row["set"] == args.labels.stem for row in rows])
    fixed = ~dev
    train = dev | (fixed & (not args.dev_only))
    fresh = dev & np.isin(groups, list(FRESH))
    print(f"{int(dev.sum())} dev shots ({len(set(groups[dev]))} games, {int(y[dev].sum())} made), "
          f"{int(fixed.sum())} fixed shots ({len(set(groups[fixed]))} videos); training on {int(train.sum())}")
    with ProcessPoolExecutor(args.workers) as pool:
        p = grouped_oof(X, y, groups, train, pool)
        lo, hi = nested_thresholds(X, y, rules, groups, train, pool)
        t_lo, t_hi = choose_thresholds(p[train], y[train], rules[train])
        model = OutcomeModel(GBM(**PARAMS).fit(X[train], y[train]), FEATURES, t_lo, t_hi)
    if args.dev_only and fixed.any():
        p[fixed] = model.p_made([dict(zip(FEATURES, x)) for x in X[fixed]])
        lo[fixed], hi[fixed] = t_lo, t_hi
    print("\nLOGO (each game's shots from a model that never saw that game; thresholds by nested LOGO):")
    for name, mask in (("all dev", dev), ("fresh 16", fresh), ("tuned 40", dev & ~fresh), ("fixed", fixed)):
        if not mask.any():
            continue
        print(line(f"rules, {name}", mask, y, rules))
        tag = "trained on dev, thresholds shipped" if name == "fixed" and args.dev_only else "LOGO"
        print(line(f"model, {name} ({tag})", mask, y, rules, p, lo, hi))
    chosen = sorted({(float(a), float(b)) for a, b in zip(lo[train], hi[train])})
    print(f"\nnested-LOGO thresholds per fold (t_lo, t_hi): {chosen}")
    print(f"shipped thresholds (from all LOGO p): missed at p <= {t_lo}, made at p >= {t_hi}")
    model.meta = {"trained_on": [p_.name for p_ in paths if p_ == args.labels or not args.dev_only],
                  "shots": int(train.sum()), "games": len(set(groups[train])), "params": PARAMS,
                  "trained": time.strftime("%Y-%m-%d")}
    model.save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
