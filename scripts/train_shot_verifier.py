"""Train the shot verifier (app/shot_verifier.json): p(real) for each shot the rules find.

Replays every labeled broadcast run (eval/labels.csv) with both learned models off, labels
each predicted shot real (matched to a labeled attempt) or false, then reports leave-one-
game-out numbers with the drop threshold chosen by nested LOGO (most false shots dropped -
3 x real shots dropped), trains on all of them and writes the model. The app applies it to
a camera following play only; eval/fixed.csv is reported as transfer.

    python scripts/train_shot_verifier.py --runs eval/runs --jobs data/jobs
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
os.environ["BALLFORM_OUTCOME_MODEL"] = "0"
os.environ["BALLFORM_SHOT_VERIFIER"] = "0"  # every shot the rules find
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np  # noqa: E402

from app import evaluation as EV  # noqa: E402
from app.outcome_model import GBM  # noqa: E402
from app.shot_verifier import FEATURES, MODEL_PATH, Verifier  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import train_outcome_model as TOM  # noqa: E402  (replay, LOGO and threshold helpers)

PARAMS = TOM.PARAMS
GRID = [0., .05, .1, .15, .2, .25, .3, .35, .4, .5]
REAL_COST = 3  # dropping a real shot costs as much as keeping three false ones


def replay_job(job_id: str, jobs_dir: Path, runs_dir: Path, court_landmarks: Path | None) -> list[dict]:
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
    features = scored.get("verifier_features") or [None] * len(scored["shots"])
    return [{"release_s": shot.release_s, "features": f} for shot, f in zip(scored["shots"], features)]


def collect(label_paths: list[Path], jobs_dir: Path, runs_dir: Path, workers: int) -> list[dict]:
    rows = []
    court_landmarks = ROOT / "eval" / "court_landmarks.json"
    for path in label_paths:
        assert "holdout" not in path.name, "never train on a holdout set"
        by_job: dict[str, list] = {}
        for label in EV.read_labels(path):
            by_job.setdefault(label.job_id, []).append(label)
        started = time.time()
        with ProcessPoolExecutor(workers, initializer=TOM._one_thread) as pool:
            futures = {job: pool.submit(replay_job, job, jobs_dir, runs_dir, court_landmarks) for job in by_job}
            for job, labels in by_job.items():
                try:
                    shots = futures[job].result()
                except Exception as exc:
                    print(f"  {job}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                game = next((EV.game_of(label.notes) for label in labels if EV.game_of(label.notes)), job)
                n_labels = sum(label.release_s is not None for label in labels)
                for match in EV.match_shots(labels, shots, .75):
                    if match.shot is None or match.shot["features"] is None:
                        continue
                    rows.append({"set": path.stem, "job_id": job, "game": game, "real": int(match.label is not None),
                                 "release_s": match.shot["release_s"], "features": match.shot["features"]})
                rows.append({"set": path.stem, "job_id": job, "game": game, "labels": n_labels})
        print(f"{path.name}: {len(by_job)} clips replayed in {time.time() - started:.0f} s", file=sys.stderr)
    return rows


def choose(p, y) -> float:
    best, choice = None, 0.
    for t in GRID:
        drop = p < t
        value = int((drop & (y == 0)).sum()) - REAL_COST * int((drop & (y == 1)).sum())
        if best is None or value > best:
            best, choice = value, t
    return choice


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path, default=ROOT / "eval" / "runs")
    parser.add_argument("--jobs", type=Path, default=ROOT / "data" / "jobs")
    parser.add_argument("--labels", type=Path, default=ROOT / "eval" / "labels.csv")
    parser.add_argument("--fixed", type=Path, default=ROOT / "eval" / "fixed.csv")
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
    n_labels = {}
    for row in rows:
        if "labels" in row:
            n_labels[(row["set"], row["game"])] = n_labels.get((row["set"], row["game"]), 0) + row["labels"]
    rows = [row for row in rows if "features" in row]
    X = np.array([[row["features"].get(name, np.nan) for name in FEATURES] for row in rows], float)
    y = np.array([row["real"] for row in rows])
    groups = np.array([row["game"] for row in rows])
    dev = np.array([row["set"] == args.labels.stem for row in rows])
    fresh = dev & np.isin(groups, list(TOM.FRESH))
    everything = dev  # trained and validated on broadcast clips only
    print(f"{len(y)} predicted shots, {int(y.sum())} real, {int((y == 0).sum())} false; {len(set(groups))} groups")
    with ProcessPoolExecutor(args.workers) as pool:
        p = TOM.grouped_oof(X, y, groups, everything, pool)
        t_shot = np.full(len(y), np.nan)
        for g in sorted(set(groups[dev])):
            inner = dev & (groups != g)
            p_inner = TOM.grouped_oof(X, y, groups, inner, pool, folds=5)
            t_shot[groups == g] = choose(p_inner[inner], y[inner])
        threshold = choose(p[dev], y[dev])
        model = Verifier(GBM(**PARAMS).fit(X[dev], y[dev]), FEATURES, threshold)
    if (~dev).any():  # fixed cameras: the shipped model, for reference (the app does not apply it there)
        p[~dev] = model.p_real([dict(zip(FEATURES, x)) for x in X[~dev]])
        t_shot[~dev] = threshold
    print(f"\nLOGO AUC {TOM.auc(y[dev], p[dev]):.3f}; drops by nested-LOGO thresholds (fixed: shipped model):")
    for name, mask in (("all dev", dev), ("fresh 16", fresh), ("tuned 40", dev & ~fresh), ("fixed", ~dev)):
        drop = mask & (p < t_shot)
        labels = sum(v for (s, g), v in n_labels.items()
                     if (s == args.labels.stem) == (name != "fixed")
                     and (name not in ("fresh 16", "tuned 40") or (g in TOM.FRESH) == (name == "fresh 16")))
        print(f"  {name:<9} labels {labels:3d} predicted {int(mask.sum()):3d} real {int((mask & (y == 1)).sum()):3d} "
              f"false {int((mask & (y == 0)).sum()):2d} | dropped false {int((drop & (y == 0)).sum()):2d}, "
              f"dropped real {int((drop & (y == 1)).sum())}")
    print(f"nested thresholds: { {float(t): int((t_shot == t).sum()) for t in sorted(set(t_shot))} }")
    print(f"shipped threshold (from all LOGO p): drop at p(real) < {threshold}")
    model.meta = {"trained_on": [args.labels.name], "shots": int(dev.sum()), "groups": len(set(groups[dev])),
                  "params": PARAMS, "trained": time.strftime("%Y-%m-%d")}
    model.save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
