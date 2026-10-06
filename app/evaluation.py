"""Score analyses against hand labels in eval/labels.csv.

Each labeled row is one shot in a finished job: its release time plus whatever
is known (outcome, shot type, distance, zone). A row with an empty release_s
marks a clip with no shots, so every detection in it counts as a false shot.
Predicted shots are matched to labels by release time, one to one.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from app.shots import RIM_TYPES

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "data" / "jobs"
EVAL_DIR = ROOT / "eval"

OUTCOMES = {"made", "missed", "unclear", ""}
SHOT_TYPES = {"jump shot", "floater", "layup", "dunk", "tip", ""}
ZONES = {"paint", "midrange", "corner_three", "above_break_three", ""}
THREES = {"corner_three", "above_break_three"}


@dataclass
class Label:
    job_id: str
    release_s: float | None
    outcome: str = ""
    shot_type: str = ""
    distance_ft: float | None = None
    zone: str = ""
    shooter: str = ""
    source: str = ""
    notes: str = ""
    line: int = 0


@dataclass
class Match:
    label: Label | None
    shot: dict | None

    @property
    def job_id(self) -> str:
        return self.label.job_id if self.label else ""


@dataclass
class ClipResult:
    job_id: str
    matches: list[Match] = field(default_factory=list)
    error: str | None = None


def _float(value: str, line: int, name: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"line {line}: {name} must be a number, not {value!r}") from None


def read_labels(path: Path) -> list[Label]:
    labels = []
    with path.open(newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            row = {key.strip(): (value or "").strip() for key, value in row.items() if key}
            if not row.get("job_id") or row["job_id"].startswith("#"):
                continue
            label = Label(row["job_id"], _float(row.get("release_s", ""), line, "release_s"),
                          row.get("outcome", "").lower(), row.get("shot_type", "").lower(),
                          _float(row.get("distance_ft", ""), line, "distance_ft"), row.get("zone", "").lower(),
                          row.get("shooter", ""), row.get("source", ""), row.get("notes", ""), line)
            for name, allowed in (("outcome", OUTCOMES), ("shot_type", SHOT_TYPES), ("zone", ZONES)):
                if getattr(label, name) not in allowed:
                    raise ValueError(f"line {line}: {name} must be one of {sorted(allowed - {''})}, "
                                     f"not {getattr(label, name)!r}")
            labels.append(label)
    return labels


def match_shots(labels: list[Label], shots: list[dict], tolerance_s: float) -> list[Match]:
    """Pair labels with predicted shots by release time, closest pairs first."""
    labels = [label for label in labels if label.release_s is not None]
    pairs = sorted((abs(label.release_s - shot["release_s"]), i, j)
                   for i, label in enumerate(labels) for j, shot in enumerate(shots)
                   if shot.get("release_s") is not None and abs(label.release_s - shot["release_s"]) <= tolerance_s)
    used_labels, used_shots, matches = set(), set(), []
    for _, i, j in pairs:
        if i not in used_labels and j not in used_shots:
            used_labels.add(i)
            used_shots.add(j)
            matches.append(Match(labels[i], shots[j]))
    matches += [Match(label, None) for i, label in enumerate(labels) if i not in used_labels]
    matches += [Match(None, shot) for j, shot in enumerate(shots) if j not in used_shots]
    return matches


def _called(outcome: str) -> str | None:
    """The make/miss call a predicted outcome makes, or None when it abstains."""
    return {"made": "made", "likely made": "made", "missed": "missed"}.get(outcome)


def _type_correct(label: str, predicted: str) -> bool:
    return predicted == label or (predicted == "layup or dunk" and label in {"layup", "dunk"})


def _type_group(shot_type: str) -> str:
    """The coarse split scored until there are enough labels per fine type: rim finishes vs everything else."""
    return "rim" if shot_type in RIM_TYPES else "shot"


def _metric(shot: dict, name: str):
    return ((shot.get("game") or {}).get("metrics") or {}).get(name)


def _pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "0/0"


def summarize(clips: list[ClipResult]) -> dict:
    matches = [match for clip in clips if clip.error is None for match in clip.matches]
    paired = [m for m in matches if m.label and m.shot]
    labeled = [m for m in matches if m.label]
    predicted = [m for m in matches if m.shot]
    summary = {"clips": len(clips), "clips_failed": sum(clip.error is not None for clip in clips),
               "labeled_shots": len(labeled), "predicted_shots": len(predicted), "matched": len(paired)}
    summary["detection"] = {"recall": _pct(len(paired), len(labeled)), "precision": _pct(len(paired), len(predicted))}
    if paired:
        summary["release_error_s_median"] = round(statistics.median(
            abs(m.label.release_s - m.shot["release_s"]) for m in paired), 3)

    judged = [m for m in paired if m.label.outcome in {"made", "missed"}]
    called = [m for m in judged if _called(m.shot.get("outcome", "")) is not None]
    confusion: dict[str, dict[str, int]] = {}
    for m in judged:
        row = confusion.setdefault(m.label.outcome, {})
        row[m.shot.get("outcome", "unknown")] = row.get(m.shot.get("outcome", "unknown"), 0) + 1
    summary["outcome"] = {
        "coverage": _pct(len(called), len(judged)),
        "accuracy_when_called": _pct(sum(_called(m.shot["outcome"]) == m.label.outcome for m in called), len(called)),
        "labeled_vs_predicted": confusion,
        "misses_called_made": sum(m.label.outcome == "missed" and _called(m.shot.get("outcome", "")) == "made"
                                  for m in judged),
    }

    typed = [m for m in paired if m.label.shot_type]
    groups: dict[str, dict[str, int]] = {}
    for m in typed:
        row = groups.setdefault(_type_group(m.label.shot_type), {})
        predicted_group = _type_group(m.shot.get("shot_type") or "")
        row[predicted_group] = row.get(predicted_group, 0) + 1
    summary["shot_type"] = {
        "accuracy": _pct(sum(_type_correct(m.label.shot_type, m.shot.get("shot_type") or "") for m in typed),
                         len(typed)),
        "shot_vs_rim": _pct(sum(row.get(group, 0) for group, row in groups.items()), len(typed)),
        "shot_vs_rim_labeled_vs_predicted": groups,
    }

    distanced = [m for m in paired if m.label.distance_ft is not None]
    measured = [m for m in distanced if _metric(m.shot, "shot_distance_ft") is not None]
    errors = [_metric(m.shot, "shot_distance_ft") - m.label.distance_ft for m in measured]
    summary["distance_ft"] = {"coverage": _pct(len(measured), len(distanced))}
    if errors:
        summary["distance_ft"].update({
            "mean_abs_error": round(statistics.fmean(abs(e) for e in errors), 2),
            "median_abs_error": round(statistics.median(abs(e) for e in errors), 2),
            "bias": round(statistics.fmean(errors), 2), "worst": round(max(errors, key=abs), 1)})

    zoned = [m for m in paired if m.label.zone and _metric(m.shot, "shot_zone")]
    summary["zone"] = {"accuracy": _pct(sum(_metric(m.shot, "shot_zone") == m.label.zone for m in zoned), len(zoned)),
                       "two_vs_three": _pct(sum((_metric(m.shot, "shot_zone") in THREES) == (m.label.zone in THREES)
                                                for m in zoned), len(zoned))}
    return summary


def shot_rows(clips: list[ClipResult]) -> list[dict]:
    rows = []
    for clip in clips:
        if clip.error:
            rows.append({"job_id": clip.job_id, "problem": clip.error})
            continue
        for m in sorted(clip.matches, key=lambda m: (m.label.release_s if m.label and m.label.release_s is not None
                                                     else m.shot["release_s"])):
            shot = m.shot or {}
            rows.append({
                "job_id": clip.job_id,
                "label_release_s": m.label.release_s if m.label else None,
                "predicted_release_s": shot.get("release_s"),
                "problem": "missed shot" if not m.shot else "false shot" if not m.label else "",
                "outcome": f"{m.label.outcome if m.label else '-'} / {shot.get('outcome', '-')}",
                "shot_type": f"{m.label.shot_type if m.label else '-'} / {shot.get('shot_type') or '-'}",
                "distance_ft": f"{m.label.distance_ft if m.label and m.label.distance_ft is not None else '-'} / "
                               f"{_metric(shot, 'shot_distance_ft') if shot else '-'}",
                "zone": f"{m.label.zone if m.label else '-'} / {_metric(shot, 'shot_zone') or '-'}",
                "shooter_label": m.label.shooter if m.label else "",
                "predicted_shooter_track": (shot.get("game") or {}).get("players", {}).get("shooter_track_id"),
            })
    return rows


def _settings(job_dir: Path) -> dict:
    saved = job_dir / "settings.json"
    if saved.exists():
        return json.loads(saved.read_text())
    result = json.loads((job_dir / "result.json").read_text())
    print(f"  {job_dir.name}: no settings.json (job predates it); rerunning with mode/camera/handedness only, "
          "no rim or court calibration", file=sys.stderr)
    return {"mode": result.get("mode", "form"), "camera": result.get("camera_profile", "courtside"),
            "handedness": result.get("handedness", "right")}


def rerun(job_id: str, jobs_dir: Path, runs_dir: Path) -> dict:
    from app.analyzer import analyze_video

    # Keep the frame loop's measurements so --replay can rescore without inference.
    os.environ["BALLFORM_SAVE_SCORING_INPUTS"] = "1"

    job_dir = jobs_dir / job_id
    source = next(iter(sorted(job_dir.glob("input.*"))), None)
    if source is None:
        raise FileNotFoundError(f"no input video in {job_dir}")
    settings = _settings(job_dir)
    out = runs_dir / job_id
    out.mkdir(parents=True, exist_ok=True)
    rim = settings.get("rim")
    return analyze_video(source, out, tuple(rim) if rim else None, mode=settings.get("mode", "form"),
                         handedness=settings.get("handedness", "right"), camera=settings.get("camera", "courtside"),
                         court=settings.get("court"), rim_frame=settings.get("rim_frame"),
                         rim_time_s=settings.get("rim_time_s"), pose_model=settings.get("pose_model", "yolo26s-pose"),
                         court_landmarks=settings.get("court_landmarks"))


def replay(job_id: str, jobs_dir: Path, runs_dir: Path) -> dict:
    """Rescore a rerun's saved measurements with the current scoring code (no inference).

    Court landmarks in the job's settings.json are refit each time, so a calibration
    added after the rerun is used too.
    """
    from app.analyzer import SCORING_INPUTS, fit_calibration, load_scoring_inputs, score_clip

    job_dir = jobs_dir / job_id
    inputs = load_scoring_inputs(runs_dir / job_id / SCORING_INPUTS)
    landmarks = _settings(job_dir).get("court_landmarks")
    if landmarks and inputs["game_mode"]:
        inputs["court_landmarks"], inputs["calibration"] = fit_calibration(landmarks, inputs["width"], inputs["height"])
    source = next(iter(sorted(job_dir.glob("input.*"))), None)
    scored = score_clip(inputs, None, source)
    return {"shots": [shot.to_dict() for shot in scored["shots"]]}


def _load(job_id: str, jobs_dir: Path, how: str, runs_dir: Path) -> dict:
    if how == "rerun":
        print(f"  analyzing {job_id} ...", file=sys.stderr)
        return rerun(job_id, jobs_dir, runs_dir)
    if how == "replay":
        return replay(job_id, jobs_dir, runs_dir)
    return json.loads((jobs_dir / job_id / "result.json").read_text())


def evaluate(labels: list[Label], jobs_dir: Path, tolerance_s: float, runs_dir: Path | None = None,
             how: str = "saved", workers: int = 1) -> list[ClipResult]:
    """Match each labeled job's shots. ``how`` is "saved" (the job's result.json),
    "rerun" (full re-analysis into ``runs_dir``) or "replay" (rescore ``runs_dir``'s
    saved measurements)."""
    by_job: dict[str, list[Label]] = {}
    for label in labels:
        by_job.setdefault(label.job_id, []).append(label)
    with ProcessPoolExecutor(workers) if workers > 1 else nullcontext() as pool:
        if pool is None:
            pending = {job_id: None for job_id in by_job}
        else:
            pending = {job_id: pool.submit(_load, job_id, jobs_dir, how, runs_dir) for job_id in by_job}
        clips = []
        for job_id, job_labels in by_job.items():
            try:
                future = pending[job_id]
                result = future.result() if future else _load(job_id, jobs_dir, how, runs_dir)
            except (OSError, ValueError, RuntimeError, EOFError, KeyError) as exc:
                clips.append(ClipResult(job_id, error=f"{type(exc).__name__}: {exc}"))
                continue
            clips.append(ClipResult(job_id, match_shots(job_labels, result.get("shots") or [], tolerance_s)))
    return clips


def _print(summary: dict, rows: list[dict]) -> None:
    print(f"\n{summary['clips']} clips ({summary['clips_failed']} failed), {summary['labeled_shots']} labeled shots, "
          f"{summary['predicted_shots']} predicted, {summary['matched']} matched")
    for key in ("detection", "outcome", "shot_type", "distance_ft", "zone"):
        print(f"  {key}: " + ", ".join(f"{name} {value}" for name, value in summary[key].items()))
    if "release_error_s_median" in summary:
        print(f"  median release-time error: {summary['release_error_s_median']} s")
    print("\nshots (label / predicted):")
    for row in rows:
        if "label_release_s" not in row:
            print(f"  {row['job_id']}: {row['problem']}")
            continue
        when = row["label_release_s"] if row["label_release_s"] is not None else row["predicted_release_s"]
        print(f"  {row['job_id']} @{when}s {row['problem'] or 'matched'}: outcome {row['outcome']}, "
              f"type {row['shot_type']}, distance {row['distance_ft']}, zone {row['zone']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Ballform results against eval/labels.csv.")
    parser.add_argument("--labels", type=Path, default=EVAL_DIR / "labels.csv")
    parser.add_argument("--jobs", type=Path, default=JOBS_DIR, help="directory holding the labeled jobs")
    parser.add_argument("--rerun", action="store_true",
                        help="re-analyze each labeled clip with its saved settings into eval/runs/ "
                             "(use after changing the code) instead of scoring the saved results")
    parser.add_argument("--replay", action="store_true",
                        help="rescore the measurements saved by the last --rerun with the current scoring code; "
                             "seconds instead of minutes, valid while the frame loop (models, tracking) is unchanged")
    parser.add_argument("--runs", type=Path, default=EVAL_DIR / "runs",
                        help="where --rerun writes and --replay reads (default eval/runs)")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)),
                        help="parallel processes for --replay")
    parser.add_argument("--tolerance", type=float, default=.75, help="max release-time gap to match a shot, seconds")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "report.json")
    args = parser.parse_args(argv)
    try:
        labels = read_labels(args.labels)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not labels:
        parser.error(f"no labels in {args.labels}")
    if args.rerun and args.replay:
        parser.error("choose --rerun or --replay")
    how = "rerun" if args.rerun else "replay" if args.replay else "saved"
    clips = evaluate(labels, args.jobs, args.tolerance, args.runs, how, args.workers if args.replay else 1)
    summary, rows = summarize(clips), shot_rows(clips)
    _print(summary, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "shots": rows}, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
