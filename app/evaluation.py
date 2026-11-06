"""Score analyses against hand labels in eval/labels.csv.

Each labeled row is one shot in a finished job: its release time plus whatever
is known (outcome, shot type, distance, zone). A row with an empty release_s
marks a clip with no shots, so every detection in it counts as a false shot.
Predicted shots are matched to labels by release time, one to one.

Two reports of the same labels can be compared shot by shot (``--compare``,
``--compare-reports``): a paired test, which sees much smaller changes than
the two summaries' intervals do.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
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


# A broadcast game ("LAL@HOU 2026-03-18"), or the source video of a non-broadcast one ("youtube:dQw4w9WgXcQ").
GAME = re.compile(r"\b([A-Z]{2,4}@[A-Z]{2,4} \d{4}-\d{2}-\d{2}|youtube:[A-Za-z0-9_-]{11})")


def game_of(notes: str) -> str:
    """The game a label's notes name ("LAL@HOU 2026-03-18", or "youtube:<video id>" for footage
    with no play-by-play), or "" when they name none.

    Clips from one game share an arena, a broadcast and players, so tuning checks
    hold out whole games (leave-one-game-out), not single clips."""
    found = GAME.search(notes or "")
    return found.group(1) if found else ""


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
    game: str = ""
    duration_s: float | None = None


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


def _pair_times(first: list[float | None], second: list[float | None], tolerance_s: float) -> list[tuple[int, int]]:
    """Pair two lists of times one to one, closest pairs first, at most ``tolerance_s`` apart."""
    pairs = sorted((abs(a - b), i, j) for i, a in enumerate(first) for j, b in enumerate(second)
                   if a is not None and b is not None and abs(a - b) <= tolerance_s)
    used_first, used_second, paired = set(), set(), []
    for _, i, j in pairs:
        if i not in used_first and j not in used_second:
            used_first.add(i)
            used_second.add(j)
            paired.append((i, j))
    return paired


def match_shots(labels: list[Label], shots: list[dict], tolerance_s: float) -> list[Match]:
    """Pair labels with predicted shots by release time, closest pairs first."""
    labels = [label for label in labels if label.release_s is not None]
    paired = _pair_times([label.release_s for label in labels], [shot.get("release_s") for shot in shots],
                         tolerance_s)
    matches = [Match(labels[i], shots[j]) for i, j in paired]
    used_labels, used_shots = {i for i, _ in paired}, {j for _, j in paired}
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


Z_95 = 1.959964  # two-sided 95% normal quantile
PROPORTION = re.compile(r"(\d+)/(\d+)(?: \(\d+%\))?$")  # what _pct writes


def wilson(part: int, whole: int) -> tuple[float, float] | None:
    """The 95% Wilson score interval of part/whole, or None when whole is 0."""
    if not whole:
        return None
    rate, z2 = part / whole, Z_95 ** 2
    centre = (rate + z2 / (2 * whole)) / (1 + z2 / whole)
    half = Z_95 * math.sqrt(rate * (1 - rate) / whole + z2 / (4 * whole ** 2)) / (1 + z2 / whole)
    return max(0.0, centre - half), min(1.0, centre + half)


def intervals(summary: dict) -> dict:
    """The 95% Wilson interval, in percent, of each "part/whole (pct%)" value in a summary, keyed
    the same way. Works on old reports too."""
    found = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            if inner := intervals(value):
                found[key] = inner
        elif isinstance(value, str) and (proportion := PROPORTION.match(value)):
            if bounds := wilson(int(proportion[1]), int(proportion[2])):
                found[key] = [round(100 * bound, 1) for bound in bounds]
    return found


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

    # One number for finding a shot and calling it right, and one for false shots: the pair to watch.
    right = sum(_called(m.shot.get("outcome", "")) == m.label.outcome for m in judged)
    summary["primary"] = {"found_and_called_right": _pct(right, len(labeled)),
                          "false_shots": len(predicted) - len(paired)}
    scored = [clip for clip in clips if clip.error is None]
    timed = [clip for clip in scored if clip.duration_s]
    minutes = sum(clip.duration_s for clip in timed) / 60
    if minutes:
        false_shots = sum(1 for clip in timed for m in clip.matches if m.shot and not m.label)
        summary["primary"]["false_shots_per_minute"] = round(false_shots / minutes, 2)
        summary["primary"]["footage_minutes"] = round(minutes, 1)
    if len(timed) < len(scored):
        summary["primary"]["clips_without_duration"] = len(scored) - len(timed)
    summary["interval_95_pct"] = intervals(summary)
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
                "game": clip.game,
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


def clip_rows(clips: list[ClipResult]) -> list[dict]:
    """Each clip with its length, or its problem if it failed. A comparison reads which clips were
    scored from these, since a clip with no shot and no false shot has no shot row."""
    return [{"job_id": clip.job_id, "problem": clip.error} if clip.error else
            {"job_id": clip.job_id, "duration_s": clip.duration_s} for clip in clips]


def _duration_s(result: dict, job_dir: Path) -> float | None:
    """The clip's length: the result's (saved and rerun results, and replays from the rerun's
    frame count), else read from the job's video, else None."""
    duration = (result.get("video") or {}).get("duration_s")
    if duration:
        return float(duration)
    source = next(iter(sorted(job_dir.glob("input.*"))), None)
    if source is None:
        return None
    import cv2

    capture = cv2.VideoCapture(str(source))
    fps, frames = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    return round(frames / fps, 2) if fps > 0 and frames > 0 else None


def _settings(job_dir: Path) -> dict:
    saved = job_dir / "settings.json"
    if saved.exists():
        return json.loads(saved.read_text())
    result = json.loads((job_dir / "result.json").read_text())
    print(f"  {job_dir.name}: no settings.json (job predates it); rerunning with mode/camera/handedness only, "
          "no rim or court calibration", file=sys.stderr)
    return {"mode": result.get("mode", "form"), "camera": result.get("camera_profile", "courtside"),
            "handedness": result.get("handedness", "right")}


def _court_landmarks(job_id: str, settings: dict, court_landmarks: Path | None) -> dict | None:
    """The job's own landmarks, else an accepted automatic calibration from ``court_landmarks``
    (written by scripts/auto_court_landmarks.py), else None."""
    if settings.get("court_landmarks") or court_landmarks is None or not court_landmarks.exists():
        return settings.get("court_landmarks")
    return (json.loads(court_landmarks.read_text()).get(job_id) or {}).get("court_landmarks")


def rerun(job_id: str, jobs_dir: Path, runs_dir: Path, court_landmarks: Path | None = None) -> dict:
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
                         court_landmarks=_court_landmarks(job_id, settings, court_landmarks))


def replay(job_id: str, jobs_dir: Path, runs_dir: Path, court_landmarks: Path | None = None) -> dict:
    """Rescore a rerun's saved measurements with the current scoring code (no inference).

    Court landmarks in the job's settings.json (or, failing that, in ``court_landmarks``)
    are refit each time, so a calibration added after the rerun is used too.
    """
    from app.analyzer import SCORING_INPUTS, fit_calibration, load_scoring_inputs, score_clip

    job_dir = jobs_dir / job_id
    saved = runs_dir / job_id / SCORING_INPUTS
    inputs = load_scoring_inputs(saved)
    landmarks = _court_landmarks(job_id, _settings(job_dir), court_landmarks)
    if landmarks and inputs["game_mode"]:
        inputs["court_landmarks"], inputs["calibration"] = fit_calibration(landmarks, inputs["width"], inputs["height"])
    source = next(iter(sorted(job_dir.glob("input.*"))), None)
    cache = None
    if inputs["calibration"] is not None and source is not None:
        cache = runs_dir / job_id / f"court_map-{_court_map_key(inputs['court_landmarks'], saved, source)}.pkl"
    scored = score_clip(inputs, None, source, court_map_cache=cache)
    fps, total = inputs["fps"], inputs["total"]
    return {"shots": [shot.to_dict() for shot in scored["shots"]],
            "video": {"duration_s": round(total / fps, 2) if fps and total else None}}


def _court_map_key(landmarks: dict, saved: Path, source: Path) -> str:
    """Everything the court followed through camera motion depends on: the landmarks, the
    rerun's measurements (player boxes, cuts), the video and the court-following code."""
    digest = hashlib.sha1(json.dumps(landmarks, sort_keys=True, default=str).encode())
    for path in (saved, source):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for module in ("camera_motion.py", "court.py", "vision.py"):
        digest.update((Path(__file__).parent / module).read_bytes())
    return digest.hexdigest()[:16]


def _one_thread() -> None:
    """Parallel replays each get one thread; OpenCV and BLAS otherwise oversubscribe the cores."""
    import cv2
    cv2.setNumThreads(1)


def _load(job_id: str, jobs_dir: Path, how: str, runs_dir: Path, court_landmarks: Path | None = None) -> dict:
    if how == "rerun":
        print(f"  analyzing {job_id} ...", file=sys.stderr)
        return rerun(job_id, jobs_dir, runs_dir, court_landmarks)
    if how == "replay":
        return replay(job_id, jobs_dir, runs_dir, court_landmarks)
    return json.loads((jobs_dir / job_id / "result.json").read_text())


def evaluate(labels: list[Label], jobs_dir: Path, tolerance_s: float, runs_dir: Path | None = None,
             how: str = "saved", workers: int = 1, court_landmarks: Path | None = None) -> list[ClipResult]:
    """Match each labeled job's shots. ``how`` is "saved" (the job's result.json),
    "rerun" (full re-analysis into ``runs_dir``) or "replay" (rescore ``runs_dir``'s
    saved measurements). Rerun and replay calibrate the court from ``court_landmarks``
    for jobs whose settings have no landmarks."""
    by_job: dict[str, list[Label]] = {}
    for label in labels:
        by_job.setdefault(label.job_id, []).append(label)
    if workers > 1:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(name, "1")  # read by the worker processes as they start
    with ProcessPoolExecutor(workers, initializer=_one_thread) if workers > 1 else nullcontext() as pool:
        if pool is None:
            pending = {job_id: None for job_id in by_job}
        else:
            pending = {job_id: pool.submit(_load, job_id, jobs_dir, how, runs_dir, court_landmarks)
                       for job_id in by_job}
        clips = []
        for job_id, job_labels in by_job.items():
            try:
                future = pending[job_id]
                result = future.result() if future else _load(job_id, jobs_dir, how, runs_dir, court_landmarks)
            except (OSError, ValueError, RuntimeError, EOFError, KeyError) as exc:
                clips.append(ClipResult(job_id, error=f"{type(exc).__name__}: {exc}"))
                continue
            game = next((game_of(label.notes) for label in job_labels if game_of(label.notes)), "")
            clips.append(ClipResult(job_id, match_shots(job_labels, result.get("shots") or [], tolerance_s),
                                    game=game, duration_s=_duration_s(result, jobs_dir / job_id)))
    return clips


def mcnemar_p(gains: int, losses: int) -> float:
    """Exact two-sided McNemar p value: how often a split of the changed shots at least this
    uneven happens if each change were as likely a gain as a loss (binomial, p = 1/2)."""
    changed = gains + losses
    tail = sum(math.comb(changed, k) for k in range(max(gains, losses), changed + 1))
    return min(1.0, 2 * tail / 2 ** changed)


def _shot_state(row: dict) -> dict:
    """What a labeled shot row says: found or not, and whether its outcome call and type are right."""
    label_outcome, _, predicted_outcome = row["outcome"].partition(" / ")
    label_type, _, predicted_type = row["shot_type"].partition(" / ")
    found = row["problem"] == ""
    return {"found": found, "judged": label_outcome in {"made", "missed"}, "typed": bool(label_type),
            "right": found and _called(predicted_outcome) == label_outcome,
            "type_right": found and _type_correct(label_type, predicted_type)}


def _index(report: dict) -> tuple[dict, dict, set]:
    """A report's labeled shot rows by (job_id, label_release_s, repeat), its false shots' release
    times by job, and the jobs it scored. Reports from before the clip list show only jobs with a
    shot row."""
    labeled, false_shots, failed, seen = {}, {}, set(), Counter()
    for row in report["shots"]:
        if "label_release_s" not in row:
            failed.add(row["job_id"])
        elif row["label_release_s"] is not None:
            key = (row["job_id"], row["label_release_s"])
            labeled[(*key, seen[key])] = row  # a label repeated at the same time stays two shots
            seen[key] += 1
        else:
            false_shots.setdefault(row["job_id"], []).append(row["predicted_release_s"])
    if "clips" in report:
        scored = {clip["job_id"] for clip in report["clips"] if not clip.get("problem")}
    else:
        scored = {row["job_id"] for row in report["shots"]} - failed
    return labeled, false_shots, scored


def _changes(pairs: list[tuple[dict, dict]], name: str) -> dict:
    gains = sum(not before[name] and after[name] for before, after in pairs)
    losses = sum(before[name] and not after[name] for before, after in pairs)
    return {"before": sum(before[name] for before, _ in pairs), "after": sum(after[name] for _, after in pairs),
            "of": len(pairs), "gains": gains, "losses": losses, "p": mcnemar_p(gains, losses)}


def compare(before: dict, after: dict, tolerance_s: float = .75) -> dict:
    """Pair two reports' shot rows and count, per metric, the shots that got better (gains) and
    worse (losses) from ``before`` to ``after``, with the exact McNemar p value of that split.

    Labeled shots pair by (job_id, label_release_s); a metric scored only on found shots counts
    the shots found in both. False shots pair within each clip scored by both, by predicted
    release time (one to one, closest first, within ``tolerance_s``); a false shot gone is a gain.
    """
    before_labeled, before_false, before_scored = _index(before)
    after_labeled, after_false, after_scored = _index(after)
    pairs = [(_shot_state(before_labeled[key]), _shot_state(after_labeled[key]))
             for key in before_labeled if key in after_labeled]
    both_found = [(a, b) for a, b in pairs if a["found"] and b["found"]]
    metrics = {
        "found": _changes(pairs, "found"),
        "outcome_right_found_in_both": _changes([(a, b) for a, b in both_found if a["judged"] and b["judged"]],
                                                "right"),
        "found_and_called_right": _changes(pairs, "right"),
        "fine_type_right_found_in_both": _changes([(a, b) for a, b in both_found if a["typed"] and b["typed"]],
                                                  "type_right"),
    }
    jobs = before_scored & after_scored
    gone = new = counted_before = counted_after = 0
    for job in jobs:
        old, now = before_false.get(job, []), after_false.get(job, [])
        same = len(_pair_times(old, now, tolerance_s))
        gone, new = gone + len(old) - same, new + len(now) - same
        counted_before, counted_after = counted_before + len(old), counted_after + len(now)
    metrics["false_shots"] = {"before": counted_before, "after": counted_after, "gains": gone, "losses": new,
                              "p": mcnemar_p(gone, new)}
    return {"labeled_shots_in_both": len(pairs), "labeled_shots_only_before": len(before_labeled) - len(pairs),
            "labeled_shots_only_after": len(after_labeled) - len(pairs), "clips_compared": len(jobs),
            "false_shots_not_compared": (sum(len(times) for job, times in before_false.items() if job not in jobs)
                                         + sum(len(times) for job, times in after_false.items() if job not in jobs)),
            "metrics": metrics}


def _print_paired(paired: dict, before: str, after: str) -> None:
    print(f"\npaired, {after} against {before}: {paired['labeled_shots_in_both']} labeled shots in both "
          f"({paired['labeled_shots_only_before']} only before, {paired['labeled_shots_only_after']} only after), "
          f"false shots compared in {paired['clips_compared']} clips")
    for name, metric in paired["metrics"].items():
        of = f" of {metric['of']}" if "of" in metric else ""
        p = f"{metric['p']:.3f}" if metric["p"] >= .001 else "<0.001"
        print(f"  {name:<30} gains {metric['gains']:<3} losses {metric['losses']:<3} p {p:<6}  "
              f"{metric['before']} -> {metric['after']}{of}")
    if paired["false_shots_not_compared"]:
        print(f"  ({paired['false_shots_not_compared']} false shots not compared: their clip is missing from one "
              "report, or it is an older report that lists only clips with a shot row)")
    print("  gains: right after, not before (a false shot: gone after); p: exact two-sided McNemar test")


def _with_interval(value) -> str:
    """A summary value, and after a proportion its 95% Wilson interval in percent."""
    proportion = PROPORTION.match(value) if isinstance(value, str) else None
    bounds = wilson(int(proportion[1]), int(proportion[2])) if proportion else None
    return f"{value} [{100 * bounds[0]:.0f}-{100 * bounds[1]:.0f}]" if bounds else f"{value}"


def _print_summary(summary: dict) -> None:
    print(f"\n{summary['clips']} clips ({summary['clips_failed']} failed), {summary['labeled_shots']} labeled shots, "
          f"{summary['predicted_shots']} predicted, {summary['matched']} matched")
    for key in ("primary", "detection", "outcome", "shot_type", "distance_ft", "zone"):
        print(f"  {key}: " + ", ".join(f"{name} {_with_interval(value)}" for name, value in summary[key].items()))
    if "release_error_s_median" in summary:
        print(f"  median release-time error: {summary['release_error_s_median']} s")
    print("  [low-high]: 95% Wilson interval, in percent")


def _print_shots(rows: list[dict]) -> None:
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
    parser.add_argument("--court-landmarks", type=Path, default=EVAL_DIR / "court_landmarks.json",
                        help="automatic court calibrations for jobs without landmarks in settings.json "
                             "(scripts/auto_court_landmarks.py); used by --rerun and --replay")
    parser.add_argument("--tolerance", type=float, default=.75,
                        help="max release-time gap to match a shot (and to pair false shots in a comparison), seconds")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "report.json")
    parser.add_argument("--summary-only", action="store_true",
                        help="print only the summary, any failed clips and the --compare result, and leave the "
                             "per-shot and per-game sections out of the report; score holdouts this way")
    parser.add_argument("--compare", type=Path, metavar="REPORT",
                        help="pair this run's shots with an earlier report.json's and print each metric's gains, "
                             "losses and exact McNemar p value")
    parser.add_argument("--compare-reports", type=Path, nargs=2, metavar=("BEFORE", "AFTER"),
                        help="compare two saved reports the same way, scoring nothing")
    args = parser.parse_args(argv)
    if args.compare_reports:
        if args.compare:
            parser.error("choose --compare or --compare-reports")
        before, after = (_read_report(path, parser) for path in args.compare_reports)
        _print_paired(compare(before, after, args.tolerance), *map(str, args.compare_reports))
        return 0
    earlier = _read_report(args.compare, parser) if args.compare else None
    try:
        labels = read_labels(args.labels)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not labels:
        parser.error(f"no labels in {args.labels}")
    if args.rerun and args.replay:
        parser.error("choose --rerun or --replay")
    how = "rerun" if args.rerun else "replay" if args.replay else "saved"
    clips = evaluate(labels, args.jobs, args.tolerance, args.runs, how, args.workers if args.replay else 1,
                     args.court_landmarks)
    summary, rows = summarize(clips), shot_rows(clips)
    _print_summary(summary)
    if not args.summary_only:
        _print_shots(rows)
    elif failed := [row for row in rows if "label_release_s" not in row]:
        # Which clips failed, and why, says nothing about their shots.
        print("\nfailed clips:\n" + "\n".join(f"  {row['job_id']}: {row['problem']}" for row in failed))
    report = {"summary": summary}
    if earlier is not None:
        report["paired"] = {"against": str(args.compare),
                            **compare(earlier, {"shots": rows, "clips": clip_rows(clips)}, args.tolerance)}
        _print_paired(report["paired"], str(args.compare), "this run")
    if not args.summary_only:
        report.update(by_game=_by_game(clips), shots=rows, clips=clip_rows(clips))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


def _by_game(clips: list[ClipResult]) -> dict:
    games: dict[str, list[ClipResult]] = {}
    for clip in clips:
        games.setdefault(clip.game or "unknown", []).append(clip)
    by_game = {game: {key: value for key, value in summarize(group).items()
                      if key in {"clips", "labeled_shots", "predicted_shots", "matched", "detection", "outcome"}}
               for game, group in sorted(games.items())}
    for value in by_game.values():
        value["outcome"] = {key: value["outcome"][key] for key in ("accuracy_when_called", "misses_called_made")}
    return by_game


def _read_report(path: Path, parser: argparse.ArgumentParser) -> dict:
    """A saved report with its per-shot rows, to pair with."""
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        parser.error(f"{path}: {exc}")
    if "shots" not in report:
        parser.error(f"{path} has no per-shot rows (written with --summary-only?), so there is nothing to pair")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
