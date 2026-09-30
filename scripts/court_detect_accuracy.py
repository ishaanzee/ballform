"""Compare automatic court proposals with the hand-marked calibrations of finished jobs.

For every job under data/jobs with a court calibration (court.json plus result.json),
the detector runs on the marked (anchor) frame of the job's video, and its homography
is compared with the one fitted to the user's clicks:

* pixels: how far apart the two mappings put the template lines and landmarks that fall
  inside the frame under the manual mapping (median, 90th percentile, maximum);
* feet: a floor point is mapped into the frame with the manual mapping and back to the
  court with the automatic one, for the shooter spots recorded in the result, the
  free-throw line centre and the top of the three-point arc.

The manual mappings are not ground truth: separate calibrations of the same frame are
compared with each other too, as a floor for what "agreement" can mean.

    uv run python scripts/court_detect_accuracy.py                       # every calibrated job
    uv run python scripts/court_detect_accuracy.py 5eff568bbf6f f16c7e018121 \\
        --input merged-3074-court=data/jobs/3074e08ada0f/input.mp4 --frames 4 --json out.json

--frames N also checks N later frames per job against the manual mapping as followed
through camera motion (court.json), which adds that tracking's drift to the reference.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.court import apply, template  # noqa: E402
from app.court_detect import _line_samples, propose  # noqa: E402

JOBS = ROOT / "data" / "jobs"


def _video(job: Path, overrides: dict[str, Path]) -> Path | None:
    if job.name in overrides:
        return overrides[job.name]
    return next((p for p in sorted(job.glob("input.*"))), None)


def _frame(video: Path, index: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = capture.read()
    capture.release()
    return image if ok else None


def _pixel_difference(manual: np.ndarray, auto: np.ndarray, standard: str, width: int, height: int) -> dict:
    court = template(standard)
    samples, _ = _line_samples(court, 1.)
    landmarks = np.asarray(list(court.landmarks.values()))
    out = {}
    for label, points in (("lines", samples), ("landmarks", landmarks)):
        a, b = apply(manual, points), apply(auto, points)
        inside = (np.all(np.isfinite(a), axis=1) & (a[:, 0] >= 0) & (a[:, 1] >= 0)
                  & (a[:, 0] < width) & (a[:, 1] < height))
        error = np.linalg.norm(a[inside] - b[inside], axis=1)
        error = error[np.isfinite(error)]
        out[label] = None if not len(error) else {
            "median_px": round(float(np.median(error)), 1), "p90_px": round(float(np.percentile(error, 90)), 1),
            "max_px": round(float(error.max()), 1), "points": int(len(error))}
    return out


def _feet_difference(manual: np.ndarray, auto: np.ndarray, point, width: int, height: int) -> dict:
    pixel = apply(manual, [point])[0]
    back = apply(np.linalg.inv(auto), [pixel])[0]
    inside = bool(np.all(np.isfinite(pixel)) and 0 <= pixel[0] < width and 0 <= pixel[1] < height)
    return {"court_ft": [round(float(v), 2) for v in point], "in_frame": inside,
            "difference_ft": round(float(np.linalg.norm(back - np.asarray(point))), 2) if np.all(np.isfinite(back)) else None}


def evaluate_job(job: Path, video: Path, frames: int) -> dict:
    result = json.loads((job / "result.json").read_text())
    court_json = json.loads((job / "court.json").read_text())
    summary = result["court_calibration"]
    standard = summary["standard"]
    by_frame = {f["frame"]: f for f in court_json["frames"]}
    anchor = summary["anchor_frame"]
    image = _frame(video, anchor)
    if image is None:
        return {"job": job.name, "error": f"could not read frame {anchor} of {video}"}
    height, width = image.shape[:2]
    manual = np.linalg.inv(np.asarray(by_frame[anchor]["image_to_court"]))
    started = time.perf_counter()
    proposal = propose(image, standard)
    seconds = time.perf_counter() - started
    row = {"job": job.name, "video": str(video.relative_to(ROOT) if video.is_relative_to(ROOT) else video),
           "anchor_frame": anchor, "standard": standard,
           "manual": {"points": summary["points"], "clicked_rms_px": summary["clicked_error_px"]["rms"],
                      "leave_one_out_rms_px": (summary.get("leave_one_out_error_px") or {}).get("rms")},
           "auto": {"ok": proposal.ok, "reason": proposal.reason, "confidence": round(proposal.confidence, 3),
                    "landmarks": len(proposal.points), "method": proposal.diagnostics.get("method"),
                    "seconds": round(seconds, 2)}}
    if proposal.court_to_image is None:
        return row
    auto = proposal.court_to_image
    row["pixels"] = _pixel_difference(manual, auto, standard, width, height)
    court = template(standard)
    points = {"free_throw_centre": (0., court.dims["ft_y"]),
              "three_top": court.landmarks["three_top"]}
    shooters = []
    for index, shot in enumerate(result.get("shots", [])):
        game = shot.get("game") or {}
        metrics = game.get("metrics") or {}
        if metrics.get("shooter_court_x_ft") is not None and metrics.get("shooter_court_y_ft") is not None:
            spot = (metrics["shooter_court_x_ft"], metrics["shooter_court_y_ft"])
            points[f"shooter_spot_{index + 1}"] = spot
            shooters.append((index + 1, spot, game.get("release_frame")))
    row["feet"] = {name: _feet_difference(manual, auto, p, width, height) for name, p in points.items()}
    # The shooter spot again on the release frame, against the manual mapping followed there.
    release = []
    for number, spot, frame in shooters:
        item = by_frame.get(frame)
        release_image = _frame(video, frame) if item and item.get("image_to_court") else None
        if release_image is None:
            continue
        reference = np.linalg.inv(np.asarray(item["image_to_court"]))
        found = propose(release_image, standard)
        entry = {"shot": number, "frame": frame, "ok": found.ok, "confidence": round(found.confidence, 3)}
        if found.court_to_image is not None:
            entry.update(_feet_difference(reference, found.court_to_image, spot, width, height))
        release.append(entry)
    row["shooter_spot_at_release"] = release
    if frames:
        tracked = [f for f in court_json["frames"] if f["status"] == "tracked" and f["image_to_court"]]
        picks = [tracked[i] for i in np.linspace(len(tracked) // frames, len(tracked) - 2, frames).astype(int)] if tracked else []
        later = []
        for item in picks:
            later_image = _frame(video, item["frame"])
            if later_image is None:
                continue
            reference = np.linalg.inv(np.asarray(item["image_to_court"]))
            later_proposal = propose(later_image, standard)
            entry = {"frame": item["frame"], "ok": later_proposal.ok, "confidence": round(later_proposal.confidence, 3)}
            if later_proposal.court_to_image is not None:
                entry["lines"] = _pixel_difference(reference, later_proposal.court_to_image, standard, width, height)["lines"]
            later.append(entry)
        row["later_frames"] = later
    return row


def manual_spread(jobs: list[Path], overrides: dict[str, Path]) -> list[dict]:
    """Pairwise differences between separate manual calibrations of the same video frame."""
    groups: dict[tuple, list[tuple[str, np.ndarray, str]]] = {}
    for job in jobs:
        video = _video(job, overrides)
        if video is None or not video.exists():
            continue
        summary = json.loads((job / "result.json").read_text())["court_calibration"]
        frames = {f["frame"]: f for f in json.loads((job / "court.json").read_text())["frames"]}
        digest = hashlib.sha256(video.read_bytes()).hexdigest()[:12]
        h = np.linalg.inv(np.asarray(frames[summary["anchor_frame"]]["image_to_court"]))
        groups.setdefault((digest, summary["anchor_frame"]), []).append((job.name, h, summary["standard"]))
    out = []
    for (digest, frame), members in groups.items():
        for (a, ha, standard), (b, hb, _) in itertools.combinations(members, 2):
            video = _video(next(j for j in jobs if j.name == a), overrides)
            image = _frame(video, frame)
            height, width = image.shape[:2]
            court = template(standard)
            out.append({"video_sha256": digest, "frame": frame, "jobs": [a, b],
                        "lines": _pixel_difference(ha, hb, standard, width, height)["lines"],
                        "free_throw_centre_ft": _feet_difference(ha, hb, (0., court.dims["ft_y"]), width, height)["difference_ft"],
                        "three_top_ft": _feet_difference(ha, hb, court.landmarks["three_top"], width, height)["difference_ft"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", nargs="*", help="job ids (default: every job with a court calibration)")
    parser.add_argument("--input", action="append", default=[], metavar="JOB=VIDEO",
                        help="video for a job whose directory has no input file")
    parser.add_argument("--frames", type=int, default=0, help="also check N later tracked frames per job")
    parser.add_argument("--json", type=Path, help="write the full results here")
    args = parser.parse_args()
    overrides = {}
    for item in args.input:
        name, _, path = item.partition("=")
        overrides[name] = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)
    if args.jobs:
        jobs = [JOBS / name for name in args.jobs]
    else:
        jobs = sorted(p.parent for p in JOBS.glob("*/court.json")
                      if (p.parent / "result.json").exists()
                      and json.loads((p.parent / "result.json").read_text()).get("court_calibration"))
    rows = []
    for job in jobs:
        video = _video(job, overrides)
        if video is None or not video.exists():
            rows.append({"job": job.name, "error": "no input video; pass --input JOB=VIDEO"})
            continue
        rows.append(evaluate_job(job, video, args.frames))
    spread = manual_spread([j for j in jobs if _video(j, overrides)], overrides)

    print("| job | manual clicks (RMS px) | auto | lines median / p90 / max px | landmarks median px | "
          "FT centre ft | arc top ft | shooter spot ft |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        if "error" in row:
            print(f"| {row['job']} | | {row['error']} | | | | | |")
            continue
        auto = row["auto"]
        status = (f"ok, conf {auto['confidence']:.2f}" if auto["ok"]
                  else f"rejected (conf {auto['confidence']:.2f})")
        manual = f"{row['manual']['points']} ({row['manual']['clicked_rms_px']})"
        if "pixels" not in row:
            print(f"| {row['job']} | {manual} | {status} | | | | | |")
            continue
        lines, marks = row["pixels"]["lines"], row["pixels"]["landmarks"]
        feet = row["feet"]

        def ft(name):
            item = feet.get(name)
            if not item or item["difference_ft"] is None:
                return "–"
            return f"{item['difference_ft']:.2f}" + ("" if item["in_frame"] else " (off frame)")

        at_release = [f"{e['difference_ft']:.2f} at release{'' if e['ok'] else ' (rejected)'}"
                      for e in row.get("shooter_spot_at_release", []) if e.get("difference_ft") is not None]
        shooters = ", ".join([ft(k) for k in feet if k.startswith("shooter_spot")] + at_release) or "–"
        print(f"| {row['job']} | {manual} | {status} | {lines['median_px']} / {lines['p90_px']} / {lines['max_px']} | "
              f"{marks['median_px'] if marks else '–'} | {ft('free_throw_centre')} | {ft('three_top')} | {shooters} |")
        for later in row.get("later_frames", []):
            lines = later.get("lines")
            print(f"|   frame {later['frame']} | | {'ok' if later['ok'] else 'rejected'} "
                  f"(conf {later['confidence']:.2f}) | "
                  f"{'–' if not lines else f'{lines['median_px']} / {lines['p90_px']} / {lines['max_px']}'} | | | | |")
    if spread:
        print("\nManual against manual, same frame (lines median / p90 / max px; floor points in feet):")
        for item in spread:
            lines = item["lines"]
            print(f"  {item['jobs'][0]} vs {item['jobs'][1]}: {lines['median_px']} / {lines['p90_px']} / {lines['max_px']}"
                  f"; FT centre {item['free_throw_centre_ft']} ft, arc top {item['three_top_ft']} ft")
    if args.json:
        args.json.write_text(json.dumps({"jobs": rows, "manual_spread": spread}, indent=2))


if __name__ == "__main__":
    main()
