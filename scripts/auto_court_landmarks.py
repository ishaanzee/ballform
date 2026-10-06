"""Calibrate the labeled clips' courts automatically, for distance scoring in the eval.

For each job in the labels file(s), auto-detect (`app/court_detect.py`) runs on frames
sampled through the clip. A broadcast camera pans and zooms from a fixed spot, so correct
fits of different frames recover the same camera position while wrong fits scatter;
`pick_calibration` keeps the frame whose fit agrees with the most others and accepts it
only when enough frames agree (see its docstring for the gate). Accepted landmarks are
written to eval/court_landmarks.json keyed by job id, in the settings.json
`court_landmarks` format; `ballform-eval --replay` uses them for jobs whose settings have
none. Rejected jobs are written too, with the reason.

    uv run python scripts/auto_court_landmarks.py                     # eval/labels.csv
    uv run python scripts/auto_court_landmarks.py --labels eval/holdout.csv --step-s .5
    uv run python scripts/auto_court_landmarks.py --cache /tmp/proposals.json  # reuse proposals

Only the video is read; labels supply job ids, never shot times or positions.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.court_detect import pick_calibration, propose  # noqa: E402

JOBS = ROOT / "data" / "jobs"
OUT = ROOT / "eval" / "court_landmarks.json"


def job_ids(paths: list[Path]) -> list[str]:
    ids: list[str] = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                job = (row.get("job_id") or "").strip()
                if job and not job.startswith("#") and job not in ids:
                    ids.append(job)
    return ids


def proposals(job: str, jobs_dir: Path, step_s: float, standard: str) -> dict:
    """Auto-detect on every step_s seconds of the job's video."""
    video = next(iter(sorted((jobs_dir / job).glob("input.*"))), None)
    if video is None:
        return {"job": job, "error": "no input video"}
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(step_s * fps))
    found, frame = [], 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if frame % step == 0:
            proposal = propose(image, standard).to_dict()
            proposal["frame"] = frame
            found.append(proposal)
        frame += 1
    capture.release()
    return {"job": job, "width": width, "height": height, "fps": fps, "frames": frame, "proposals": found}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, action="append", help="labels CSV (default eval/labels.csv)")
    parser.add_argument("--jobs", type=Path, default=JOBS)
    parser.add_argument("--out", type=Path, default=OUT, help="landmarks file to update (default eval/court_landmarks.json)")
    parser.add_argument("--step-s", type=float, default=.5, help="seconds between the frames tried")
    parser.add_argument("--standard", default="nba")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, help="JSON of raw proposals per job: read if present, else written")
    args = parser.parse_args()
    ids = job_ids(args.labels or [ROOT / "eval" / "labels.csv"])

    cache = json.loads(args.cache.read_text()) if args.cache and args.cache.exists() else {}
    todo = [job for job in ids if job not in cache]
    with ProcessPoolExecutor(args.workers) as pool:
        for done in pool.map(proposals, todo, [args.jobs] * len(todo), [args.step_s] * len(todo),
                             [args.standard] * len(todo)):
            cache[done["job"]] = done
            print(f"  {done['job']}: {sum(p['ok'] for p in done.get('proposals', []))} of "
                  f"{len(done.get('proposals', []))} frames proposed", file=sys.stderr, flush=True)
            if args.cache:
                args.cache.write_text(json.dumps(cache))

    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    accepted = 0
    for job in ids:
        item = cache[job]
        if "error" in item:
            out[job] = {"rejected": item["error"]}
            continue
        picked = pick_calibration(item["proposals"], item["width"], item["height"], args.standard)
        out[job] = picked
        accepted += "court_landmarks" in picked
        print(f"{job}: " + (f"frame {picked['court_landmarks']['frame']}, {picked['check']}" if "court_landmarks" in picked
                            else f"rejected: {picked['rejected']}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # One job per line keeps the file small and its diffs readable.
    args.out.write_text("{\n" + ",\n".join(f"{json.dumps(job)}: {json.dumps(out[job])}" for job in sorted(out)) + "\n}\n")
    print(f"{accepted} of {len(ids)} clips calibrated; wrote {args.out}")


if __name__ == "__main__":
    main()
