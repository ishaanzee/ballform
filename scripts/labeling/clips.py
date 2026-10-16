#!/usr/bin/env python3
"""Cut a labeled clip into a new Ballform job, and check a clip for camera cuts.

    uv run --inexact python scripts/labeling/clips.py cut eval/staging/video/<id>.mp4 412.35 7.0
    uv run --inexact python scripts/labeling/clips.py cuts <job id or clip.mp4>

cut re-encodes exactly [start, start + duration) of the source to match the existing
broadcast clips (H.264 High, yuv420p, 1920x1080, CFR 59.94 fps, bt709, CRF 16, no audio)
into data/jobs/<token_hex(6)>/input.mp4 with the standard settings.json, and prints
the job id. Clip time 0 is the first source frame at or after start, so a time t in
the clip is source time start + t. --keep-source-format keeps a fixed camera's own
resolution and frame rate (never upscaled), for non-NBA footage.

cuts runs app/vision.py's CutDetector over every frame and a local frame-difference
spike test. A single bright frame (flash photography) differs from both neighbours
while the frames either side match each other; those are reported as flashes, not cuts.
"""
from __future__ import annotations

import argparse
import json
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JOBS_DIR = ROOT / "data" / "jobs"
SETTINGS = {"rim": None, "mode": "one_on_one", "handedness": "right", "camera": "moving", "court": None,
            "rim_frame": None, "rim_time_s": None, "pose_model": "yolo26s-pose", "court_landmarks": None}
CAMERAS = ("moving", "elevated", "courtside")


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames"
                          ":format=duration", "-of", "json", str(path)], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"ffprobe cannot read {path}: {out.stderr.strip()}")
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    stream["duration"] = float(data["format"].get("duration", 0))
    return stream


def resolve_clip(clip: str) -> Path:
    path = Path(clip)
    if path.exists():
        return path
    job = JOBS_DIR / clip / "input.mp4"
    if job.exists():
        return job
    raise SystemExit(f"no clip at {clip} and no job {clip} under data/jobs")


def cut(source: Path, start: float, duration: float, camera: str, keep_source: bool, crf: int) -> int:
    if not source.exists():
        raise SystemExit(f"no source video at {source}")
    info = probe(source)
    if start < 0 or start + duration > info["duration"] + 0.01:
        raise SystemExit(f"span {start}-{start + duration} s is outside the source (0-{info['duration']:.2f} s)")
    job = secrets.token_hex(6)
    while (JOBS_DIR / job).exists():
        job = secrets.token_hex(6)
    job_dir = JOBS_DIR / job
    job_dir.mkdir(parents=True)
    target = job_dir / "input.mp4"
    if keep_source:
        rate = info.get("r_frame_rate", "30/1")
        filters = ["fps=" + rate]
    else:
        rate = "60000/1001"
        filters = ["scale=1920:1080:flags=lanczos", "fps=" + rate]
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
           "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", ",".join(filters + ["format=yuv420p"]),
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-profile:v", "high",
           "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-fps_mode", "cfr", "-r", rate, "-movflags", "+faststart", str(target)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode or not target.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
        raise SystemExit(f"ffmpeg failed: {result.stderr.strip()}")
    settings = dict(SETTINGS, camera=camera)
    (job_dir / "settings.json").write_text(json.dumps(settings))
    out = probe(target)
    print(f"cut {source.name} {start:.3f}s +{duration:.3f}s -> {out['width']}x{out['height']} "
          f"{out['r_frame_rate']} fps, {out.get('nb_frames')} frames, {out['duration']:.3f} s")
    print(f"job {job}  ({target})")
    print(job)
    return 0


def small_frames(path: Path, start: float | None = None, duration: float | None = None):
    """Every frame as a small grayscale image, decoded by ffmpeg (fast). Returns (fps, first
    frame number, frames). With start, the first frame is the first at or after it."""
    import numpy as np
    info = probe(path)
    num, den = (info.get("r_frame_rate") or "60000/1001").split("/")
    fps = float(num) / float(den or 1)
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", f"{max(0., start):.3f}"]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-map", "0:v:0", "-vf", "scale=160:90:flags=area,format=gray", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 90, 160)
    first = int(round(max(0., start or 0.) * fps))
    return fps, first, frames


def analyze_cuts(frames, fps: float, first: int = 0, spike: float = 4.0, floor: float = 18.0) -> list[dict]:
    """Suspect frames from a run of small gray frames: CutDetector's confirmed cuts, plus
    difference spikes; one-frame bright flashes are labeled as such."""
    import numpy as np
    from app.vision import CutDetector
    detector = CutDetector()
    confirmed = set()
    small = [f.astype(np.float32) for f in frames]
    for i, f in enumerate(frames):
        found = detector.observe(first + i, f)
        if found is not None:
            confirmed.add(found)
    n = len(small)
    bright = [float(f.mean()) for f in small]
    diffs = [0.] + [float(np.mean(np.abs(small[i - 1] - small[i]))) for i in range(1, n)]
    rows = []
    for i in range(1, n):
        local = sorted(diffs[max(1, i - 15):i] + diffs[i + 1:i + 16]) or [0.]
        median = local[len(local) // 2]
        number = first + i
        is_detector = number in confirmed
        if not (is_detector or (diffs[i] > floor and diffs[i] > spike * max(median, 1.0))):
            continue
        skip = float(np.mean(np.abs(small[i - 1] - small[i + 1]))) if i + 1 < n else None
        flash = (skip is not None and skip < 0.5 * diffs[i] and bright[i] > bright[i - 1] + 2
                 and bright[i] > bright[i + 1] + 2)
        after_flash = rows and rows[-1]["frame"] == number - 1 and rows[-1]["verdict"].startswith("flash")
        if flash:
            verdict = "flash (single bright frame; not a cut)"
        elif after_flash:
            verdict = "flash end (frame after a flash)"
        elif is_detector:
            verdict = "CUT (CutDetector confirmed)"
        else:
            verdict = "suspect cut (difference spike; look at the frames)"
        rows.append({"frame": number, "time": round(number / fps, 3), "diff": round(diffs[i], 1),
                     "median": round(median, 1), "brightness": round(bright[i] - bright[i - 1], 1),
                     "verdict": verdict})
    return rows


def is_cut(row: dict) -> bool:
    return not row["verdict"].startswith("flash")


def cuts(clip: str) -> int:
    path = resolve_clip(clip)
    fps, first, frames = small_frames(path)
    rows = analyze_cuts(frames, fps)
    print(f"{path}: {len(frames)} frames at {fps:.3f} fps")
    real = [r for r in rows if is_cut(r)]
    for r in rows:
        print(f"  f{r['frame']:<4} {r['time']:7.3f}s  diff {r['diff']:5.1f} (local median {r['median']:.1f}), "
              f"brightness {r['brightness']:+5.1f}  {r['verdict']}")
    if not real:
        flashes = len(rows)
        print("no camera cuts" + (f" ({flashes} flash frame(s), harmless)" if flashes else ""))
        return 0
    print(f"{len(real)} possible cut(s): look at them with frames.py; recut the clip to avoid real ones")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cut", help="re-encode a span into a new job")
    c.add_argument("source", type=Path)
    c.add_argument("start_s", type=float)
    c.add_argument("duration_s", type=float)
    c.add_argument("--camera", choices=CAMERAS, default="moving",
                   help="settings.json camera: moving for broadcast; elevated or courtside for fixed cameras")
    c.add_argument("--keep-source-format", action="store_true",
                   help="keep the source resolution and frame rate (fixed-camera non-NBA footage)")
    c.add_argument("--crf", type=int, default=16)
    k = sub.add_parser("cuts", help="flag camera cuts and flashes in a clip")
    k.add_argument("clip", help="clip path or job id")
    args = parser.parse_args(argv)
    if args.command == "cut":
        return cut(args.source, args.start_s, args.duration_s, args.camera, args.keep_source_format, args.crf)
    return cuts(args.clip)


if __name__ == "__main__":
    sys.exit(main())
