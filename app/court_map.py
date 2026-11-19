"""Top-down court map of a calibrated game analysis: where each tracked player stood on the
floor, frame by frame, for the results page to play beside the review video.

Everything comes from files the analysis already wrote: the smoothed floor trajectories in
court.json (app/trajectory.py), jersey colours in observations.json and the shots in
result.json. Nothing is tracked or measured here.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CACHE = "court_map.json"


def _hex(lab: np.ndarray) -> str:
    """OpenCV 8-bit Lab scaled to 0-1 (app/tracking.jersey_descriptor) -> #rrggbb."""
    pixel = np.clip(np.rint(lab * 255), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = (int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_LAB2BGR)[0, 0])
    return f"#{r:02x}{g:02x}{b:02x}"


def jersey_colours(observations: dict) -> dict[int, str]:
    """Median jersey colour of each track over the frames it was seen in."""
    seen: dict[int, list] = defaultdict(list)
    for frame in observations.get("players", []):
        for player in frame.get("players", []):
            if player.get("track_id") is not None and player.get("appearance"):
                seen[player["track_id"]].append(player["appearance"])
    return {track: _hex(np.median(np.asarray(colours, dtype=float), axis=0)) for track, colours in seen.items()}


def toward_camera(court: dict, width: float, height: float) -> list[float] | None:
    """Unit vector in court feet pointing from the floor in view toward the camera: the way
    the floor runs down the middle of the picture, on the median over mapped frames. The map
    is turned so this points down and the court faces the way it does in the video."""
    steps = []
    for frame in [f for f in court.get("frames", []) if f.get("reliable") and f.get("image_to_court")][::10]:
        points = cv2.perspectiveTransform(np.asarray([[[width / 2, 0.65 * height], [width / 2, 0.95 * height]]], dtype=float),
                                          np.asarray(frame["image_to_court"], dtype=float))[0]
        step = points[1] - points[0]
        if np.isfinite(step).all() and np.linalg.norm(step) > 1e-6:
            steps.append(step / np.linalg.norm(step))
    if not steps:
        return None
    mean = np.median(np.asarray(steps), axis=0)
    return [round(float(v), 3) for v in mean / np.linalg.norm(mean)] if np.linalg.norm(mean) > 1e-6 else None


def build(court: dict, observations: dict, result: dict) -> dict:
    """The court-map payload: per track its frames and x, y in court feet (x across the court
    from the lane's centre line, y from the baseline), and per shot who shot and who defended."""
    colours = jersey_colours(observations)
    tracks = [{"id": t["track_id"], "colour": colours.get(t["track_id"]), "frames": t["frames"], "x": t["x"], "y": t["y"]}
              for t in court.get("trajectories", {}).get("tracks", []) if t["frames"]]
    shots = []
    for shot in result.get("shots", []):
        players = (shot.get("game") or {}).get("players") or {}
        shots.append({"number": shot.get("number"), "start_s": shot.get("start_s"), "release_s": shot.get("release_s"),
                      "end_s": shot.get("end_s"), "outcome": shot.get("outcome"),
                      "shooter": players.get("shooter_track_id"), "defender": players.get("defender_track_id")})
    frames = court.get("frames", [])
    return {"standard": court.get("summary", {}).get("standard", "nba"),
            "fps": observations.get("fps") or (result.get("video") or {}).get("fps") or 30.0,
            "toward_camera": toward_camera(court, observations.get("width") or 1920, observations.get("height") or 1080),
            "frames": len(frames), "mapped_frames": sum(1 for f in frames if f.get("reliable")),
            "tracks": tracks, "shots": shots}


def load(job_dir: Path) -> dict | None:
    """The court map of a finished job, or None when it was analysed without court calibration.
    Built once and kept beside the job's other files."""
    cache, sources = job_dir / CACHE, [job_dir / name for name in ("court.json", "observations.json", "result.json")]
    if not all(path.exists() for path in sources):
        return None
    if cache.exists() and cache.stat().st_mtime >= max(path.stat().st_mtime for path in sources):
        return json.loads(cache.read_text())
    payload = build(*(json.loads(path.read_text()) for path in sources))
    cache.write_text(json.dumps(payload, separators=(",", ":")))
    return payload
