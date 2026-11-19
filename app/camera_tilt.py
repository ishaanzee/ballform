"""Camera roll from the upright edges of the scene: walls, posts, door frames, the backboard.

A phone that is not level turns every angle measured against the image's vertical. The
shooter cannot be the reference, because how upright the shooter stands is one of the
measurements. Straight edges behind the shooter can: most long near-vertical edges in a gym
or on an outdoor court are truly vertical.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from app.models import PoseFrame

WORK_HEIGHT = 540
MAX_EDGE_DEG = 20.0     # edges further from the image's vertical are not candidates
MIN_EDGE = 0.06         # shortest edge used, in frame heights
INLIER_DEG = 1.5
MIN_INLIER_LENGTH = 0.6  # frame heights of agreeing edges per sampled frame
MIN_INLIER_SHARE = 0.5   # of all candidate edge length
MAX_TILT_DEG = 12.0
MIN_SPREAD = 0.15        # frame heights; edges closer together than this say nothing about convergence
MAX_CONVERGENCE = 8.0    # degrees of lean per frame height across the frame: the camera points steeply up or down
PLACE_WIDTH, MIN_PLACES = 0.1, 3   # agreeing edges must come from this many places across the frame


def upright_edges(frame: np.ndarray, exclude: Sequence[tuple[float, float, float, float]] = (),
                  below: float | None = None) -> np.ndarray:
    """Near-vertical straight edges as rows (u, lean, length): u is the edge's horizontal
    distance from the frame centre in frame heights, lean its angle in degrees from the
    image's vertical (positive when its top is to the right), length in frame heights.
    Edges inside the ``exclude`` boxes (x0, y0, x1, y1 as fractions of the frame) are left
    out, and so are edges reaching under ``below`` (a fraction of frame height): the floor's
    lines run away from the camera and look upright without being so."""
    scale = WORK_HEIGHT / frame.shape[0]
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    rows = []
    for x0, y0, x1, y1 in ([] if found is None else found.reshape(-1, 4)):
        if y0 < y1:   # (x1, y1) is the top end
            x0, y0, x1, y1 = x1, y1, x0, y0
        length = math.hypot(x1 - x0, y1 - y0) / height
        lean = math.degrees(math.atan2(x1 - x0, y0 - y1))
        if length < MIN_EDGE or abs(lean) > MAX_EDGE_DEG:
            continue
        mx, my = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        if (below is not None and y0 / height > below) or any(a <= mx <= c and b <= my <= d for a, b, c, d in exclude):
            continue
        rows.append(((x0 + x1) / 2 / height - width / height / 2, lean, length))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def tilt_from_edges(edges: np.ndarray, frames: int = 1) -> float | None:
    """Degrees the scene's vertical leans to the right in the image, or None when the edges
    do not settle it. A camera pointing up or down makes upright edges lean apart or
    together across the frame, so the lean is fitted as a line across the frame and read at
    the centre. Edges on one side only cannot say what the centre reads and give None."""
    if len(edges) < 4:
        return None
    u, lean, weight = edges.T
    order = np.argsort(lean)
    centre = float(lean[order][np.searchsorted(np.cumsum(weight[order]), weight.sum() / 2)])
    slope = 0.0
    for _ in range(8):
        inside = np.abs(lean - centre - slope * u) <= INLIER_DEG
        if weight[inside].sum() <= 0:
            return None
        w, x, y = weight[inside], u[inside], lean[inside]
        mean_x, mean_y = np.average(x, weights=w), np.average(y, weights=w)
        spread = np.average((x - mean_x) ** 2, weights=w)
        # Shrunk toward no convergence where the edges sit too close together to show one.
        slope = float(np.average((x - mean_x) * (y - mean_y), weights=w) / (spread + MIN_SPREAD ** 2))
        centre = float(mean_y - slope * mean_x)
    agreeing = float(weight[inside].sum())
    places = len(set(np.round(x / PLACE_WIDTH).astype(int)))
    if (agreeing / frames < MIN_INLIER_LENGTH or agreeing / weight.sum() < MIN_INLIER_SHARE or places < MIN_PLACES
            or abs(mean_x) > math.sqrt(spread + MIN_SPREAD ** 2)   # the centre would be an extrapolation
            or abs(slope) > MAX_CONVERGENCE or abs(centre) > MAX_TILT_DEG):
        return None
    return round(centre, 2)


def _person_box(pose: PoseFrame, pad: float = 0.25) -> tuple[float, float, float, float] | None:
    points = np.asarray([p[:2] for p in pose.landmarks.values()], dtype=float)
    if len(points) < 4:
        return None
    (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
    grow = pad * (y1 - y0)
    return x0 - grow, y0 - grow, x1 + grow, y1 + grow


def _floor_line(pose: PoseFrame) -> float | None:
    hips = [pose.landmarks[name][1] for name in ("left_hip", "right_hip") if name in pose.landmarks]
    return float(np.mean(hips)) if hips else None


def camera_tilt(video: Path, poses: Sequence[PoseFrame], samples: int = 8) -> float | None:
    """Camera roll of a clip in degrees (see tilt_from_edges), from frames spread over the
    clip. The shooter is masked out and only edges above the hips are used."""
    if not poses:
        return None
    chosen = [poses[round(i)] for i in np.linspace(0, len(poses) - 1, min(samples, len(poses)))]
    capture = cv2.VideoCapture(str(video))
    edges, used = [], 0
    try:
        for pose in chosen:
            capture.set(cv2.CAP_PROP_POS_FRAMES, pose.frame)
            ok, frame = capture.read()
            box = _person_box(pose)
            if not ok or box is None:
                continue
            edges.append(upright_edges(frame, [box], _floor_line(pose)))
            used += 1
    finally:
        capture.release()
    return tilt_from_edges(np.concatenate(edges), used) if used else None
