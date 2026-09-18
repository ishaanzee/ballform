from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point = tuple[float, float]


def joint_angle(a: Point, b: Point, c: Point) -> float | None:
    """Return the smaller ABC angle in degrees."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def elevation_angle(a: Point, b: Point) -> float | None:
    """Angle of vector a->b above the image horizontal (positive is up)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if abs(dx) + abs(dy) < 1e-8:
        return None
    return math.degrees(math.atan2(-dy, abs(dx)))


def distance(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def median_smooth(values: Sequence[float | None], radius: int = 2) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        window = [v for v in values[max(0, i - radius) : i + radius + 1] if v is not None]
        out.append(float(np.median(window)) if window else None)
    return out


def interpolate_crossing(p1: Point, p2: Point, y: float) -> float | None:
    if (p1[1] - y) * (p2[1] - y) > 0 or abs(p2[1] - p1[1]) < 1e-8:
        return None
    alpha = (y - p1[1]) / (p2[1] - p1[1])
    if not 0 <= alpha <= 1:
        return None
    return p1[0] + alpha * (p2[0] - p1[0])

