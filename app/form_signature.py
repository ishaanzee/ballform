"""Pose-only signature of a set shot (a free throw) filmed from the side.

A rep is found from the shooting arm alone: the wrist rises above the nose while the elbow
opens from its most folded angle (the set point) to its straightest (the release). No ball
or rim is needed, so the same code measures a phone clip and a broadcast crop, and the two
are compared feature by feature in app/form_match.py.

Lengths are in torso lengths (mid-shoulder to mid-hip), angles in degrees in the image
plane, times in seconds. They are comparable between clips only when both are side views.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from app.models import PoseFrame

JOINTS = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
# feature -> (label, unit)
FEATURES = {
    "set_elbow_deg": ("Elbow bend at the set point", "°"),
    "set_height": ("Set point height above the nose", "torso"),
    "set_forward": ("Set point distance in front of the face", "torso"),
    "release_elbow_deg": ("Elbow extension at release", "°"),
    "release_height": ("Release height above the nose", "torso"),
    "release_arm_deg": ("Arm angle at release", "°"),
    "extend_time_s": ("Set point to release", "s"),
    "lift_time_s": ("Waist to release", "s"),
    "knee_bend_deg": ("Deepest knee bend", "°"),
    "knee_lead_s": ("Knee bend before release", "s"),
    "trunk_lean_deg": ("Forward lean at release", "°"),
    "follow_hold_s": ("Follow-through hold", "s"),
}
SKELETON_TIMES = tuple(round(-0.8 + 0.05 * i, 2) for i in range(25))  # seconds from release
MIN_EXTENSION = 40.0   # degrees the elbow must open between set point and release
MIN_RELEASE_ELBOW = 135.0
REOPEN_DEG = 12.0       # elbow opening, walking back in time, that ends the search for the set point
LOOKBACK_S = 0.7       # set point is searched this long before the wrist passes the nose
MAX_HOLD_S = 2.0
MAX_LEAN_DEG, MIN_KNEE_DEG, MIN_RAISED_S = 30.0, 100.0, 0.2
MIN_EXTEND_S = 0.08     # an elbow cannot open 40 degrees faster; a quicker one is a pose jump


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    u, w = a - b, c - b
    norm = np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arccos(np.clip(np.sum(u * w, axis=1) / norm, -1, 1)))


def _smooth(values: np.ndarray, radius: int = 1) -> np.ndarray:
    """Running median over time; it removes single-frame pose jumps and keeps ramps. A wider
    window would flatten the set point, the sharpest turn of the elbow."""
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        window = values[max(0, i - radius): i + radius + 1]
        if np.isfinite(window).any():
            out[i] = np.nanmedian(window, axis=0)
    return out


class Series:
    """Smoothed per-frame measurements of one shooter."""

    def __init__(self, poses: Sequence[PoseFrame], aspect_ratio: float, handedness: str):
        poses = sorted(poses, key=lambda p: p.time_s)
        self.t = np.asarray([p.time_s for p in poses], dtype=float)
        raw = {name: np.asarray([[p.landmarks[name][0] * aspect_ratio, p.landmarks[name][1]]
                                 if name in p.landmarks else [np.nan, np.nan] for p in poses], dtype=float).reshape(-1, 2)
               for name in JOINTS}
        self.xy = {name: _smooth(points) for name, points in raw.items()}
        xy, side = self.xy, handedness
        self.shoulder = (xy["left_shoulder"] + xy["right_shoulder"]) / 2
        self.hip = (xy["left_hip"] + xy["right_hip"]) / 2
        self.ankle = (xy["left_ankle"] + xy["right_ankle"]) / 2
        self.torso = np.linalg.norm(self.shoulder - self.hip, axis=1)
        wrist = xy[f"{side}_wrist"]
        self.elbow = _angle(xy[f"{side}_shoulder"], xy[f"{side}_elbow"], wrist)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.wrist_h = (xy["nose"][:, 1] - wrist[:, 1]) / self.torso
            self.wrist_dx = (wrist[:, 0] - xy["nose"][:, 0]) / self.torso
            self.arm = wrist - xy[f"{side}_shoulder"]
            self.wrist_over_shoulder = (xy[f"{side}_shoulder"][:, 1] - wrist[:, 1]) / self.torso
            self.trunk_dx = (self.shoulder[:, 0] - self.hip[:, 0]) / self.torso
            self.trunk_dy = (self.hip[:, 1] - self.shoulder[:, 1]) / self.torso
            self.shoulder_width = np.linalg.norm(xy["left_shoulder"] - xy["right_shoulder"], axis=1) / self.torso
        knees = (xy["left_knee"] + xy["right_knee"]) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            self.forward = ((xy["nose"][:, 0] - self.shoulder[:, 0])
                            + (knees[:, 0] - (self.hip[:, 0] + self.ankle[:, 0]) / 2)) / self.torso
        self.knee = np.nanmean(np.stack([_angle(xy[f"{s}_hip"], xy[f"{s}_knee"], xy[f"{s}_ankle"])
                                         for s in ("left", "right")]), axis=0) if len(poses) else np.zeros(0)

    def between(self, start: float, end: float) -> np.ndarray:
        return np.flatnonzero((self.t >= start) & (self.t <= end))


def _crossing(t: np.ndarray, values: np.ndarray, start: int, end: int, level: float) -> float:
    """First time ``values`` rises through ``level`` between two frames, interpolated."""
    for i in range(start, end + 1):
        if values[i] >= level:
            if i > start and np.isfinite(values[i - 1]) and values[i] > values[i - 1]:
                return float(t[i - 1] + (level - values[i - 1]) / (values[i] - values[i - 1]) * (t[i] - t[i - 1]))
            return float(t[i])
    return float(t[end])


def _reps(series: Series) -> list[tuple[int, float, int]]:
    """(set-point frame index, release time, last frame index of the raised arm) per rep."""
    t, raised = series.t, np.nan_to_num(series.wrist_h, nan=-9) > 0
    runs: list[list[int]] = []
    for i in np.flatnonzero(raised):
        if runs and t[i] - t[runs[-1][1]] <= 0.3:   # a short dropout does not end the raise
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    reps = []
    for start, end in runs:
        if t[end] - t[start] < 0.1:
            continue
        # Straightest while raised, then most folded before that. An arm hanging straight
        # before the lift must not be taken for the release.
        up = series.between(t[start], min(t[end], t[start] + 0.6))
        if not np.isfinite(series.elbow[up]).any():
            continue
        top = int(up[np.nanargmax(series.elbow[up])])
        # The set point is the fold the extension starts from: walk back from the release
        # until the elbow opens again or the wrist is back below the shoulder. The elbow is
        # also folded while the ball is held at the waist, and some shooters fold it further
        # all the way up from there; neither is the set point.
        low, first = top, int(series.between(t[start] - LOOKBACK_S, t[top])[0])
        for i in range(top - 1, first - 1, -1):
            if series.wrist_over_shoulder[i] < 0:
                break
            if series.elbow[i] < series.elbow[low]:
                low = i
            elif series.elbow[i] > series.elbow[low] + REOPEN_DEG:
                break
        lo, hi = series.elbow[low], series.elbow[top]
        if hi < MIN_RELEASE_ELBOW or hi - lo < MIN_EXTENSION:
            continue
        release = _crossing(t, series.elbow, low, top, lo + 0.9 * (hi - lo))
        if reps and release - reps[-1][1] < 1.0:
            continue
        reps.append((low, release, end))
    return reps


def _signature(series: Series, low: int, release: float, end: int, handedness: str) -> dict:
    t = series.t
    after = series.between(release, release + 0.25)
    lift = series.between(t[low] - 0.1, release + 0.25)
    if len(after) == 0 or np.isfinite(series.torso[lift]).mean() < 0.8:
        return {}
    # The shooter faces the way the nose sits from the shoulders and the bent knees from the
    # hip-ankle line. The wrist does not say: a high set point is over or behind the head.
    facing = 1.0 if np.nanmedian(series.forward[lift]) >= 0 else -1.0
    lo, hi = float(series.elbow[low]), float(np.nanmax(series.elbow[after]))
    # The elbow stays near its smallest angle for several frames while the ball still rises,
    # so the set point is read where the elbow starts to open, not at the smallest angle.
    opening = _crossing(t, series.elbow, low, int(after[-1]), lo + 0.1 * (hi - lo))
    at_set = lambda values: float(np.interp(opening, t, values))  # noqa: E731
    arm = np.nanmedian(series.arm[series.between(release, release + 0.15)], axis=0)
    legs = series.between(release - 0.8, release)
    knee_at = int(legs[np.nanargmin(series.knee[legs])]) if np.isfinite(series.knee[legs]).any() else None
    near = series.between(release - 0.05, release + 0.05)
    below = [i for i in series.between(release - 1.5, t[low]) if series.wrist_h[i] < -0.7]
    hold_end = min(t[end], release + MAX_HOLD_S)
    values = {
        "set_elbow_deg": lo,
        "set_height": at_set(series.wrist_h),
        "set_forward": at_set(series.wrist_dx) * facing,
        "release_elbow_deg": hi,
        "release_height": float(np.nanmax(series.wrist_h[after])),
        "release_arm_deg": math.degrees(math.atan2(-arm[1], arm[0] * facing)),
        "extend_time_s": _crossing(t, series.elbow, low, int(after[-1]), lo + 0.9 * (hi - lo)) - opening,
        "lift_time_s": release - float(t[below[-1]]) if below else None,
        "knee_bend_deg": float(series.knee[knee_at]) if knee_at is not None else None,
        "knee_lead_s": release - float(t[knee_at]) if knee_at is not None else None,
        "trunk_lean_deg": math.degrees(math.atan2(float(np.nanmedian(series.trunk_dx[near])) * facing,
                                                  float(np.nanmedian(series.trunk_dy[near])))) if len(near) else None,
        # A hold cut off by the end of the clip (or the cap) is a lower bound, not a measurement.
        "follow_hold_s": hold_end - release if hold_end < min(t[-1] - 0.1, release + MAX_HOLD_S) else None,
    }
    # An arm that ends below its set point or pointing down was lost by the pose model at release.
    if not values["release_height"] > values["set_height"] or not values["release_arm_deg"] > 0:
        return {}
    # A pose that jumps for a few frames can look like a lift: a real shooter stands upright
    # on legs that are not folded, and keeps the arm up for a moment after the release.
    lean, knee = values["trunk_lean_deg"], values["knee_bend_deg"]
    if (lean is None or abs(lean) > MAX_LEAN_DEG or knee is None or knee < MIN_KNEE_DEG
            or t[end] - release < MIN_RAISED_S or values["extend_time_s"] < MIN_EXTEND_S):
        return {}
    features = {k: round(float(v), 3) if v is not None and math.isfinite(v) else None for k, v in values.items()}
    # Seen from the shooter's right when facing image-right. The far arm passes behind the
    # head and the two sides are different camera angles, so they are not compared.
    near = (facing > 0) == (handedness == "right")
    return {"release_s": round(release, 3), "set_s": round(opening, 3), "features": features,
            "arm_side": "near" if near else "far",
            "shoulder_width": round(float(np.nanmedian(series.shoulder_width[lift])), 3),
            "skeleton": _skeleton(series, release, facing)}


def _skeleton(series: Series, release: float, facing: float) -> list[dict | None]:
    """Joint positions around the release: origin between the ankles at release, x toward
    the basket, y up, in torso lengths. None where the clip has no frame near that time."""
    t = series.t
    at = int(np.argmin(np.abs(t - release)))
    origin, scale = series.ankle[at], np.nanmedian(series.torso[series.between(release - 0.5, release + 0.2)])
    if not np.isfinite(origin).all() or not scale > 0:
        return []
    frames = []
    for offset in SKELETON_TIMES:
        when = release + offset
        if when < t[0] - 0.02 or when > t[-1] + 0.02:
            frames.append(None)
            continue
        joints = {}
        for name in JOINTS:
            x, y = (np.interp(when, t, series.xy[name][:, k]) for k in (0, 1))
            joints[name] = [round(float((x - origin[0]) * facing / scale), 3), round(float((origin[1] - y) / scale), 3)]
        frames.append(joints if all(math.isfinite(v) for p in joints.values() for v in p) else None)
    return frames


def shot_signatures(poses: Sequence[PoseFrame], aspect_ratio: float = 1.0, handedness: str = "right") -> list[dict]:
    """One signature per set shot found in a single shooter's poses, in time order."""
    if handedness not in {"left", "right"}:
        raise ValueError("handedness must be left or right")
    if len(poses) < 5:
        return []
    series = Series(poses, aspect_ratio, handedness)
    return [s for s in (_signature(series, *rep, handedness) for rep in _reps(series)) if s]
