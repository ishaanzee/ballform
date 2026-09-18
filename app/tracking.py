"""Lightweight identity tracking and jersey-colour descriptors for game footage."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from app.models import PoseFrame


BODY_NAMES = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _visible(pose: PoseFrame, name: str) -> tuple[float, float] | None:
    point = pose.landmarks.get(name)
    if point is None or point[2] < .55 or not all(math.isfinite(v) for v in point):
        return None
    return point[0], point[1]


def body_geometry(pose: PoseFrame, aspect_ratio: float) -> tuple[tuple[float, float], float] | None:
    points = [_visible(pose, name) for name in BODY_NAMES]
    if any(point is None for point in points):
        return None
    ls, rs, lh, rh = points
    shoulder = ((ls[0] + rs[0]) * aspect_ratio / 2, (ls[1] + rs[1]) / 2)
    hip = ((lh[0] + rh[0]) * aspect_ratio / 2, (lh[1] + rh[1]) / 2)
    torso = math.dist(shoulder, hip)
    return (hip, torso) if torso >= .025 else None


def jersey_descriptor(frame: np.ndarray, pose: PoseFrame) -> tuple[float, float, float] | None:
    """Return robust normalized Lab colour from an inset torso polygon."""
    points = [_visible(pose, name) for name in BODY_NAMES]
    if any(point is None for point in points):
        return None
    height, width = frame.shape[:2]
    ordered = (points[0], points[1], points[3], points[2])
    raw = np.asarray([(point[0] * width, point[1] * height) for point in ordered], dtype=np.float32)
    center = raw.mean(axis=0)
    polygon = np.rint(center + .68 * (raw - center)).astype(np.int32)
    if abs(float(cv2.contourArea(polygon))) < 20:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0]
    if len(pixels) < 20:
        return None
    median = np.median(pixels, axis=0) / 255.
    return tuple(float(value) for value in median)


@dataclass
class _Track:
    track_id: int
    center: tuple[float, float]
    torso: float
    frame: int
    velocity: tuple[float, float] = (0., 0.)
    appearance: tuple[float, float, float] | None = None


class PoseTracker:
    """Assign short-lived IDs using motion, scale and jersey appearance."""

    def __init__(self, aspect_ratio: float, max_gap_frames: int = 20):
        self.aspect_ratio = aspect_ratio
        self.max_gap_frames = max_gap_frames
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, poses: list[PoseFrame]) -> None:
        if not poses:
            return
        frame = poses[0].frame
        self._tracks = {
            key: track for key, track in self._tracks.items()
            if frame - track.frame <= self.max_gap_frames
        }
        geometries = [body_geometry(pose, self.aspect_ratio) for pose in poses]
        edges: list[tuple[float, int, int]] = []
        for pose_index, (pose, geometry) in enumerate(zip(poses, geometries)):
            if geometry is None:
                continue
            center, torso = geometry
            for track_id, track in self._tracks.items():
                gap = max(1, frame - track.frame)
                predicted = (
                    track.center[0] + track.velocity[0] * gap,
                    track.center[1] + track.velocity[1] * gap,
                )
                scale = max(.025, (torso + track.torso) / 2)
                motion = math.dist(center, predicted) / scale
                scale_change = abs(math.log(max(torso, 1e-5) / max(track.torso, 1e-5)))
                appearance = 0.
                if pose.appearance is not None and track.appearance is not None:
                    appearance = math.dist(pose.appearance, track.appearance) / .25
                cost = motion + .45 * scale_change + .35 * appearance
                max_cost = min(3., .75 + .13 * gap)
                if cost <= max_cost:
                    edges.append((cost, pose_index, track_id))

        assigned_poses: set[int] = set()
        assigned_tracks: set[int] = set()
        for _, pose_index, track_id in sorted(edges):
            if pose_index in assigned_poses or track_id in assigned_tracks:
                continue
            self._assign(poses[pose_index], geometries[pose_index], self._tracks[track_id])
            assigned_poses.add(pose_index)
            assigned_tracks.add(track_id)

        for index, pose in enumerate(poses):
            if index in assigned_poses:
                continue
            geometry = geometries[index]
            if geometry is None:
                continue
            track = _Track(self._next_id, geometry[0], geometry[1], frame, appearance=pose.appearance)
            self._next_id += 1
            self._tracks[track.track_id] = track
            pose.track_id = track.track_id

    def _assign(self, pose: PoseFrame, geometry, track: _Track) -> None:
        center, torso = geometry
        gap = max(1, pose.frame - track.frame)
        observed_velocity = ((center[0] - track.center[0]) / gap, (center[1] - track.center[1]) / gap)
        track.velocity = (
            .55 * track.velocity[0] + .45 * observed_velocity[0],
            .55 * track.velocity[1] + .45 * observed_velocity[1],
        )
        track.center, track.torso, track.frame = center, torso, pose.frame
        if pose.appearance is not None:
            if track.appearance is None:
                track.appearance = pose.appearance
            else:
                track.appearance = tuple(
                    .75 * old + .25 * new for old, new in zip(track.appearance, pose.appearance)
                )
        pose.track_id = track.track_id
        pose.appearance = track.appearance
