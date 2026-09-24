"""Lightweight identity tracking and jersey-colour descriptors for game footage."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from app.models import Detection, PoseFrame


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
    return (hip, torso) if torso >= .008 else None


def jersey_descriptor(frame: np.ndarray, pose: PoseFrame) -> tuple[float, float, float] | None:
    """Return dominant jersey colour, reducing contamination from numbers and skin."""
    points = [_visible(pose, name) for name in BODY_NAMES]
    if any(point is None for point in points):
        return None
    height, width = frame.shape[:2]
    ordered = (points[0], points[1], points[3], points[2])
    raw = np.asarray([(point[0] * width, point[1] * height) for point in ordered], dtype=np.float32)
    center = raw.mean(axis=0)
    polygon = np.rint(center + .90 * (raw - center)).astype(np.int32)
    if abs(float(cv2.contourArea(polygon))) < 20:
        return None
    x, y, w, h = cv2.boundingRect(polygon)
    x, y = max(0, x), max(0, y)
    w, h = min(w, width-x), min(h, height-y)
    if w < 2 or h < 2:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon - (x, y), 255)
    lab = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0]
    if len(pixels) < 20:
        return None
    values = pixels[::max(1, len(pixels)//512)].astype(float)
    centers = [values.mean(axis=0)]
    for _ in range(2):
        distances = np.min(np.linalg.norm(values[:, None] - np.asarray(centers), axis=2), axis=1)
        centers.append(values[int(np.argmax(distances))])
    centers = np.asarray(centers)
    for _ in range(6):
        labels = np.argmin(np.linalg.norm(values[:, None] - centers, axis=2), axis=1)
        for k in range(3):
            if np.any(labels == k):
                centers[k] = values[labels == k].mean(axis=0)
    dominant = int(np.argmax(np.bincount(labels, minlength=3)))
    # Large jersey lettering often occupies the center of the torso. A substantial
    # saturated fabric cluster is more useful than its white/gold number.
    counts = np.bincount(labels, minlength=3) / len(labels)
    chroma = np.linalg.norm(centers[:, 1:] - 128, axis=1)
    fabric = [k for k in range(3) if counts[k] >= .20 and chroma[k] > 40]
    if fabric:
        dominant = max(fabric, key=lambda k: counts[k])
    median = np.median(values[labels == dominant], axis=0) / 255.
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

    def reset(self) -> None:
        """Start new identities after a camera cut without recycling player IDs."""
        self._tracks.clear()

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


@dataclass(frozen=True)
class HandlerDecision:
    track_id: int | None
    source: Literal["observed", "held", "none"]


class BallHandlerTracker:
    """Smooth visual ball-handler labels without changing shot-scoring evidence."""

    def __init__(self, aspect_ratio: float, occlusion_grace_s: float = .8,
                 loose_ball_grace_s: float = .35):
        self.aspect_ratio = aspect_ratio
        self.occlusion_grace_s = occlusion_grace_s
        self.loose_ball_grace_s = loose_ball_grace_s
        self.reset()

    def reset(self) -> None:
        self.track_id: int | None = None
        self.last_confirmed_s: float | None = None
        self.pending_id: int | None = None
        self.pending_count = 0
        self.pending_time_s: float | None = None

    def _direct_contact(self, players: list[PoseFrame], ball: Detection | None) -> tuple[float, int] | None:
        if ball is None or ball.confidence < .45:
            return None
        candidates = []
        for player in players:
            if player.track_id is None:
                continue
            landmarks = player.landmarks
            body = [landmarks.get(name) for name in
                    ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
            if any(point is None or point[2] < .65 for point in body):
                continue
            shoulder = ((body[0][0] + body[1][0]) * self.aspect_ratio / 2,
                        (body[0][1] + body[1][1]) / 2)
            hip = ((body[2][0] + body[3][0]) * self.aspect_ratio / 2,
                   (body[2][1] + body[3][1]) / 2)
            torso = math.dist(shoulder, hip)
            if torso < .008:
                continue
            distances = [math.dist((point[0] * self.aspect_ratio, point[1]),
                                   (ball.x * self.aspect_ratio, ball.y)) / torso
                         for name in ("left_wrist", "right_wrist")
                         if (point := landmarks.get(name)) is not None and point[2] >= .65]
            if distances:
                candidates.append((min(distances), player.track_id))
        candidates.sort()
        if (candidates and candidates[0][0] <= .9
                and (len(candidates) == 1 or candidates[1][0] - candidates[0][0] >= .3)):
            return candidates[0]
        return None

    def _confirm(self, track_id: int, time_s: float) -> HandlerDecision:
        self.track_id = track_id
        self.last_confirmed_s = time_s
        self.pending_id = None
        self.pending_count = 0
        self.pending_time_s = None
        return HandlerDecision(track_id, "observed")

    def update(self, players: list[PoseFrame], ball: Detection | None, time_s: float) -> HandlerDecision:
        visible_ids = {player.track_id for player in players}
        direct = self._direct_contact(players, ball)
        if direct is not None:
            distance, candidate_id = direct
            if candidate_id == self.track_id or self.track_id is None:
                return self._confirm(candidate_id, time_s)
            grace = self.occlusion_grace_s if ball is None else self.loose_ball_grace_s
            if self.last_confirmed_s is None or time_s - self.last_confirmed_s > grace:
                return self._confirm(candidate_id, time_s)
            if self.pending_id == candidate_id and self.pending_time_s is not None and time_s - self.pending_time_s <= .12:
                self.pending_count += 1
            else:
                self.pending_id = candidate_id
                self.pending_count = 1
            self.pending_time_s = time_s
            # A near hand on one frame may be a defender contest or occlusion,
            # not a possession change. Require two close, consecutive sightings.
            if self.pending_count >= 2 and distance <= .65:
                return self._confirm(candidate_id, time_s)
        else:
            self.pending_id = None
            self.pending_count = 0
            self.pending_time_s = None
        grace = self.occlusion_grace_s if ball is None else self.loose_ball_grace_s
        if self.track_id is not None and self.last_confirmed_s is not None and time_s - self.last_confirmed_s <= grace:
            return HandlerDecision(self.track_id, "held") if self.track_id in visible_ids else HandlerDecision(None, "none")
        self.reset()
        return HandlerDecision(None, "none")


def _median_appearance(observations) -> np.ndarray | None:
    values = [o[3] for o in observations if o[3] is not None]
    return np.median(np.asarray(values, dtype=float), axis=0) if values else None


def stitch_tracks(frames: list[dict], aspect_ratio: float, fps: float, max_gap_s: float = .5) -> dict[int, int]:
    """Merge fragments of one camera segment that are the same player; returns {old: kept id}.

    The online tracker splits a player when jersey colour (hidden by the ball or
    arms) or a crouch pushes the association cost over its limit, even though
    the player barely moved. Offline, a later fragment joins an earlier track
    when the two never share a frame, it starts where the earlier track left
    off and, if that track resumes, ends where it resumes. Jersey colour only
    vetoes longer gaps, and a fragment with two similar candidates is left alone.
    Limits are conservative: a missed merge leaves a duplicate label, while a
    wrong one swaps identities (an occluder's jersey can mask the occluded
    player's, so appearance alone cannot justify a long gap).
    """
    tracks: dict[int, list] = {}
    for index, frame in enumerate(frames):
        for pose in frame["players"]:
            geometry = body_geometry(pose, aspect_ratio) if pose.track_id is not None else None
            if geometry is not None:
                tracks.setdefault(pose.track_id, []).append((index, geometry[0], geometry[1], pose.appearance))
    max_gap = max(1, round(max_gap_s * fps))

    def transition(src, dst) -> float | None:
        """Normalized cost of one track ending at src and continuing at dst; None if implausible."""
        gap = dst[0] - src[0]
        scale = dst[2] / src[2]
        if not 0 < gap <= max_gap or not .75 <= scale <= 1 / .75:
            return None
        jump = math.dist(src[1], dst[1]) / ((src[2] + dst[2]) / 2)
        cost = jump / (.6 + .03 * gap)
        return cost if cost <= 1 else None

    def appearance_differs(before: list, after: list) -> bool:
        a_look, b_look = _median_appearance(before[-5:]), _median_appearance(after[:5])
        return a_look is not None and b_look is not None and float(np.linalg.norm(a_look - b_look)) > .3

    mapping: dict[int, int] = {}
    while True:
        options: dict[int, list[tuple[float, int]]] = {}
        for a, A in tracks.items():
            a_frames = {o[0] for o in A}
            for b, B in tracks.items():
                if a == b or B[0][0] <= A[0][0] or a_frames & {o[0] for o in B}:
                    continue
                # Fragments may interleave while both online tracks stay alive;
                # every point where the identity switches must be plausible.
                merged = sorted([(o, a) for o in A] + [(o, b) for o in B], key=lambda item: item[0][0])
                costs = []
                for k, ((src, src_id), (dst, dst_id)) in enumerate(zip(merged, merged[1:])):
                    if src_id == dst_id:
                        continue
                    cost = transition(src, dst)
                    if cost is None or (dst[0] - src[0] > 3 and appearance_differs(
                            [o for o, t in merged[:k + 1] if t == src_id],
                            [o for o, t in merged[k + 1:] if t == dst_id])):
                        costs = None
                        break
                    costs.append(cost)
                if costs:
                    options.setdefault(b, []).append((max(costs), a))
        best = None
        for b, candidates in options.items():
            candidates.sort()
            if len(candidates) > 1 and candidates[1][0] < 1.5 * candidates[0][0] + .1:
                continue  # two earlier tracks fit: do not guess
            if best is None or candidates[0][0] < best[0]:
                best = (candidates[0][0], candidates[0][1], b)
        if best is None:
            break
        _, a, b = best
        tracks[a] = sorted(tracks[a] + tracks.pop(b), key=lambda o: o[0])
        mapping = {old: (a if kept == b else kept) for old, kept in mapping.items()}
        mapping[b] = a
    for frame in frames:
        for pose in frame["players"]:
            if pose.track_id in mapping:
                pose.track_id = mapping[pose.track_id]
    return mapping
