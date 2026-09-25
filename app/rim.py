"""Rim boxes for make/miss: detected automatically, or followed from a user mark.

The basketball detector's rim class finds a snug box around the hoop ring on
almost every game frame (median confidence ~0.85, sub-pixel jitter on test
clips), so no marking is needed. A user-marked rim still overrides; on a moving
camera it is followed with a CSRT tracker from the marked frame.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from app.basketball import RIM_CLASS

RimBox = tuple[float, float, float, float]  # normalized x, y, w, h
MIN_CONFIDENCE = .5


class RimTracker:
    """Follow a user-marked hoop through a continuous pan/zoom shot."""

    def __init__(self, initial: tuple[float, float, float, float], width: int, height: int):
        self.initial, self.width, self.height = initial, width, height
        self.tracker = None
        self.started = False
        self.lost = False
        self.last = initial
        self.failures = 0

    def _soft_failure(self):
        self.failures += 1
        if self.failures <= 5 and self.last:
            # Short ball/net occlusions should not erase the hoop annotation.
            self.started = False
            self.initial = self.last
            return self.last
        self.lost = True
        return None

    def update(self, frame: np.ndarray) -> tuple[float, float, float, float] | None:
        if self.lost:
            return None
        if not self.started:
            create = getattr(cv2, "TrackerCSRT_create", None)
            if create is None and hasattr(cv2, "legacy"):
                create = getattr(cv2.legacy, "TrackerCSRT_create", None)
            if create is None:
                raise RuntimeError("This OpenCV build does not include the CSRT tracker.")
            self.tracker = create()
            x, y, w, h = self.initial
            self.tracker.init(frame, (round(x * self.width), round(y * self.height),
                                      round(w * self.width), round(h * self.height)))
            self.started = True
            return self.initial
        ok, box = self.tracker.update(frame)
        if not ok:
            return self._soft_failure()
        x, y, w, h = (float(value) for value in box)
        if w < 5 or h < 3 or x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            return self._soft_failure()
        current = x / self.width, y / self.height, w / self.width, h / self.height
        old_x, old_y, old_w, old_h = self.last
        center_jump = np.hypot((current[0] + current[2] / 2) - (old_x + old_w / 2),
                               (current[1] + current[3] / 2) - (old_y + old_h / 2))
        scale = current[2] / max(old_w, 1e-6)
        if center_jump > .12 or not .62 <= scale <= 1.6:
            return self._soft_failure()
        self.failures = 0
        self.last = current
        return current

    def stop_at_cut(self) -> None:
        # A box from the old camera angle is not a valid initialization in the new shot.
        self.lost = True


def track_marked_rim(input_path: Path, initial: tuple[float, float, float, float],
                     anchor_frame: int, width: int, height: int, total: int
                     ) -> tuple[dict[int, tuple[float, float, float, float]], bool]:
    """Track the marked rim forward and backward from its annotation frame."""
    if anchor_frame < 0 or anchor_frame >= total:
        return {}, True
    forward, backward = {}, {}
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        return {}, True
    capture.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame)
    ok, frame = capture.read()
    if not ok:
        capture.release()
        return {}, True
    tracker = RimTracker(initial, width, height)
    forward[anchor_frame] = tracker.update(frame)  # type: ignore[assignment]
    for frame_no in range(anchor_frame + 1, total):
        ok, frame = capture.read()
        if not ok:
            break
        box = tracker.update(frame)
        if box is None:
            break
        forward[frame_no] = box
    lost = tracker.lost
    capture.release()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        return forward, True
    tracker = RimTracker(initial, width, height)
    capture.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame)
    ok, anchor = capture.read()
    if not ok:
        capture.release()
        return forward, True
    tracker.update(anchor)
    for frame_no in range(anchor_frame - 1, -1, -1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = capture.read()
        if not ok:
            lost = True
            break
        box = tracker.update(frame)
        if box is None:
            lost = True
            break
        backward[frame_no] = box
    capture.release()
    return {**backward, **forward}, lost


def rim_candidates(objects) -> list[tuple[float, tuple[float, float, float, float]]]:
    """Confident detector rim boxes (confidence, normalized xyxy) from one frame's objects."""
    return [(conf, box) for cls, conf, box in objects or [] if cls == RIM_CLASS and conf >= MIN_CONFIDENCE]


def xywh(box) -> RimBox:
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1


def box_iou(a: RimBox, b: RimBox) -> float:
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - ix * iy
    return ix * iy / union if union > 0 else 0.


def track_detected_rims(detections: dict[int, list], cuts: list[int], step: int = 1,
                        max_gap_frames: int = 15) -> tuple[dict[int, RimBox], dict]:
    """Link per-frame rim detections into hoops and keep the dominant hoop per camera segment.

    detections: {source frame: [(confidence, normalized xyxy), ...]}. Within a
    segment, a detection joins the hoop whose last box it is near (the hoop may
    pan by a fraction of its width per frame) and similar in size. The hoop
    with the most total confidence wins; one-off false positives (a red sleeve
    near the baseline) form short, weak tracks and are dropped. Box jitter is
    reduced with a 5-sample local linear fit. Gaps are left to the scorer's rim lookup,
    which interpolates up to 12 frames and never bridges a cut.
    """
    frames = sorted(detections)
    edges = [-math.inf, *sorted(cuts), math.inf]
    rims: dict[int, RimBox] = {}
    stats = {"segments": 0, "segments_with_rim": 0, "other_hoop_tracks": 0}
    for start, end in zip(edges, edges[1:]):
        segment = [f for f in frames if start <= f < end]
        if not segment:
            continue
        stats["segments"] += 1
        tracks: list[list[tuple[int, float, RimBox]]] = []
        for frame in segment:
            used = set()
            for conf, box in sorted(detections[frame], reverse=True):
                box = xywh(box)
                best, best_cost = None, math.inf
                for index, track in enumerate(tracks):
                    last_frame, _, last = track[-1]
                    gap = frame - last_frame
                    if index in used or not 0 < gap <= max_gap_frames:
                        continue
                    scale = box[2] / max(last[2], 1e-9)
                    if not .6 <= scale <= 1 / .6:
                        continue
                    jump = math.dist((box[0] + box[2] / 2, box[1] + box[3] / 2),
                                     (last[0] + last[2] / 2, last[1] + last[3] / 2)) / max(box[2], last[2])
                    cost = jump / (1. + .25 * gap / step)
                    if cost <= 1 and cost < best_cost:
                        best, best_cost = index, cost
                if best is None:
                    tracks.append([(frame, conf, box)])
                    used.add(len(tracks) - 1)
                else:
                    tracks[best].append((frame, conf, box))
                    used.add(best)
        if not tracks:
            continue
        dominant = max(tracks, key=lambda track: sum(conf for _, conf, _ in track))
        if len(dominant) < 3:
            continue
        stats["segments_with_rim"] += 1
        stats["other_hoop_tracks"] += len(tracks) - 1
        boxes = np.asarray([box for _, _, box in dominant])
        times = np.asarray([frame for frame, _, _ in dominant], dtype=float)
        for index, (frame, _, _) in enumerate(dominant):
            # A local straight-line fit follows a steady pan exactly (a median of
            # a trending window would just return the raw box) and averages jitter.
            lo, hi = max(0, index - 2), index + 3
            if len(times[lo:hi]) < 3:
                rims[frame] = tuple(float(v) for v in boxes[index])
                continue
            t = times[lo:hi] - frame
            coefficients = np.polyfit(t, boxes[lo:hi], 1)
            rims[frame] = tuple(float(v) for v in coefficients[1])
    return rims, stats


def detect_fixed_rim(input_path: Path, detector, samples: int = 15) -> RimBox | None:
    """Median rim box over frames sampled across a stationary-camera clip, or None.

    Requires the same hoop on at least 40% of the samples, so a clip whose rim is
    mostly out of view does not get a guessed box.
    """
    capture = cv2.VideoCapture(str(input_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not capture.isOpened() or total < 1:
        capture.release()
        return None
    wanted = sorted({min(total - 1, round((i + .5) * total / samples)) for i in range(samples)})
    detections: dict[int, list] = {}
    for frame_no in wanted:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = capture.read()
        if ok:
            detections[frame_no] = rim_candidates(detector.detect(frame))
    capture.release()
    rims, _ = track_detected_rims(detections, [], step=max(1, total // samples),
                                  max_gap_frames=total)
    if len(rims) < .4 * max(1, len(detections)):
        return None
    return tuple(float(v) for v in np.median(np.asarray(list(rims.values())), axis=0))
