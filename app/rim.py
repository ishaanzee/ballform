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

from app.basketball import PLAYER_CLASSES, RIM_CLASS

RimBox = tuple[float, float, float, float]  # normalized x, y, w, h
MIN_CONFIDENCE = .5
# Hoops in view were boxed at 0.8-0.9 on the broadcast clips; a second hoop kept by time
# on a camera that follows play needs that median confidence (track_detected_rims).
HOOP_IN_VIEW_CONFIDENCE = .7
# Two rim detections in one frame covering this much of the smaller box are
# one hoop: the ball at the rim gave 0.43 and 0.75 (f6f0, 3e78).
SAME_HOOP_OVERLAP = .3
# Ball frames near a hoop needed before ball activity, not detector confidence,
# picks between several steady hoops.
BALL_VISITS_TO_PICK_HOOP = 3
BALL_VISIT_CONFIDENCE = .5


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


def _overlap(a: RimBox, b: RimBox) -> float:
    """Intersection over the smaller box's area."""
    ix = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    smaller = min(a[2] * a[3], b[2] * b[3])
    return ix * iy / smaller if smaller > 0 else 0.


def _ball_visits(track: list[tuple[int, float, RimBox]], balls: dict[int, tuple[float, float, float]],
                 frames: list[int], aspect: float) -> int:
    """Frames whose confident ball is coming down onto the hoop (within 2 rim widths
    sideways, up to 4 above the ring, and lower than a frame or two before) or being
    lifted to it (within 1.5 sideways, rising, up to 1 above the ring).

    A jump shot drops onto its own hoop and a layup is carried up to it. On a
    head-height camera in a gym (bKA6) the neighbouring courts' hoops also collect
    ball frames, but those balls are a still, faint false positive, a ball rising
    away, or a nearer ball passing in front of the far hoop at or below its ring.
    """
    entries = {frame: box for frame, _, box in track}
    known = sorted(entries)
    visits = 0
    for frame in frames:
        ball = balls.get(frame)
        previous = balls.get(frame - 1) or balls.get(frame - 2)
        if ball is None or previous is None or ball[2] < BALL_VISIT_CONFIDENCE:
            continue
        nearest = min(known, key=lambda f: abs(f - frame))
        x, y, w, h = entries[nearest]
        rim_width_y = max(w * aspect, 1e-9)  # one rim width in normalized-height units
        dx = (ball[0] - (x + w / 2)) / max(w, 1e-9)
        dy = (ball[1] - (y + h / 2)) / rim_width_y
        falling = (ball[1] - previous[1]) / rim_width_y
        dropping = abs(dx) <= 2 and -4 <= dy <= -.25 and falling >= .05
        lifted = abs(dx) <= 1.5 and -1 <= dy <= -.1 and falling <= -.05
        visits += dropping or lifted
    return visits


def _confidence(track: list[tuple[int, float, RimBox]]) -> float:
    return sum(conf for _, conf, _ in track)


def _apart(track, others) -> bool:
    """True when the track's frames lie wholly before or after each of the other tracks'."""
    return all(track[-1][0] < other[0][0] or track[0][0] > other[-1][0] for other in others)


def track_detected_rims(detections: dict[int, list], cuts: list[int], step: int = 1,
                        max_gap_frames: int = 15, balls: dict[int, tuple[float, float, float]] | None = None,
                        aspect: float = 16 / 9, follows_play: bool = False) -> tuple[dict[int, RimBox], dict]:
    """Link per-frame rim detections into hoops and keep the dominant hoop per camera segment.

    detections: {source frame: [(confidence, normalized xyxy), ...]}. Within a
    segment, a detection joins the hoop whose last box it is near (the hoop may
    pan by a fraction of its width per frame) and similar in size. The hoop
    with the most total confidence wins; one-off false positives (a red sleeve
    near the baseline) form short, weak tracks and are dropped. When ``balls``
    ({frame: normalized ball x, y, confidence}) is given and several steady
    hoops are in view, the one the ball drops onto or is lifted to wins instead:
    a fixed camera in a gym can see the neighbouring courts' hoops, sharper and
    more confident than the one in play (bKA6 picked a side court's hoop on all
    nine clips). With ``follows_play`` (a camera that pans with play), a hoop
    tracked only while the dominant one is out of view is kept for its own time
    too: in one continuous pan from end to end the camera shows one hoop and then
    the other (20a1, 6c6a, d731, 1e79 lost the far hoop at 0.8-0.9 confidence).
    Box jitter is reduced with a 5-sample local linear fit. Gaps are left to the
    scorer's rim lookup, which interpolates and never bridges a cut.
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
            kept: list[RimBox] = []
            for conf, box in sorted(detections[frame], reverse=True):
                box = xywh(box)
                # The ball at the rim can add a second, overlapping hoop box (ball
                # and rim together). Starting a track from it split the hoop just
                # as the ball arrived (3e78, f6f0), losing the rim for the outcome.
                if any(_overlap(box, other) >= SAME_HOOP_OVERLAP for other in kept):
                    continue
                kept.append(box)
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
        dominant = max(tracks, key=_confidence)
        steady = [track for track in tracks if len(track) >= 3]
        if balls and len(steady) > 1:
            span = [f for f in balls if start <= f < end]
            visits = {id(track): _ball_visits(track, balls, span, aspect) for track in steady}
            if max(visits.values()) >= BALL_VISITS_TO_PICK_HOOP:
                dominant = max(steady, key=lambda track: (visits[id(track)], _confidence(track)))
        if len(dominant) < 3:
            continue
        chosen = [dominant]
        if follows_play:
            for track in sorted(tracks, key=_confidence, reverse=True):
                if track is not dominant and _steady_hoop(track, max_gap_frames) and _apart(track, chosen):
                    chosen.append(track)
        stats["segments_with_rim"] += 1
        stats["other_hoop_tracks"] += len(tracks) - len(chosen)
        stats["hoops_by_time"] = stats.get("hoops_by_time", 0) + len(chosen) - 1
        # Frames where one hoop takes over from another.
        firsts = sorted(hoop[0][0] for hoop in chosen)
        stats["hoop_switches"] = [*stats.get("hoop_switches", []), *firsts[1:]]
        for hoop in chosen:
            boxes = np.asarray([box for _, _, box in hoop])
            times = np.asarray([frame for frame, _, _ in hoop], dtype=float)
            for index, (frame, _, _) in enumerate(hoop):
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


def _steady_hoop(track, max_gap_frames: int) -> bool:
    """A hoop seen for at least as long as the longest gap a track bridges, at the
    detector's usual confidence for a hoop in view: not a one-off false box."""
    return (track[-1][0] - track[0][0] >= max_gap_frames
            and float(np.median([conf for _, conf, _ in track])) >= HOOP_IN_VIEW_CONFIDENCE)


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


# Zoomed hoop search for fixed cameras (see fixed_hoop_candidates).
TILED_MIN_CONFIDENCE = .25
TILED_MIN_SAMPLES = 3
# In pixels a hoop box is 0.30-0.56 as tall as wide on the fixed-camera clips; a light
# pole on the outdoor clip was boxed at 0.68-0.73.
TILED_MAX_HEIGHT_TO_WIDTH = .63
# An 18 in rim beside 6 ft players: rim width over the median player box height was
# 0.23-0.52 on the fixed-camera clips, the light pole 0.11.
TILED_MIN_WIDTH_TO_PLAYER = .15


def _tiles(width: int, height: int):
    """Six overlapping half-width, two-thirds-height crops: a 2x zoom over the frame."""
    crop_width, crop_height = width // 2, round(height * 2 / 3)
    for column in range(3):
        for row in range(2):
            x1, y1 = round(column * width / 4), round(row * height / 3)
            yield x1, y1, min(width, x1 + crop_width), min(height, y1 + crop_height)


def fixed_hoop_candidates(input_path: Path, detector, samples: int = 6
                          ) -> tuple[list[tuple[float, tuple[float, float, float, float]]], dict]:
    """Hoops for a fixed camera whose full frames show no steady rim, from zoomed tiles.

    On a 720p wall camera with hoops at the frame edges (jzIB) or an outdoor
    gooseneck hoop (Rghu), the full-frame detector boxes the rim below
    MIN_CONFIDENCE or not at all; 2x tiles find it at 0.25-0.6 (13 of 17 such
    clips, no false hoops). A hoop must appear in the same place on
    TILED_MIN_SAMPLES of the sampled frames, be hoop-shaped and be plausibly
    sized next to the players. Returns ([(confidence, normalized xyxy), ...], stats), empty when
    the full frames already show a steady rim (the per-frame detection then
    suffices) or no hoop is found.
    """
    stats = {"samples": 0, "full_frame_rim_samples": 0, "searched_tiles": False, "hoops": 0}
    capture = cv2.VideoCapture(str(input_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if capture.isOpened() and total > 0:
        for index in range(samples):
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, round((index + .5) * total / samples)))
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
    capture.release()
    stats["samples"] = len(frames)
    if not frames:
        return [], stats
    player_heights = []
    for frame in frames:
        objects = detector.detect(frame) or []
        stats["full_frame_rim_samples"] += bool(rim_candidates(objects))
        player_heights += [box[3] - box[1] for cls, conf, box in objects if cls in PLAYER_CLASSES and conf >= .4]
    if stats["full_frame_rim_samples"] >= .4 * len(frames):
        return [], stats
    stats["searched_tiles"] = True
    clusters: list[dict] = []
    for index, frame in enumerate(frames):
        height, width = frame.shape[:2]
        found = []
        for x1, y1, x2, y2 in _tiles(width, height):
            for cls, conf, box in detector.detect(frame[y1:y2, x1:x2]) or []:
                if cls != RIM_CLASS or conf < TILED_MIN_CONFIDENCE:
                    continue
                found.append((conf, ((x1 + box[0] * (x2 - x1)) / width, (y1 + box[1] * (y2 - y1)) / height,
                                     (x1 + box[2] * (x2 - x1)) / width, (y1 + box[3] * (y2 - y1)) / height)))
        for conf, box in sorted(found, reverse=True):
            x, y, w, h = xywh(box)
            centre = (x + w / 2, y + h / 2)
            for cluster in clusters:
                cx, cy, cw, _ = cluster["box"]
                if (math.dist(centre, (cx + cw / 2, cy + cluster["box"][3] / 2)) <= max(w, cw)
                        and .6 <= w / max(cw, 1e-9) <= 1 / .6):
                    if index not in cluster["samples"]:
                        cluster["samples"].add(index)
                        cluster["boxes"].append((x, y, w, h))
                        cluster["confidences"].append(conf)
                    break
            else:
                clusters.append({"box": (x, y, w, h), "samples": {index}, "boxes": [(x, y, w, h)],
                                 "confidences": [conf]})
    aspect = frames[0].shape[1] / frames[0].shape[0]
    player_height = float(np.median(player_heights)) if player_heights else None
    hoops = []
    for cluster in clusters:
        x, y, w, h = (float(v) for v in np.median(np.asarray(cluster["boxes"]), axis=0))
        if (len(cluster["samples"]) >= TILED_MIN_SAMPLES
                and h <= TILED_MAX_HEIGHT_TO_WIDTH * w * aspect
                and (player_height is None or w * aspect >= TILED_MIN_WIDTH_TO_PLAYER * player_height)):
            hoops.append((float(np.mean(cluster["confidences"])), (x, y, x + w, y + h)))
    stats["hoops"] = len(hoops)
    return sorted(hoops, reverse=True), stats
