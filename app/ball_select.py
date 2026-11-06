"""Ball choice that will not hold a still false 'ball' (opt-in: BALLFORM_STATIC_BALL_FILTER=1).

The frame loop's tracker (vision.CourtVision.detect) keeps one ball and gates new candidates
around it, so an object the detector calls a ball that never moves (crowd at the top edge, a
referee, a stanchion) can hold the track through a real release while the real ball is scored
elsewhere, often only in the side crops. With the filter on, a candidate is static when
detections stayed within STATIC_RADII ball radii of it for STATIC_SECONDS, no wrist was within
WRIST_TORSOS torso lengths and it is not at a rim (a ball resting on the rim is real). A static
candidate is never chosen, a static previous ball does not gate the next choice, and when every
confident full-frame ball is static the side crops run anyway.

With the filter on or off, every ball candidate of every analyzed frame (full frame and side
crops) is kept so the saved measurements can be chosen from again offline.
"""
from __future__ import annotations

import math
import os
import time
from statistics import median

from app import vision
from app.basketball import BALL_CLASSES, PLAYER_CLASSES
from app.models import Detection
from app.rim import rim_candidates

ENV = "BALLFORM_STATIC_BALL_FILTER"
STATIC_SECONDS = .3
STATIC_RADII = .75
# Of the frame's long side: a big ball box gets no looser test than this.
MAX_STATIC_TOLERANCE = .012
# Analyzed frames a still object may go undetected inside one static run.
MAX_GAP_FRAMES = 2
WRIST_TORSOS = 1.5
# vision.CourtVision.detect runs the side crops only without a full-frame ball this confident.
SIDE_CROP_CONF = .65
HISTORY_SECONDS = 1.5
BALL_IN_BASKET = 2


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def candidates_from(region_objects, width: int, height: int, frame_no: int, time_s: float):
    """(Detection, region index, category) per ball object; region 0 is the full frame.

    Same mapping as vision.CourtVision.detect, so the Detections match the ones it chooses from.
    """
    found = []
    for index, (x1, y1, x2, y2, objects) in enumerate(region_objects):
        for category, confidence, (bx1, by1, bx2, by2) in objects:
            if category not in BALL_CLASSES:
                continue
            found.append((Detection(frame_no, time_s, ((bx1+bx2)*(x2-x1)/2+x1)/width,
                                    ((by1+by2)*(y2-y1)/2+y1)/height, confidence,
                                    max((bx2-bx1)*(x2-x1), (by2-by1)*(y2-y1))/(2*max(width, height))),
                          index, category))
    return found


def select(balls: list[Detection], previous: Detection | None, prior: Detection | None,
           frame_no: int, time_s: float) -> Detection | None:
    """vision.CourtVision.detect's choice: a constant-velocity gate around the previous ball."""
    if previous and time_s - previous.time_s < .3:
        predicted = (previous.x, previous.y)
        if prior and previous.frame - prior.frame == frame_no - previous.frame:
            predicted = (2 * previous.x - prior.x, 2 * previous.y - prior.y)
        velocity = math.hypot(previous.x - (prior.x if prior else previous.x),
                              previous.y - (prior.y if prior else previous.y))
        allowed = max(.10, 2.5 * velocity + .06, 7.0 * previous.radius)

        def cost(candidate):
            jump = math.hypot(candidate.x - predicted[0], candidate.y - predicted[1])
            if jump > allowed and candidate.confidence < max(.82, previous.confidence + .08):
                return -100.0
            return candidate.confidence - 3.0 * jump
        best = max(balls, key=cost, default=None)
        return best if best is not None and cost(best) > -50 else None
    return max(balls, key=lambda b: b.confidence, default=None)


def wrists_of(poses, width: int, height: int) -> list[tuple[float, float, float]]:
    """(x, y, torso length) in pixels for every wrist of every pose."""
    wrists = []
    for pose in poses:
        marks = pose.landmarks
        try:
            ls, rs, lh, rh = [marks[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
        except KeyError:
            continue
        torso = math.hypot((ls[0] + rs[0] - lh[0] - rh[0]) * width / 2, (ls[1] + rs[1] - lh[1] - rh[1]) * height / 2)
        for name in ("left_wrist", "right_wrist"):
            if name in marks:
                wrists.append((marks[name][0] * width, marks[name][1] * height, torso))
    return wrists


class StaticBallFilter:
    """Recent candidate positions and wrists, to tell a still object from the ball."""

    def __init__(self):
        # (time_s, [(x, y)], wrists, rims) per analyzed frame, oldest first.
        self.history: list[tuple[float, list[tuple[float, float]], list, list]] = []
        self.size = (1, 1)

    def reset(self) -> None:
        """Positions before a camera cut say nothing about the frames after it."""
        self.history = []

    def observe(self, time_s: float, balls: list[Detection], wrists, rims=()) -> None:
        self.history.append((time_s, [(b.x, b.y) for b in balls], wrists, list(rims)))
        while self.history and time_s - self.history[0][0] > HISTORY_SECONDS:
            self.history.pop(0)

    def _distance(self, a, b) -> float:
        width, height = self.size
        return math.hypot((a[0] - b[0]) * width, (a[1] - b[1]) * height) / max(width, height)

    def _wrist_torsos(self, wrists, point) -> float:
        width, height = self.size
        x, y = point[0] * width, point[1] * height
        return min((math.hypot(wx - x, wy - y) / max(torso, 1.) for wx, wy, torso in wrists), default=math.inf)

    def _at_rim(self, ball: Detection, rims) -> bool:
        width, height = self.size
        long_side = max(width, height)
        for x1, y1, x2, y2 in rims:
            margin_x = max(.5 * (x2 - x1), 2 * ball.radius * long_side / width)
            margin_y = max(.5 * (x2 - x1) * width / height, 2 * ball.radius * long_side / height)
            if x1 - margin_x <= ball.x <= x2 + margin_x and y1 - margin_y <= ball.y <= y2 + margin_y:
                return True
        return False

    def is_static(self, ball: Detection, time_s: float, wrists=None, rims=(), category: int = 1) -> bool:
        """Still for STATIC_SECONDS (up to this frame), away from every wrist and rim.

        ``wrists`` are this frame's, when its poses are known.
        """
        # A hand or the ball itself often hides the rim on the frames the ball sits on it,
        # so any hoop seen in the last HISTORY_SECONDS counts.
        if category == BALL_IN_BASKET or self._at_rim(ball, [*rims, *(r for *_, seen in self.history for r in seen)]):
            return False
        tolerance = min(STATIC_RADII * ball.radius, MAX_STATIC_TOLERANCE)
        point = (ball.x, ball.y)
        start, missed = time_s, 0
        ratios = [self._wrist_torsos(wrists, point)] if wrists is not None else []
        for frame_time, positions, frame_wrists, _ in reversed(self.history):
            if frame_time >= time_s:
                continue
            if any(self._distance(point, p) <= tolerance for p in positions):
                start, missed = frame_time, 0
                ratios.append(self._wrist_torsos(frame_wrists, point))
            else:
                missed += 1
                if missed > MAX_GAP_FRAMES:
                    break
        if time_s - start < STATIC_SECONDS - 1e-6 or not ratios:
            return False
        return median(ratios) > WRIST_TORSOS

    def choose(self, found, previous: Detection | None, prior: Detection | None, frame_no: int,
               time_s: float, wrists, rims) -> Detection | None:
        """The frame loop's choice among the candidates that are not static."""
        moving = [ball for ball, _, category in found if not self.is_static(ball, time_s, wrists, rims, category)]
        if previous is not None and self.is_static(previous, time_s, wrists, rims):
            previous = prior = None
        elif prior is not None and self.is_static(prior, time_s, wrists, rims):
            prior = None
        return select(moving, previous, prior, frame_no, time_s)


class CandidateCourtVision(vision.CourtVision):
    """CourtVision that also reports every ball candidate and, given a filter, skips static ones."""

    def __init__(self, *args, static_filter: StaticBallFilter | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_filter = static_filter
        # Last detect(): (x, y, confidence, radius, region) per candidate; region 0 is the full frame.
        self.ball_candidates: list[tuple[float, float, float, float, int]] = []
        self.forced_side_crop_frames = 0
        self.static_choice_frames = 0
        self._side_objects = None

    def _detect_basketball_batch(self, frames):
        objects = super()._detect_basketball_batch(frames)
        self._side_objects = objects
        return objects

    def detect(self, frame, frame_no, time_s, previous_ball=None, prior_ball=None,
               prefetched_objects=None, rims=()):
        """vision.CourtVision.detect; ``rims`` are extra hoop boxes (normalized xyxy) for the filter."""
        if not isinstance(self.ball_model, vision.BasketballDetector):
            self.ball_candidates = []
            return super().detect(frame, frame_no, time_s, previous_ball, prior_ball, prefetched_objects)
        height, width = frame.shape[:2]
        if prefetched_objects is not None:
            objects = prefetched_objects
        else:
            started = time.perf_counter()
            objects = self._detect_basketball_objects(frame)
            # vision.CourtVision.detect counts this detection in its wall time when it runs it itself.
            self.timing_seconds["total"] += time.perf_counter() - started
        hoops = [box for _, box in rim_candidates(objects)] + list(rims)
        passed = objects
        # vision.CourtVision.detect returns before any ball choice on these frames.
        skipped = (self.profile == "moving"
                   and not any(cls in PLAYER_CLASSES and conf >= .4 for cls, conf, _ in objects))
        if self.static_filter is not None:
            self.static_filter.size = (width, height)
            full = candidates_from([(0, 0, width, height, objects)], width, height, frame_no, time_s)
            ball_objects = [k for k, o in enumerate(objects) if o[0] in BALL_CLASSES]
            confident = [i for i, (ball, _, _) in enumerate(full) if ball.confidence >= SIDE_CROP_CONF]
            if not skipped and self.tiled and confident and all(
                    self.static_filter.is_static(full[i][0], time_s, None, hoops, full[i][2]) for i in confident):
                # Hidden from the side-crop trigger only; they are still candidates below.
                hidden = {ball_objects[i] for i in confident}
                passed = [o for k, o in enumerate(objects) if k not in hidden]
                self.forced_side_crop_frames += 1
        self._side_objects = None
        poses, maps, ball = super().detect(frame, frame_no, time_s, previous_ball, prior_ball,
                                           prefetched_objects=passed)
        regions_objects = [(0, 0, width, height, objects)]
        if self._side_objects is not None:
            side_regions = list(vision.regions(width, height, True))[1:]
            regions_objects += [(*region, side) for region, side in zip(side_regions, self._side_objects)]
        found = candidates_from(regions_objects, width, height, frame_no, time_s)
        self.ball_candidates = [(b.x, b.y, b.confidence, b.radius, index) for b, index, _ in found]
        if self.static_filter is None:
            return poses, maps, ball
        wrists = wrists_of(poses, width, height)
        if not skipped:
            chosen = self.static_filter.choose(found, previous_ball, prior_ball, frame_no, time_s, wrists, hoops)
            if chosen != ball:
                self.static_choice_frames += 1
            ball = chosen
            if frame_no in self.side_crop_ball_selected:
                self.side_crop_ball_selected[frame_no] = ball is not None and any(
                    b is ball and index > 0 for b, index, _ in found)
        self.static_filter.observe(time_s, [b for b, _, _ in found], wrists, hoops)
        return poses, maps, ball
