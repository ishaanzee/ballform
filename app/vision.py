"""Multi-person detection for wide court views; all outputs use full-frame coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np

from app.models import Detection, PoseFrame
from app.basketball import BasketballDetector, BALL_CLASSES, PLAYER_CLASSES, REFEREE_CLASS

# COCO-17 -> the MediaPipe indices used by the existing drawing/scoring code.
COCO_TO_MP = {0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
              11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
NAMES = {0: "nose", 11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
         14: "right_elbow", 15: "left_wrist", 16: "right_wrist", 23: "left_hip",
         24: "right_hip", 25: "left_knee", 26: "right_knee", 27: "left_ankle", 28: "right_ankle"}
CAMERAS = {"auto", "broadcast", "elevated", "courtside", "moving"}


def camera_profile(camera: str, mode: str) -> str:
    if camera not in CAMERAS:
        raise ValueError("Unknown camera profile")
    return ("broadcast" if mode == "one_on_one" else "courtside") if camera == "auto" else camera


def validate_court(points):
    if points is None:
        return None
    polygon = np.asarray(points, dtype=np.float32)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= 8
            or not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any()
            or abs(cv2.contourArea(polygon)) < .01 or not cv2.isContourConvex(polygon)):
        raise ValueError("Court must be 3–8 normalized [x,y] points around a convex playing area.")
    return polygon.tolist()


def regions(width: int, height: int, tiled: bool):
    """Full view plus two overlapping crops."""
    yield (0, 0, width, height)
    if tiled and width >= 960:
        crop_width = round(width * .60)
        yield (0, 0, crop_width, height)
        yield (width - crop_width, 0, width, height)


def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(union, 1e-9)


@dataclass
class Person:
    box: tuple[float, float, float, float]
    confidence: float
    landmarks: dict


def merge_people(candidates: list[Person]) -> list[Person]:
    kept = []
    for candidate in sorted(candidates, key=lambda p: p.confidence, reverse=True):
        if not any(iou(candidate.box, old.box) > .55 for old in kept):
            kept.append(candidate)
    return kept


def map_keypoints(keypoints, region, width, height):
    x, y, _, _ = region
    return {mp_index: ((float(keypoints[index][0]) + x) / width,
                       (float(keypoints[index][1]) + y) / height,
                       float(keypoints[index][2]))
            for index, mp_index in COCO_TO_MP.items()}


def on_court(person: Person, court) -> bool:
    if court is None:
        return True
    ankles = [person.landmarks[i] for i in (27, 28) if person.landmarks[i][2] >= .4]
    # Either foot can be on court when the other is over the sideline or airborne.
    feet = [(p[0], p[1]) for p in ankles] or [((person.box[0] + person.box[2]) / 2, person.box[3])]
    polygon = np.asarray(court, dtype=np.float32)
    return any(cv2.pointPolygonTest(polygon, p, False) >= 0 for p in feet)


class CourtVision:
    def __init__(self, pose_model, ball_model, device: str, profile: str, court=None):
        self.pose_model, self.ball_model = pose_model, ball_model
        self.device, self.profile, self.court = device, profile, court
        self.tiled = profile in {"broadcast", "elevated", "moving"}
        self.imgsz = 1280 if self.tiled else 960
        self.raw_people = 0
        self.timing_seconds = {"total": 0.0, "pose": 0.0, "ball": 0.0}
        self.pose_predict_calls = 0
        self.pose_images = 0
        self.ball_crop_overlap_frames = 0
        self.side_crop_waits: dict[int, float] = {}
        self.side_crop_ball_candidates: dict[int, int] = {}
        self.side_crop_ball_selected: dict[int, bool] = {}
        self._ball_executor = None

    def close(self):
        if self._ball_executor is not None:
            self._ball_executor.shutdown(wait=True)
            self._ball_executor = None

    def _detect_basketball_objects(self, frame):
        started = time.perf_counter()
        try:
            return self.ball_model.detect(frame)
        finally:
            self.timing_seconds["ball"] += time.perf_counter() - started

    def _predict_pose(self, crop, size):
        started = time.perf_counter()
        prepared = self.pose_model.infer(crop, size)
        self.timing_seconds["pose"] += time.perf_counter() - started
        self.pose_predict_calls += 1
        self.pose_images += 1
        return prepared

    def detect(self, frame, frame_no, time_s, previous_ball=None, prior_ball=None,
               prefetched_objects=None):
        started = time.perf_counter()
        height, width = frame.shape[:2]
        candidates, balls = [], []
        basketball_objects = (prefetched_objects if prefetched_objects is not None
                              else self._detect_basketball_objects(frame)) if isinstance(self.ball_model, BasketballDetector) else None
        if (basketball_objects is not None and self.profile in {"broadcast", "moving"}
                and not any(cls in PLAYER_CLASSES and conf >= .4 for cls, conf, _ in basketball_objects)):
            self.raw_people = 0
            self.timing_seconds["total"] += time.perf_counter() - started
            return [], [], None
        ball_regions = [(0, 0, width, height, basketball_objects)] if basketball_objects is not None else []
        side_ball_future = None
        if basketball_objects is not None and self.tiled and not any(
                cls in BALL_CLASSES and conf >= .65 for cls, conf, _ in basketball_objects):
            if self._ball_executor is None:
                self._ball_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ballform-ball-crops")
            side_regions = list(regions(width, height, True))[1:]
            def detect_side_crops():
                return [(x1, y1, x2, y2, self._detect_basketball_objects(frame[y1:y2, x1:x2]))
                        for x1, y1, x2, y2 in side_regions]
            side_ball_future = self._ball_executor.submit(detect_side_crops)
            self.ball_crop_overlap_frames += 1
        pose_regions = list(regions(width, height, self.tiled))
        for index, region in enumerate(pose_regions):
            x1, y1, x2, y2 = region
            crop = frame[y1:y2, x1:x2]
            size = self.imgsz if index == 0 else 960
            pose_data = self._predict_pose(crop, size)
            if pose_data is not None:
                for box, confidence, keypoints in zip(*pose_data):
                    if keypoints.shape != (17, 3):
                        raise ValueError("Game pose model must use COCO's 17 keypoints with confidence.")
                    mapped = map_keypoints(keypoints, region, width, height)
                    normalized_box = ((box[0] + x1) / width, (box[1] + y1) / height,
                                      (box[2] + x1) / width, (box[3] + y1) / height)
                    # Reject crop-edge fragments; overlapping/full passes cover those people.
                    if index and ((x1 > 0 and box[0] < 4) or (x2 < width and box[2] > x2 - x1 - 4)):
                        continue
                    candidates.append(Person(normalized_box, float(confidence), mapped))
            if basketball_objects is not None:
                continue
            ball_started = time.perf_counter()
            result = self.ball_model.predict(crop, imgsz=size, conf=.25, classes=[32],
                                             device=self.device, verbose=False)[0]
            ball_data = None if result.boxes is None else (result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
            self.timing_seconds["ball"] += time.perf_counter() - ball_started
            if ball_data is not None:
                for box, confidence in zip(*ball_data):
                    bx1, by1, bx2, by2 = box
                    balls.append(Detection(frame_no, time_s, (bx1 + bx2 + 2*x1) / (2*width),
                                           (by1 + by2 + 2*y1) / (2*height), float(confidence),
                                           max(bx2-bx1, by2-by1) / (2*max(width, height))))
        if side_ball_future is not None:
            wait_started = time.perf_counter()
            side_ball_regions = side_ball_future.result()
            self.side_crop_waits[frame_no] = time.perf_counter() - wait_started
            ball_regions.extend(side_ball_regions)
        side_ball_ids = set()
        for region_index, (x1, y1, x2, y2, objects) in enumerate(ball_regions):
            for category, confidence, box in objects:
                if category not in BALL_CLASSES:
                    continue
                bx1, by1, bx2, by2 = box
                candidate = Detection(frame_no, time_s, ((bx1+bx2)*(x2-x1)/2+x1)/width,
                                      ((by1+by2)*(y2-y1)/2+y1)/height, confidence,
                                      max((bx2-bx1)*(x2-x1), (by2-by1)*(y2-y1))/(2*max(width,height)))
                balls.append(candidate)
                if region_index > 0:
                    side_ball_ids.add(id(candidate))
        if side_ball_future is not None:
            self.side_crop_ball_candidates[frame_no] = len(side_ball_ids)
        people = merge_people(candidates)
        self.raw_people = len(people)
        people = [p for p in people if on_court(p, self.court)]
        if basketball_objects is not None and self.profile in {"broadcast", "moving"}:
            # The basketball-trained detector distinguishes on-court players from
            # officials/crowd. Pose alone cannot make that distinction.
            people = [p for p in people if is_player(p, basketball_objects)]
        poses, maps = [], []
        for person in people:
            # Require a measurable torso, not a minimum percentage of the entire frame.
            body = [person.landmarks[i] for i in (11, 12, 23, 24)]
            if min(p[2] for p in body) < .5:
                continue
            shoulder = np.mean([(p[0]*width, p[1]*height) for p in body[:2]], axis=0)
            hip = np.mean([(p[0]*width, p[1]*height) for p in body[2:]], axis=0)
            if np.linalg.norm(shoulder-hip) < 10:
                continue
            poses.append(PoseFrame(frame_no, time_s, {NAMES[i]: p for i, p in person.landmarks.items()}))
            maps.append(person.landmarks)
        if previous_ball and time_s - previous_ball.time_s < .3:
            predicted = (previous_ball.x, previous_ball.y)
            if prior_ball and previous_ball.frame - prior_ball.frame == frame_no - previous_ball.frame:
                predicted = (previous_ball.x + previous_ball.x - prior_ball.x,
                             previous_ball.y + previous_ball.y - prior_ball.y)
            velocity = np.hypot(previous_ball.x - (prior_ball.x if prior_ball else previous_ball.x),
                                 previous_ball.y - (prior_ball.y if prior_ball else previous_ball.y))
            allowed = max(.10, 2.5 * velocity + .06, 7.0 * previous_ball.radius)
            def cost(candidate):
                jump = np.hypot(candidate.x - predicted[0], candidate.y - predicted[1])
                if jump > allowed and candidate.confidence < max(.82, previous_ball.confidence + .08):
                    return -100.0
                return candidate.confidence - 3.0 * jump
            selected = max(balls, key=cost, default=None)
            ball = selected if selected is not None and cost(selected) > -50 else None
        else:
            ball = max(balls, key=lambda b: b.confidence, default=None)
        if side_ball_future is not None:
            self.side_crop_ball_selected[frame_no] = ball is not None and id(ball) in side_ball_ids
        self.timing_seconds["total"] += time.perf_counter() - started
        return poses, maps, ball


def is_player(person, objects):
    player_match = max((iou(person.box, box)*conf for cls, conf, box in objects
                        if cls in PLAYER_CLASSES and conf >= .4), default=0.)
    referee_match = max((iou(person.box, box)*conf for cls, conf, box in objects
                         if cls == REFEREE_CLASS and conf >= .4), default=0.)
    return player_match >= .25 and player_match > referee_match


def scene_cut(previous, current) -> bool:
    if previous is None:
        return False
    old = cv2.resize(previous, (96, 54)).astype(np.float32)
    new = cv2.resize(current, (96, 54)).astype(np.float32)
    return float(np.mean(np.abs(old-new))) > 45.


class CutDetector:
    """Confirm a cut only if the new image persists past one sampled frame."""

    def __init__(self):
        self.previous = None
        self.pending_frame = None
        self.reference = None

    def observe(self, frame: int, gray: np.ndarray) -> int | None:
        cut = None
        if self.pending_frame is not None:
            # A one-frame flash differs from its neighbors twice, but the
            # image after it still resembles the image before it.
            if scene_cut(self.reference, gray):
                cut = self.pending_frame
            self.pending_frame = None
            self.reference = None
        elif scene_cut(self.previous, gray):
            self.pending_frame = frame
            self.reference = self.previous
        self.previous = gray
        return cut
