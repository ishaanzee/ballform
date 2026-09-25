from __future__ import annotations

import json
import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import psutil
from ultralytics import YOLO

from app.models import Detection, PoseFrame
from app.game import analyze_game_shots
from app.scoring import RimInput, _rim_at, analyze_shots, classify_view
from app.tracking import BallHandlerTracker, HandlerDecision, PoseTracker, jersey_descriptor, stitch_tracks
from app.vision import COURT_PROFILES, CourtVision, CutDetector, camera_profile, regions, validate_court
from app.basketball import BasketballDetector, MODEL_FILENAME, MODEL_URL, MODEL_SHA256
from app.pose import EXPORT_SHAPES, CoreMLPose, TorchPose, export_path
from app.possession import decode_handlers

ROOT = Path(__file__).resolve().parents[1]
POSE_MODEL = ROOT / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
GAME_POSE_MODEL = ROOT / "models" / "pose_landmarker_full.task"
GAME_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
BALL_MODEL = ROOT / "models" / "yolo11n.pt"
BALL_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
MODEL_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
POSE_NAMES = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
    14: "right_elbow", 15: "left_wrist", 16: "right_wrist", 23: "left_hip",
    24: "right_hip", 25: "left_knee", 26: "right_knee", 27: "left_ankle",
    28: "right_ankle",
}
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
            (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


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


def _track_rim_bidirectional(input_path: Path, initial: tuple[float, float, float, float],
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


_VERIFIED_MODELS: dict[Path, tuple[int, int]] = {}


def _ensure_model(path: Path, url: str, sha256: str | None = None) -> Path:
    def valid(candidate):
        if not candidate.exists() or candidate.stat().st_size <= 1_000_000:
            return False
        if sha256 is None:
            return True
        stat = candidate.stat()
        # Re-hash only when the file changes, not on every job.
        if _VERIFIED_MODELS.get(candidate) == (stat.st_size, stat.st_mtime_ns):
            return True
        with candidate.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sha256:
                return False
        _VERIFIED_MODELS[candidate] = (stat.st_size, stat.st_mtime_ns)
        return True
    if valid(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".download")
    urllib.request.urlretrieve(url, partial)
    if not valid(partial):
        partial.unlink(missing_ok=True)
        raise ValueError(f"Model download failed validation: {path.name}")
    partial.replace(path)
    return path


def ensure_pose_model(game_mode: bool = False) -> Path:
    return _ensure_model(
        GAME_POSE_MODEL if game_mode else POSE_MODEL,
        GAME_POSE_MODEL_URL if game_mode else POSE_MODEL_URL,
    )


def _device() -> str:
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


_MODELS: dict[tuple, object] = {}
_MODELS_LOCK = threading.Lock()


def _shared_model(key: tuple, factory: Callable[[], object], loaded: list[str] | None = None):
    """Load each model once per process.

    Analyses run one at a time, so a cached session is never used by two jobs
    at once. Core ML re-specializes the detector on every session load (~12 s),
    which otherwise dominates short clips.
    """
    with _MODELS_LOCK:
        if key not in _MODELS:
            _MODELS[key] = factory()
            if loaded is not None:
                loaded.append(key[0])
        return _MODELS[key]


def _game_pose_model(name: str, loaded: list[str] | None = None
                     ) -> tuple[str, TorchPose | CoreMLPose, str | None]:
    """Return the pose weights path, the runner to use, and why Core ML is not used."""
    if name not in {"yolo26m-pose", "yolo26s-pose"}:
        raise ValueError("Unknown game pose model")
    request = os.environ.get("BALLFORM_POSE_BACKEND", "auto")
    if request not in {"auto", "coreml", "torch"}:
        raise ValueError("BALLFORM_POSE_BACKEND must be 'auto', 'coreml', or 'torch'.")
    selected_path = ROOT / "models" / f"{name}.pt"
    path = str(_ensure_model(selected_path, MODEL_RELEASE + selected_path.name))
    torch_pose = _shared_model(("pose", path), lambda: TorchPose(YOLO(path), _device()), loaded)
    if request == "torch":
        return path, torch_pose, None
    exports = {size: export_path(ROOT / "models", name, size) for size in EXPORT_SHAPES}
    missing = [p.name for p in exports.values() if not p.exists()]
    if missing:
        reason = f"Core ML pose exports not found ({', '.join(missing)}); run scripts/export_pose_coreml.py."
        if request == "coreml":
            raise RuntimeError(reason)
        return path, torch_pose, reason
    try:
        return path, _shared_model(("pose-coreml", name), lambda: CoreMLPose(exports, torch_pose), loaded), None
    except Exception as exc:
        if request == "coreml":
            raise
        reason = f"Core ML pose initialization failed: {type(exc).__name__}: {exc}"
        logging.warning("%s; using PyTorch pose", reason)
        return path, torch_pose, reason


def _basketball_detector(role: str, loaded: list[str] | None = None,
                         backend: str | None = None) -> tuple[str, BasketballDetector]:
    path = str(_ensure_model(ROOT / "models" / MODEL_FILENAME, MODEL_URL, MODEL_SHA256))
    backend = backend or os.environ.get("BALLFORM_BALL_BACKEND", "auto")
    units = os.environ.get("BALLFORM_COREML_UNITS", "CPUAndGPU")
    # "ball" and "prefetch" are separate sessions because they run concurrently.
    return path, _shared_model((role, path, backend, units),
                               lambda: BasketballDetector(path, backend=backend, compute_units=units), loaded)


def _pipeline_request() -> str:
    request = os.environ.get("BALLFORM_FRAME_PIPELINE", "auto")
    if request not in {"auto", "1", "2"}:
        raise ValueError("BALLFORM_FRAME_PIPELINE must be 'auto', '1', or '2'.")
    return request


def preload_game_models(pose_model: str = "yolo26s-pose") -> list[str]:
    """Load and warm the default game-mode models before the first job arrives."""
    loaded: list[str] = []
    device = _device()
    _, pose, _ = _game_pose_model(pose_model, loaded)
    blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # The first call at each input shape builds kernels or Core ML plans; do it now.
    for index, (x1, y1, x2, y2) in enumerate(regions(1920, 1080, True)):
        pose.infer(blank[y1:y2, x1:x2], 1280 if index == 0 else 960)
    if os.environ.get("BALLFORM_YOLO_MODEL"):
        return loaded
    _, ball = _basketball_detector("ball", loaded)
    ball.detect(blank)
    request = _pipeline_request()
    if request == "2" or (request == "auto" and ball.backend in {"coreml", "mlx"} and device == "mps"):
        _, prefetch = _basketball_detector("prefetch", loaded, backend=ball.backend)
        prefetch.detect(blank)
    return loaded


def _track_ball(model: YOLO, bgr: np.ndarray, previous: Detection | None,
                prior: Detection | None, frame_no: int, time_s: float,
                width: int, height: int) -> Detection | None:
    result = model.predict(bgr, imgsz=640, conf=0.12, classes=[32], device=_device(), verbose=False)[0]
    choices: list[Detection] = []
    if result.boxes is not None:
        for xyxy, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            choices.append(Detection(frame_no, time_s, (x1 + x2) / (2 * width),
                                     (y1 + y2) / (2 * height), float(conf),
                                     max(x2 - x1, y2 - y1) / (2 * max(width, height))))
    if not choices:
        return None
    if previous and frame_no - previous.frame < 12:
        predicted = (previous.x, previous.y)
        if prior and previous.frame - prior.frame == frame_no - previous.frame:
            predicted = (previous.x + previous.x - prior.x, previous.y + previous.y - prior.y)
        def continuity_cost(d):
            distance = float(np.hypot(d.x - predicted[0], d.y - predicted[1]))
            # A ball can move quickly, but a crowd false positive should not jump
            # across a large part of the frame for one low-confidence detection.
            velocity = float(np.hypot(previous.x - (prior.x if prior else previous.x),
                                      previous.y - (prior.y if prior else previous.y)))
            allowed = max(.10, 2.5 * velocity + .06, 7.0 * max(previous.radius, d.radius))
            if distance > allowed and d.confidence < max(.82, previous.confidence + .08):
                return -100.0
            return d.confidence - 3.0 * distance
        selected = max(choices, key=continuity_cost)
        return selected if continuity_cost(selected) > -50 else None
    return max(choices, key=lambda d: d.confidence)


def _interpolate_track(track: list[Detection], max_gap_frames: int) -> list[Detection]:
    if not track:
        return []
    output: list[Detection] = []
    for left, right in zip(track, track[1:]):
        output.append(left)
        gap = right.frame - left.frame
        if 1 < gap <= max_gap_frames:
            for n in range(1, gap):
                a = n / gap
                output.append(Detection(
                    frame=left.frame + n,
                    time_s=left.time_s + a * (right.time_s - left.time_s),
                    x=left.x + a * (right.x - left.x), y=left.y + a * (right.y - left.y),
                    confidence=min(left.confidence, right.confidence) * 0.75,
                    radius=left.radius + a * (right.radius - left.radius),
                ))
    output.append(track[-1])
    return output


def _flow_percentile(previous: np.ndarray, current: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 2, 13, 2, 5, 1.1, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    # A net is sparse: median flow is normally zero even when its strands move.
    return float(np.percentile(magnitude, 82))


def _net_flow(previous_gray: np.ndarray | None, gray: np.ndarray, rim: tuple[float, float, float, float] | None) -> dict[str, float]:
    """Measure net movement relative to its immediate background.

    The reference patch makes pans, zooms, and general player movement less
    likely to masquerade as a swish. Geometry remains the primary make signal.
    """
    if previous_gray is None or rim is None:
        return {"net": 0.0, "reference": 0.0}
    height, width = gray.shape
    x, y, w, h = rim
    def crop(x1: float, y1: float, x2: float, y2: float) -> tuple[np.ndarray, np.ndarray] | None:
        left, right = max(0, int(x1 * width)), min(width, int(x2 * width))
        top, bottom = max(0, int(y1 * height)), min(height, int(y2 * height))
        if right - left < 8 or bottom - top < 8:
            return None
        return previous_gray[top:bottom, left:right], gray[top:bottom, left:right]

    # The lower central portion is where a hanging net can move. Side patches
    # sample the same local camera/background movement without the net itself.
    net = crop(x + .14 * w, y + .42 * h, x + .86 * w, y + 2.45 * h)
    left = crop(x - .55 * w, y + .45 * h, x - .08 * w, y + 2.2 * h)
    right = crop(x + 1.08 * w, y + .45 * h, x + 1.55 * w, y + 2.2 * h)
    net_value = _flow_percentile(*net) if net else 0.0
    reference_values = [_flow_percentile(*patch) for patch in (left, right) if patch]
    return {"net": net_value, "reference": float(np.median(reference_values)) if reference_values else 0.0}


def _normalize_net_flow(flows: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    net_values = [item["net"] for item in flows.values() if item["net"] > 0]
    reference_values = [item["reference"] for item in flows.values() if item["reference"] > 0]
    net_baseline = float(np.median(net_values)) if net_values else 1.0
    reference_baseline = float(np.median(reference_values)) if reference_values else 1.0
    normalized = {}
    for frame, item in flows.items():
        net_ratio = item["net"] / max(net_baseline, 1e-4)
        reference_ratio = item["reference"] / max(reference_baseline, 1e-4)
        # Clamp the divisor so a quiet reference patch cannot create an infinite
        # response; this is a relative signal, not a make classifier.
        normalized[frame] = {"strength": net_ratio / max(.7, reference_ratio), **item}
    return normalized


def _draw_make_animation(frame: np.ndarray, rim: tuple[float, float, float, float], elapsed_s: float,
                         likely: bool = False) -> None:
    """Overlay a short green pulse at a detected make without hiding the play."""
    if elapsed_s < -.06 or elapsed_s > .85:
        return
    height, width = frame.shape[:2]
    x, y, rw, rh = rim
    center = (int((x + .5 * rw) * width), int((y + .55 * rh) * height))
    phase = max(0.0, elapsed_s) / .85
    pulse = 1.0 - phase
    overlay = frame.copy()
    axes = (max(14, int(rw * width * (.8 + 1.8 * phase))), max(10, int(rh * height * (1.5 + 2.0 * phase))))
    cv2.ellipse(overlay, center, axes, 0, 0, 360, (50, 255, 70), max(2, int(7 * pulse)), cv2.LINE_AA)
    cv2.ellipse(overlay, center, (max(9, axes[0] // 2), max(7, axes[1] // 2)), 0, 0, 360, (40, 230, 60), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, .16 + .22 * pulse, frame, .84 - .22 * pulse, 0, frame)
    label = "LIKELY MAKE" if likely else "MAKE"
    cv2.putText(frame, label, (max(12, center[0] - axes[0]), max(30, center[1] - axes[1] - 12)),
                cv2.FONT_HERSHEY_DUPLEX, .75, (70, 255, 90), 2, cv2.LINE_AA)


def _shooter_intervals(shots: list, fps: float) -> list[tuple[int, int, int]]:
    """Return source-frame intervals from each identified release through its rim event."""
    intervals = []
    for shot in shots:
        game = shot.game or {}
        shooter = (game.get("players") or {}).get("shooter_track_id")
        if shooter is None:
            continue
        start = game.get("release_frame")
        start = round(shot.release_s * fps) if start is None else int(start)
        end = shot.outcome_frame if shot.outcome_frame is not None else round(shot.end_s * fps)
        if end >= start:
            intervals.append((start, int(end), int(shooter)))
    return intervals


def _review_highlight(source_frame: int, frame_data: dict,
                      shooter_intervals: list[tuple[int, int, int]]) -> tuple[int | None, str | None]:
    # A released shot belongs to its shooter until the rim event, even if another
    # player becomes the likely next handler in the meantime.
    for start, end, shooter_id in shooter_intervals:
        if start <= source_frame <= end:
            return shooter_id, f"P{shooter_id} SHOOTER"
    decision = frame_data.get("handler")
    if decision is None or decision.track_id is None:
        return None, None
    suffix = "BALL" if decision.source == "observed" else "BALL?"
    return decision.track_id, f"P{decision.track_id} {suffix}"


def _add_review_overlays(source: Path, destination: Path, shots: list, player_frames: list[dict],
                         output_fps: float, source_fps: float, rim: RimInput,
                         label_players: bool = False) -> None:
    shooter_intervals = _shooter_intervals(shots, source_fps)
    has_makes = rim is not None and any(
        s.outcome in {"made", "likely made"} and s.outcome_frame is not None for s in shots)
    has_handlers = any(f.get("handler") is not None and f["handler"].track_id is not None
                       for f in player_frames)
    label_players = label_players and any(p.track_id for f in player_frames for p in f["players"])
    if not shooter_intervals and not has_makes and not has_handlers and not label_players:
        source.replace(destination)
        return
    capture = cv2.VideoCapture(str(source))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        source.replace(destination)
        return
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        time_s = index / output_fps
        source_frame = (player_frames[index]["frame"] if index < len(player_frames)
                        else round(time_s * source_fps))
        if index < len(player_frames):
            frame_data = player_frames[index]
            highlighted_id, label = _review_highlight(source_frame, frame_data, shooter_intervals)
            for pose in frame_data["players"]:
                highlighted = pose.track_id is not None and pose.track_id == highlighted_id
                if not highlighted and not (label_players and pose.track_id):
                    continue
                landmarks = {idx: pose.landmarks[name] for idx, name in POSE_NAMES.items()
                             if name in pose.landmarks}
                if highlighted:
                    _draw(frame, landmarks, None, None, label=label, handler=True)
                else:
                    _draw_label(frame, landmarks, f"P{pose.track_id}")
        frame_rim = _rim_at(rim, round(time_s * source_fps))
        if frame_rim is not None:
            for shot in shots:
                if shot.outcome in {"made", "likely made"} and shot.outcome_frame is not None:
                    _draw_make_animation(frame, frame_rim, time_s - shot.outcome_frame / source_fps,
                                         shot.outcome == "likely made")
        writer.write(frame)
        index += 1
    capture.release()
    writer.release()
    source.unlink(missing_ok=True)


def _draw_label(frame: np.ndarray, landmarks: dict[int, tuple[float, float, float]], label: str,
                color: tuple[int, int, int] = (65, 235, 180), boxed: bool = False) -> None:
    h, w = frame.shape[:2]
    anchor = next((landmarks[idx] for idx in (0, 11, 12) if idx in landmarks and landmarks[idx][2] > .35), None)
    if not anchor:
        return
    x, y = int(anchor[0] * w), max(20, int(anchor[1] * h) - 10)
    if boxed:
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)
        cv2.rectangle(frame, (max(0, x - 3), max(0, y - text_h - 4)),
                      (min(w - 1, x + text_w + 3), min(h - 1, y + baseline + 3)),
                      (25, 25, 25), -1)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)


def _draw(frame: np.ndarray, landmarks: dict[int, tuple[float, float, float]], ball: Detection | None,
          rim: tuple[float, float, float, float] | None, handedness: str = "right",
          label: str | None = None, handler: bool = False) -> None:
    h, w = frame.shape[:2]
    pose_color = (35, 45, 245) if handler else (65, 235, 180)
    for a, b in SKELETON:
        if a in landmarks and b in landmarks and landmarks[a][2] > .35 and landmarks[b][2] > .35:
            p1 = int(landmarks[a][0] * w), int(landmarks[a][1] * h)
            p2 = int(landmarks[b][0] * w), int(landmarks[b][1] * h)
            cv2.line(frame, p1, p2, pose_color, 3, cv2.LINE_AA)
    for idx in ((12, 14, 16) if handedness == "right" else (11, 13, 15)):
        if idx in landmarks and landmarks[idx][2] > .35:
            cv2.circle(frame, (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)), 5,
                       pose_color if handler else (15, 245, 255), -1)
    if label and landmarks:
        _draw_label(frame, landmarks, label, pose_color, boxed=handler)
    if ball:
        center = int(ball.x * w), int(ball.y * h)
        cv2.circle(frame, center, max(7, int(ball.radius * max(w, h))), (30, 130, 255), 3, cv2.LINE_AA)
    if rim:
        x, y, rw, rh = rim
        cv2.rectangle(frame, (int(x*w), int(y*h)), (int((x+rw)*w), int((y+rh)*h)), (255, 180, 30), 2)


def _sampled_frames(capture, stride: int, detector=None, executor=None,
                    prefetch_waits: dict[int, float] | None = None):
    """Yield source frames in order while detecting the next frame in the background."""
    source_frame = -1

    def read_next():
        nonlocal source_frame
        while True:
            ok, frame = capture.read()
            if not ok:
                return None
            source_frame += 1
            if source_frame % stride == 0:
                return source_frame, frame

    def detect_timed(frame):
        started = time.perf_counter()
        return detector.detect(frame), time.perf_counter() - started

    current = read_next()
    if detector is None:
        while current is not None:
            yield (*current, None, 0.0)
            current = read_next()
        return
    current_future = executor.submit(detect_timed, current[1]) if current is not None else None
    while current is not None:
        following = read_next()
        following_future = executor.submit(detect_timed, following[1]) if following is not None else None
        wait_started = time.perf_counter()
        objects, seconds = current_future.result()
        if prefetch_waits is not None:
            prefetch_waits[current[0]] = time.perf_counter() - wait_started
        yield (*current, objects, seconds)
        current, current_future = following, following_future


def _wait_summary(samples: list[float]) -> dict:
    """Summarize time the frame thread was blocked, not total worker runtime."""
    if not samples:
        return {"frames": 0, "blocked_frames_over_1ms": 0, "total_seconds": 0.0,
                "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {"frames": len(samples),
            "blocked_frames_over_1ms": sum(value >= .001 for value in samples),
            "total_seconds": round(sum(samples), 3),
            "median_ms": round(float(np.percentile(samples, 50)) * 1000, 2),
            "p95_ms": round(float(np.percentile(samples, 95)) * 1000, 2),
            "max_ms": round(max(samples) * 1000, 2)}


def _applied_court(court, mode: str, profile: str) -> tuple[list | None, str | None]:
    """Return the playing-area polygon to apply and, if one was dropped, why.

    A drawn polygon is fixed in image coordinates, so it cannot follow a panning
    game camera; applied anyway it cut real players standing on the sideline in
    most frames of test clips. Game footage from panning profiles relies on the
    detector's player/referee classes instead.
    """
    if court is None or mode != "one_on_one" or profile in COURT_PROFILES:
        return court, None
    return None, (f"Playing-area polygon ignored for the {profile} camera profile: it is fixed in "
                  "image coordinates and cannot follow camera pans. Players are separated from "
                  "referees and spectators by the basketball detector.")


def _analyze_video(input_path: Path, output_dir: Path, rim: tuple[float, float, float, float] | None,
                  progress: Callable[[float, str], None] | None = None,
                  mode: str = "form", handedness: str = "right", camera: str = "courtside",
                  court: list[list[float]] | None = None, rim_frame: int | None = None,
                  rim_time_s: float | None = None, pose_model: str = "yolo26s-pose") -> dict:
    if mode not in {"form", "one_on_one"} or handedness not in {"left", "right"}:
        raise ValueError("Invalid analysis mode or shooting hand")
    report = progress or (lambda _value, _message: None)
    profile = camera_profile(camera)
    court, court_ignored = _applied_court(validate_court(court), mode, profile)
    game_mode = mode == "one_on_one"
    stage_times = {}
    stage_started = time.perf_counter()
    report(0.02, "Loading local vision models")
    pose_path = None if game_mode else ensure_pose_model()
    device = _device()
    game_pose_path = None
    pose_fallback = None
    loaded_models: list[str] = []
    if game_mode:
        game_pose_path, game_pose_model, pose_fallback = _game_pose_model(pose_model, loaded_models)
        pose_fallback_calls_before = getattr(game_pose_model, "fallback_calls", 0)
        report(.04, f"Pose model loaded: {Path(game_pose_path).name}")
    configured_ball_model = os.environ.get("BALLFORM_YOLO_MODEL")
    if game_mode and not configured_ball_model:
        ball_model_path, ball_model = _basketball_detector("ball", loaded_models)
    else:
        ball_model_path = configured_ball_model or str(_ensure_model(BALL_MODEL, BALL_MODEL_URL))
        ball_model = _shared_model(("yolo", ball_model_path), lambda: YOLO(ball_model_path), loaded_models)
    pipeline_request = _pipeline_request()
    pipeline_depth = (2 if game_mode and isinstance(ball_model, BasketballDetector)
                      and ball_model.backend in {"coreml", "mlx"} and device == "mps" else 1) if pipeline_request == "auto" else int(pipeline_request)
    prefetch_detector = None
    pipeline_fallback = None
    if pipeline_depth == 2:
        if not game_mode or not isinstance(ball_model, BasketballDetector):
            raise ValueError("Two-frame inference requires the basketball-trained game detector.")
        try:
            # A second session is owned by the prefetch worker; the first session
            # remains available for side crops on the current frame.
            _, prefetch_detector = _basketball_detector("prefetch", loaded_models, backend=ball_model.backend)
        except Exception as exc:
            if pipeline_request == "2":
                raise
            pipeline_fallback = f"Prefetch detector initialization failed: {type(exc).__name__}: {exc}"
            pipeline_depth = 1
    court_vision = CourtVision(game_pose_model, ball_model, device, profile, court) if game_mode else None
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not decode this video. Try exporting it as H.264 MP4.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 64 or height < 64 or total < 3:
        capture.release()
        raise ValueError("The uploaded video is empty or too small to analyze.")
    # Preserve more release/contest detail than the original 15 FPS pipeline.
    stride = max(1, int(np.ceil(fps / 30.0)))
    analyzed_fps = fps / stride
    raw_output = output_dir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), analyzed_fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not initialize the annotated video writer.")
    stage_times["model_loading_and_setup"] = time.perf_counter() - stage_started

    options = None if game_mode else mp.tasks.vision.PoseLandmarkerOptions(
        # Explicit CPU delegate avoids MediaPipe attempting to create a Metal/OpenGL
        # context in a headless web-server process on macOS.
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(pose_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1, min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45, min_tracking_confidence=0.45,
    )
    balls: list[Detection] = []
    poses: list[PoseFrame] = []
    player_frames: list[dict] = []
    flows: dict[int, dict[str, float]] = {}
    previous_ball: Detection | None = None
    prior_ball: Detection | None = None
    previous_gray: np.ndarray | None = None
    processed = 0
    cut_frames = []
    cut_detector = CutDetector() if game_mode else None
    raw_people_counts = []
    tracked_rims: dict[int, tuple[float, float, float, float]] = {}
    rim_tracking_lost = False
    stage_started = time.perf_counter()
    if profile == "moving" and rim:
        report(.06, "Tracking the marked rim forward and backward")
        tracked_rims, rim_tracking_lost = _track_rim_bidirectional(
            input_path, rim, (rim_frame if rim_frame is not None else
                              round((rim_time_s or 0.0) * fps)), width, height, total)
    stage_times["rim_tracking"] = time.perf_counter() - stage_started
    pose_tracker = PoseTracker(width / height, max_gap_frames=max(3, round(fps * .7)))
    handler_tracker = BallHandlerTracker(width / height) if game_mode else None
    prefetch_ball_seconds = 0.0
    prefetched_frames = 0
    prefetch_waits: dict[int, float] = {}
    stage_started = time.perf_counter()
    with ExitStack() as resources:
        resources.callback(capture.release)
        resources.callback(writer.release)
        if court_vision:
            resources.callback(court_vision.close)
        landmarker = None if game_mode else resources.enter_context(mp.tasks.vision.PoseLandmarker.create_from_options(options))
        prefetch_executor = (resources.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix="ballform-next-frame"))
                             if prefetch_detector else None)
        frame_no = -1
        for frame_no, frame, prefetched_objects, prefetch_seconds in _sampled_frames(
                capture, stride, prefetch_detector, prefetch_executor, prefetch_waits):
            if prefetched_objects is not None:
                prefetched_frames += 1
                prefetch_ball_seconds += prefetch_seconds
            time_s = frame_no / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cut_frame = cut_detector.observe(frame_no, gray) if cut_detector else None
            if cut_frame is not None:
                cut_frames.append(cut_frame)
                pose_tracker.reset()
                # The cut is confirmed one sample late, so its first frame was matched
                # against the previous shot's tracks; give it fresh identities.
                if player_frames and player_frames[-1]["frame"] == cut_frame:
                    for pose in player_frames[-1]["players"]:
                        pose.track_id = None
                    pose_tracker.update(player_frames[-1]["players"])
                if handler_tracker:
                    handler_tracker.reset()
                previous_ball = None
                prior_ball = None
            frame_rim = tracked_rims.get(frame_no) if profile == "moving" else rim
            players = []
            landmark_maps = []
            if court_vision:
                players, landmark_maps, ball = court_vision.detect(
                    frame, frame_no, time_s, previous_ball, prior_ball, prefetched_objects=prefetched_objects)
                raw_people_counts.append(court_vision.raw_people)
                for pose in players:
                    pose.appearance = jersey_descriptor(frame, pose)
                pose_tracker.update(players)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time_s * 1000))
                for pose_landmarks in result.pose_landmarks:
                    landmark_map = {i: (float(v.x), float(v.y), float(v.visibility or 0)) for i, v in enumerate(pose_landmarks)}
                    landmark_maps.append(landmark_map)
                    players.append(PoseFrame(frame_no, time_s, {name: landmark_map[idx] for idx, name in POSE_NAMES.items()}))
                ball = _track_ball(ball_model, frame, previous_ball, prior_ball, frame_no, time_s, width, height)
            poses.extend(players)
            handler = handler_tracker.update(players, ball, time_s) if handler_tracker else None
            player_frames.append({"frame": frame_no, "time_s": time_s, "players": players,
                                  "handler": handler, "ball": ball,
                                  "possession": court_vision.possession if court_vision else [],
                                  "unposed": court_vision.unposed if court_vision else []})
            if ball:
                balls.append(ball)
                prior_ball = previous_ball
                previous_ball = ball
            flows[frame_no] = _net_flow(previous_gray, gray, frame_rim)
            previous_gray = gray
            # Game-mode player labels are drawn in the review pass, after track stitching.
            for visible_pose in landmark_maps:
                _draw(frame, visible_pose, None, None, handedness)
            _draw(frame, {}, ball, frame_rim)
            if court:
                polygon = np.rint(np.asarray(court) * (width, height)).astype(np.int32)
                cv2.polylines(frame, [polygon], True, (255, 210, 80), 2)
            writer.write(frame)
            processed += 1
            if processed % 5 == 0:
                report(min(.88, .08 + .78 * frame_no / max(1, total)), f"Analyzing frame {frame_no:,} of {total:,}")
    stage_times["frame_processing"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    observed_balls = list(balls)
    handler_method = os.environ.get("BALLFORM_HANDLER", "smoothed")
    if handler_method not in {"smoothed", "online"}:
        raise ValueError("BALLFORM_HANDLER must be 'smoothed' or 'online'.")
    stitched: dict[int, int] = {}
    if game_mode:
        edges = [0, *cut_frames, frame_no + 1]
        segments = [[f for f in player_frames if start <= f["frame"] < end] for start, end in zip(edges, edges[1:])]
        # Rejoin fragments of the same player before anything reads identities
        # (handler decoding, game analysis, review labels). Identities reset at cuts.
        for segment in segments:
            stitched.update(stitch_tracks(segment, width / height, analyzed_fps))
        for frame_data in player_frames:
            online = frame_data["handler"]
            if online is not None and online.track_id in stitched:
                online = HandlerDecision(stitched[online.track_id], online.source)
            frame_data["handler_online"] = frame_data["handler"] = online
        if handler_method == "smoothed":
            for segment in segments:
                for frame_data, decision in zip(segment, decode_handlers(segment, width / height)):
                    frame_data["handler"] = decision
        # Keep measured inputs for scoring review; handler is a separate, inferred
        # visual-annotation decision and is never fed back into shot scoring.
        (output_dir / "observations.json").write_text(json.dumps({
            "fps": fps, "width": width, "height": height, "cuts": cut_frames,
            "balls": [asdict(b) for b in observed_balls],
            "players": [{"frame": f["frame"], "time_s": f["time_s"],
                         "players": [asdict(p) for p in f["players"]],
                         "handler": asdict(f["handler"]) if f["handler"] else None,
                         "handler_online": asdict(f["handler_online"]) if f["handler_online"] else None,
                         "possession": f["possession"], "unposed": f["unposed"]}
                        for f in player_frames],
        }, default=lambda value: float(value)))
    normalized_flow = _normalize_net_flow(flows)
    # A fixed rim box is unsafe on an elevated game view. Moving mode instead uses a CSRT box
    # at each source frame and stops supplying it after a cut or tracking loss.
    scoring_rim: RimInput = (tracked_rims if profile == "moving" and tracked_rims else
                             None if game_mode and profile == "elevated" else rim)
    shots = []
    boundaries = [0, *cut_frames, frame_no + 1]
    for start, end in zip(boundaries, boundaries[1:]):
        segment_balls = _interpolate_track([b for b in balls if start <= b.frame < end], max(2, round(fps * .25)))
        segment_poses = [p for p in poses if start <= p.frame < end]
        segment_shots = analyze_shots(segment_balls, segment_poses, fps, scoring_rim, normalized_flow,
                                     aspect_ratio=width / height, handedness=handedness, game_mode=game_mode)
        for shot in segment_shots:
            if game_mode and profile == "elevated":
                shot.evidence = ["Outcome unavailable: the elevated profile does not track a rim; use moving (with a marked rim) or courtside."
                                 if item == "Outcome unavailable because the rim was not marked" else item
                                 for item in shot.evidence]
            shot.number = len(shots) + 1
            shots.append(shot)
    game_summary = None
    if mode == "one_on_one":
        # Multi-person pose order is not a player identity. Do not publish mixed-player form metrics.
        for shot in shots:
            shot.metrics = {}
            shot.cues = []
        game_summary = analyze_game_shots(shots, player_frames, observed_balls, fps, width / height,
                                          min_torso=10 / height)
    view, view_confidence = classify_view(poses, aspect_ratio=width / height)
    if game_mode:
        view, view_confidence = profile, 0.0  # User-selected profile, not inferred calibration.
    stage_times["scoring_and_observations"] = time.perf_counter() - stage_started

    report(.92, "Marking verified makes in review video")
    stage_started = time.perf_counter()
    marked_output = output_dir / "annotated_marked.mp4"
    _add_review_overlays(raw_output, marked_output, shots, player_frames, analyzed_fps, fps, scoring_rim,
                         label_players=game_mode)
    stage_times["review_overlays"] = time.perf_counter() - stage_started
    report(.95, "Encoding review video")
    stage_started = time.perf_counter()
    annotated = output_dir / "annotated.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(marked_output), "-c:v", "libx264",
                   "-preset", "fast", "-crf", "22", "-movflags", "+faststart", "-an", str(annotated)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            marked_output.unlink(missing_ok=True)
        else:
            marked_output.replace(annotated)
    else:
        marked_output.replace(annotated)
    stage_times["video_encoding"] = time.perf_counter() - stage_started
    if court_vision:
        stage_times["pose_inference_and_readback"] = court_vision.timing_seconds["pose"]
        stage_times["ball_inference_and_readback"] = court_vision.timing_seconds["ball"] + prefetch_ball_seconds
        stage_times["prefetched_full_ball_inference"] = prefetch_ball_seconds
        stage_times["vision_wall_time"] = court_vision.timing_seconds["total"]
    side_crop_waits = court_vision.side_crop_waits if court_vision else {}
    pipeline_waits = {
        "prefetched_full_ball": _wait_summary(list(prefetch_waits.values())),
        "side_ball_crops": _wait_summary(list(side_crop_waits.values())),
        "side_crop_ball_candidates": sum(court_vision.side_crop_ball_candidates.values()) if court_vision else 0,
        "side_crop_selected_frames": sum(court_vision.side_crop_ball_selected.values()) if court_vision else 0,
        "combined_wait_seconds": round(sum(prefetch_waits.values()) + sum(side_crop_waits.values()), 3),
        "fraction_of_frame_processing": round(
            (sum(prefetch_waits.values()) + sum(side_crop_waits.values())) /
            max(stage_times["frame_processing"], 1e-9), 3),
    }
    if game_mode:
        (output_dir / "pipeline_timing.json").write_text(json.dumps({
            "note": "Waits measure only time the frame thread blocked on an inference result. "
                    "Zero means the worker finished before its result was needed.",
            "frames": [{"frame": f["frame"],
                        "prefetched_full_ball_wait_ms": round(prefetch_waits.get(f["frame"], 0) * 1000, 2),
                        "side_ball_crops_wait_ms": round(side_crop_waits.get(f["frame"], 0) * 1000, 2),
                        "side_ball_crops_triggered": f["frame"] in side_crop_waits,
                        "side_ball_candidates": court_vision.side_crop_ball_candidates.get(f["frame"], 0),
                        "selected_ball_from_side_crop": court_vision.side_crop_ball_selected.get(f["frame"], False)}
                       for f in player_frames],
        }, indent=2))
    result = {
        "video": {"duration_s": round(total / fps, 2), "fps": round(fps, 2), "resolution": f"{width}×{height}",
                  "analyzed_fps": round(analyzed_fps, 2)},
        "camera_view": view, "camera_view_confidence": round(view_confidence, 2),
        "mode": mode, "handedness": handedness, "game_summary": game_summary,
        "camera_profile": profile, "court_polygon": court, "court_polygon_ignored": court_ignored,
        "vision": {"pose_model": Path(game_pose_path).name if game_pose_path else pose_path.name,
                   "pose_model_requested": pose_model if game_mode else None,
                   "pose_model_choice": pose_model if game_mode else None,
                   "ball_model": Path(ball_model_path).name, "device": device,
                   "ball_device": ({"coreml": f"coreml_{ball_model.compute_units}_requested", "mlx": "gpu_mlx_metal"}.get(ball_model.backend, "cpu"))
                   if isinstance(ball_model, BasketballDetector) else device,
                   "ball_backend": ball_model.backend if isinstance(ball_model, BasketballDetector) else "ultralytics",
                   "ball_backend_requested": ball_model.requested_backend if isinstance(ball_model, BasketballDetector) else None,
                   "ball_backend_fallback": ball_model.fallback_reason if isinstance(ball_model, BasketballDetector) else None,
                   "ball_execution_providers": ball_model.providers if isinstance(ball_model, BasketballDetector) else None,
                   "models_loaded_this_job": loaded_models,
                   "pose_backend": court_vision.pose_model.backend if court_vision else "mediapipe",
                   "pose_backend_requested": os.environ.get("BALLFORM_POSE_BACKEND", "auto") if game_mode else None,
                   "pose_backend_fallback": pose_fallback,
                   "pose_calls_on_torch_fallback": (getattr(court_vision.pose_model, "fallback_calls", 0)
                                                    - pose_fallback_calls_before) if court_vision else None,
                   "frame_pipeline_depth": pipeline_depth,
                   "frame_pipeline_requested": pipeline_request,
                   "frame_pipeline_fallback": pipeline_fallback,
                   "prefetch_ball_backend": prefetch_detector.backend if prefetch_detector else None,
                   "prefetch_ball_fallback": prefetch_detector.fallback_reason if prefetch_detector else None,
                   "input_size": court_vision.imgsz if court_vision else 640,
                   "ball_input_size": 640 if isinstance(ball_model, BasketballDetector) else (court_vision.imgsz if court_vision else 640),
                   "broadcast_role_filter": bool(isinstance(ball_model, BasketballDetector) and profile == "moving"),
                   "overlapping_crops": bool(court_vision and court_vision.tiled),
                   "ball_crops_overlapped_with_pose": bool(court_vision and court_vision.ball_crop_overlap_frames)},
        "right_handed": handedness == "right", "shots": [shot.to_dict() for shot in shots],
        "diagnostics": {"pose_frames": sum(bool(p["players"]) for p in player_frames),
                        "two_player_frames": sum(len(p["players"]) == 2 for p in player_frames),
                        "multi_player_frames": sum(len(p["players"]) > 2 for p in player_frames),
                        "max_players_visible": max((len(p["players"]) for p in player_frames), default=0),
                        "max_people_detected": max(raw_people_counts, default=0),
                        "scene_cuts": len(cut_frames), "scene_cut_times_s": [round(f / fps, 3) for f in cut_frames],
                        "court_filtered": court is not None,
                        "ball_detections": len(observed_balls),
                        "handler_observed_frames": sum(f["handler"] is not None and f["handler"].source == "observed"
                                                       for f in player_frames),
                        "handler_method": handler_method if game_mode else None,
                        "tracks_stitched": len(stitched) if game_mode else None,
                        "handler_differs_from_online_frames": sum(
                            (f["handler"].track_id if f["handler"] else None)
                            != (f["handler_online"].track_id if f["handler_online"] else None)
                            for f in player_frames) if game_mode else None,
                        "handler_held_frames": sum(f["handler"] is not None and f["handler"].source == "held"
                                                   for f in player_frames),
                        "rim_marked": rim is not None,
                        "rim_tracking_frames": len(tracked_rims),
                        "rim_tracking_lost": rim_tracking_lost,
                        "pose_predict_calls": court_vision.pose_predict_calls if court_vision else None,
                        "pose_images": court_vision.pose_images if court_vision else None,
                        "ball_crop_overlap_frames": court_vision.ball_crop_overlap_frames if court_vision else None,
                        "prefetched_full_ball_frames": prefetched_frames},
        "performance": {
            "stages_seconds": {name: round(seconds, 2) for name, seconds in stage_times.items()},
            "pipeline_waits": pipeline_waits,
            "pipeline_timing_file": "pipeline_timing.json" if game_mode else None,
            "timing_note": "Pose and ball times are sums of inference work and can overlap each other and adjacent frames; do not add them to wall times. Vision wall time excludes prefetched full-frame detection. GPU timing includes tensor readback.",
        },
        "limitations": [
            "Angles and distances are 2D image-plane estimates, not calibrated 3D measurements.",
            "Make/miss is inferred from visible ball/rim geometry. Net movement only supports a trajectory that reaches the rim, so a net-only airball is not called a make.",
            "Feedback is descriptive and should complement, not replace, coaching judgment.",
            "Shot candidates and outcomes require video review; passes, occlusion and camera movement can cause errors.",
        ],
        "annotated_video": "annotated.mp4",
    }
    if mode == "one_on_one":
        result["limitations"].extend(game_summary.get("limitations", []))
        if court_ignored:
            result["limitations"].append(court_ignored)
        elif court is None and profile == "moving":
            result["limitations"].append("No playing-area polygon applied. The basketball detector separates players from referees and spectators and can make mistakes.")
        elif court is None:
            result["limitations"].append("No playing-area polygon applied. With this fixed camera, spectators and bench players in view can be counted as players; mark the playing area to exclude them.")
        if profile == "elevated":
            result["limitations"].append("Elevated camera profile is not court calibration. Pans and zooms affect projected trajectories and spacing. Make/miss is withheld because a fixed rim box cannot follow a moving camera; use courtside for a stationary clip, or moving with a marked rim for a panning one.")
        if profile == "moving":
            result["limitations"].append("Moving-camera outcomes use a user-initialized visual rim tracker. It supports continuous pans and moderate zooms, but withholds outcomes after tracking loss or a camera cut; mark the rim on any clear frame and review every result.")
        result["limitations"].append("Game shots require raised-hand ball contact and an arc rising above the release shoulders. Fully occluded releases, flat arcs and underhand shots may be omitted; passes and slow-motion edits need manual review.")
        result["limitations"].append("Ball-handler highlights are decoded over the whole clip from hand contact, dribble position and the detector's possession class; BALL? marks frames held without direct evidence. They are uncertain visual annotations, not measured possession or scoring evidence.")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    report(1.0, "Complete")
    return result


def analyze_video(input_path: Path, output_dir: Path, rim: tuple[float, float, float, float] | None,
                  progress: Callable[[float, str], None] | None = None,
                  mode: str = "form", handedness: str = "right", camera: str = "courtside",
                  court: list[list[float]] | None = None, rim_frame: int | None = None,
                  rim_time_s: float | None = None, pose_model: str = "yolo26s-pose") -> dict:
    """Run analysis and append elapsed time plus sampled peak process memory."""
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(.1):
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                return

    sampler = threading.Thread(target=sample_memory, name="ballform-memory-sampler", daemon=True)
    started = time.perf_counter()
    sampler.start()
    try:
        result = _analyze_video(input_path, output_dir, rim, progress, mode, handedness, camera,
                                court, rim_frame, rim_time_s, pose_model)
    finally:
        stop.set()
        sampler.join()
        try:
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        except psutil.Error:
            pass
    result.setdefault("performance", {}).update({
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_process_memory_mb": round(peak_rss[0] / (1024 * 1024), 1),
        "memory_note": "Peak resident memory of the Ballform process sampled during this analysis; includes the app and loaded models.",
    })
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
