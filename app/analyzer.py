from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from collections.abc import Callable
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from app.models import Detection, PoseFrame
from app.scoring import analyze_shots, classify_view

ROOT = Path(__file__).resolve().parents[1]
POSE_MODEL = ROOT / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
BALL_MODEL = ROOT / "models" / "yolo11n.pt"
BALL_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
POSE_NAMES = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
    14: "right_elbow", 15: "left_wrist", 16: "right_wrist", 23: "left_hip",
    24: "right_hip", 25: "left_knee", 26: "right_knee", 27: "left_ankle",
    28: "right_ankle",
}
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
            (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def _ensure_model(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".download")
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)
    return path


def ensure_pose_model() -> Path:
    return _ensure_model(POSE_MODEL, POSE_MODEL_URL)


def _device() -> str:
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


def _track_ball(model: YOLO, rgb: np.ndarray, previous: Detection | None, frame_no: int,
                time_s: float, width: int, height: int) -> Detection | None:
    result = model.predict(rgb, imgsz=640, conf=0.12, classes=[32], device=_device(), verbose=False)[0]
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
        return max(choices, key=lambda d: d.confidence - 2.0 * ((d.x - previous.x) ** 2 + (d.y - previous.y) ** 2) ** 0.5)
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


def _net_flow(previous_gray: np.ndarray | None, gray: np.ndarray, rim: tuple[float, float, float, float] | None) -> float:
    if previous_gray is None or rim is None:
        return 0.0
    height, width = gray.shape
    x, y, w, h = rim
    x1, x2 = max(0, int((x - 0.15 * w) * width)), min(width, int((x + 1.15 * w) * width))
    y1, y2 = max(0, int((y + 0.35 * h) * height)), min(height, int((y + 2.8 * h) * height))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0
    old, new = previous_gray[y1:y2, x1:x2], gray[y1:y2, x1:x2]
    flow = cv2.calcOpticalFlowFarneback(old, new, None, 0.5, 2, 13, 2, 5, 1.1, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    return float(np.median(magnitude))


def _draw(frame: np.ndarray, landmarks: dict[int, tuple[float, float, float]], ball: Detection | None,
          rim: tuple[float, float, float, float] | None) -> None:
    h, w = frame.shape[:2]
    for a, b in SKELETON:
        if a in landmarks and b in landmarks and landmarks[a][2] > .35 and landmarks[b][2] > .35:
            p1 = int(landmarks[a][0] * w), int(landmarks[a][1] * h)
            p2 = int(landmarks[b][0] * w), int(landmarks[b][1] * h)
            cv2.line(frame, p1, p2, (65, 235, 180), 3, cv2.LINE_AA)
    for idx in (12, 14, 16):
        if idx in landmarks and landmarks[idx][2] > .35:
            cv2.circle(frame, (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)), 5, (15, 245, 255), -1)
    if ball:
        center = int(ball.x * w), int(ball.y * h)
        cv2.circle(frame, center, max(7, int(ball.radius * max(w, h))), (30, 130, 255), 3, cv2.LINE_AA)
    if rim:
        x, y, rw, rh = rim
        cv2.rectangle(frame, (int(x*w), int(y*h)), (int((x+rw)*w), int((y+rh)*h)), (255, 180, 30), 2)


def analyze_video(input_path: Path, output_dir: Path, rim: tuple[float, float, float, float] | None,
                  progress: Callable[[float, str], None] | None = None) -> dict:
    report = progress or (lambda _value, _message: None)
    report(0.02, "Loading local vision models")
    pose_path = ensure_pose_model()
    configured_ball_model = os.environ.get("BALLFORM_YOLO_MODEL")
    ball_model_path = configured_ball_model or str(_ensure_model(BALL_MODEL, BALL_MODEL_URL))
    ball_model = YOLO(ball_model_path)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not decode this video. Try exporting it as H.264 MP4.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 64 or height < 64 or total < 3:
        raise ValueError("The uploaded video is empty or too small to analyze.")
    # Analyze at <=15 FPS, but retain source frame numbers/timestamps.
    stride = max(1, round(fps / 15.0))
    analyzed_fps = fps / stride
    raw_output = output_dir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), analyzed_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not initialize the annotated video writer.")

    options = mp.tasks.vision.PoseLandmarkerOptions(
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
    flows: dict[int, float] = {}
    previous_ball: Detection | None = None
    previous_gray: np.ndarray | None = None
    processed = 0
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_no = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_no += 1
            if frame_no % stride:
                continue
            time_s = frame_no / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(time_s * 1000))
            landmark_map: dict[int, tuple[float, float, float]] = {}
            if result.pose_landmarks:
                landmark_map = {i: (float(v.x), float(v.y), float(v.visibility or 0)) for i, v in enumerate(result.pose_landmarks[0])}
                poses.append(PoseFrame(frame_no, time_s, {name: landmark_map[idx] for idx, name in POSE_NAMES.items()}))
            ball = _track_ball(ball_model, rgb, previous_ball, frame_no, time_s, width, height)
            if ball:
                balls.append(ball)
                previous_ball = ball
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flows[frame_no] = _net_flow(previous_gray, gray, rim)
            previous_gray = gray
            _draw(frame, landmark_map, ball, rim)
            writer.write(frame)
            processed += 1
            if processed % 15 == 0:
                report(min(.88, .08 + .78 * frame_no / max(1, total)), f"Analyzing frame {frame_no:,} of {total:,}")
    capture.release()
    writer.release()
    balls = _interpolate_track(balls, max(2, round(fps * .25)))
    nonzero_flow = [v for v in flows.values() if v > 0]
    baseline = float(np.median(nonzero_flow)) if nonzero_flow else 1.0
    normalized_flow = {frame: value / max(baseline, 1e-5) for frame, value in flows.items()}
    shots = analyze_shots(balls, poses, fps, rim, normalized_flow)
    view, view_confidence = classify_view(poses)

    report(.92, "Encoding review video")
    annotated = output_dir / "annotated.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_output), "-c:v", "libx264",
                   "-preset", "fast", "-crf", "22", "-movflags", "+faststart", "-an", str(annotated)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0:
            raw_output.unlink(missing_ok=True)
        else:
            raw_output.replace(annotated)
    else:
        raw_output.replace(annotated)
    result = {
        "video": {"duration_s": round(total / fps, 2), "fps": round(fps, 2), "resolution": f"{width}×{height}",
                  "analyzed_fps": round(analyzed_fps, 2)},
        "camera_view": view, "camera_view_confidence": round(view_confidence, 2),
        "right_handed": True, "shots": [shot.to_dict() for shot in shots],
        "diagnostics": {"pose_frames": len(poses), "ball_detections": len([b for b in balls if b.confidence >= .12]),
                        "rim_marked": rim is not None},
        "limitations": [
            "Angles and distances are 2D image-plane estimates, not calibrated 3D measurements.",
            "Make/miss is inferred from visible ball/rim geometry and net motion; occlusion can lower confidence.",
            "Feedback is descriptive and should complement, not replace, coaching judgment.",
        ],
        "annotated_video": "annotated.mp4",
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    report(1.0, "Complete")
    return result
