from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import urllib.request
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from app.models import Detection, PoseFrame
from app.game import analyze_game_shots
from app.scoring import analyze_shots, classify_view
from app.tracking import PoseTracker, jersey_descriptor
from app.vision import CourtVision, camera_profile, scene_cut, validate_court
from app.basketball import BasketballDetector, MODEL_FILENAME, MODEL_URL, MODEL_SHA256

ROOT = Path(__file__).resolve().parents[1]
POSE_MODEL = ROOT / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
GAME_POSE_MODEL = ROOT / "models" / "pose_landmarker_full.task"
GAME_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
BALL_MODEL = ROOT / "models" / "yolo11n.pt"
BALL_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
COURT_POSE_MODEL = ROOT / "models" / "yolo11m-pose.pt"
MODEL_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
POSE_NAMES = {
    0: "nose", 11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
    14: "right_elbow", 15: "left_wrist", 16: "right_wrist", 23: "left_hip",
    24: "right_hip", 25: "left_knee", 26: "right_knee", 27: "left_ankle",
    28: "right_ankle",
}
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
            (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def _ensure_model(path: Path, url: str, sha256: str | None = None) -> Path:
    def valid(candidate):
        if not candidate.exists() or candidate.stat().st_size <= 1_000_000:
            return False
        if sha256 is None:
            return True
        with candidate.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == sha256
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


def _track_ball(model: YOLO, bgr: np.ndarray, previous: Detection | None, frame_no: int,
                time_s: float, width: int, height: int) -> Detection | None:
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
          rim: tuple[float, float, float, float] | None, handedness: str = "right",
          label: str | None = None) -> None:
    h, w = frame.shape[:2]
    for a, b in SKELETON:
        if a in landmarks and b in landmarks and landmarks[a][2] > .35 and landmarks[b][2] > .35:
            p1 = int(landmarks[a][0] * w), int(landmarks[a][1] * h)
            p2 = int(landmarks[b][0] * w), int(landmarks[b][1] * h)
            cv2.line(frame, p1, p2, (65, 235, 180), 3, cv2.LINE_AA)
    for idx in ((12, 14, 16) if handedness == "right" else (11, 13, 15)):
        if idx in landmarks and landmarks[idx][2] > .35:
            cv2.circle(frame, (int(landmarks[idx][0] * w), int(landmarks[idx][1] * h)), 5, (15, 245, 255), -1)
    if label and landmarks:
        anchor = next((landmarks[idx] for idx in (0, 11, 12) if idx in landmarks and landmarks[idx][2] > .35), None)
        if anchor:
            cv2.putText(frame, label, (int(anchor[0] * w), max(20, int(anchor[1] * h) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (65, 235, 180), 2, cv2.LINE_AA)
    if ball:
        center = int(ball.x * w), int(ball.y * h)
        cv2.circle(frame, center, max(7, int(ball.radius * max(w, h))), (30, 130, 255), 3, cv2.LINE_AA)
    if rim:
        x, y, rw, rh = rim
        cv2.rectangle(frame, (int(x*w), int(y*h)), (int((x+rw)*w), int((y+rh)*h)), (255, 180, 30), 2)


def analyze_video(input_path: Path, output_dir: Path, rim: tuple[float, float, float, float] | None,
                  progress: Callable[[float, str], None] | None = None,
                  mode: str = "form", handedness: str = "right", camera: str = "auto",
                  court: list[list[float]] | None = None) -> dict:
    if mode not in {"form", "one_on_one"} or handedness not in {"left", "right"}:
        raise ValueError("Invalid analysis mode or shooting hand")
    report = progress or (lambda _value, _message: None)
    profile = camera_profile(camera, mode)
    court = validate_court(court)
    game_mode = mode == "one_on_one"
    report(0.02, "Loading local vision models")
    pose_path = None if game_mode else ensure_pose_model()
    device = _device()
    game_pose_path = None
    if game_mode:
        game_pose_path = os.environ.get("BALLFORM_GAME_POSE_MODEL") or str(_ensure_model(
            COURT_POSE_MODEL, MODEL_RELEASE + COURT_POSE_MODEL.name))
        game_pose_model = YOLO(game_pose_path)
    configured_ball_model = os.environ.get("BALLFORM_YOLO_MODEL")
    if game_mode and not configured_ball_model:
        ball_model_path = str(_ensure_model(ROOT / "models" / MODEL_FILENAME, MODEL_URL, MODEL_SHA256))
        ball_model = BasketballDetector(ball_model_path)
    else:
        ball_model_path = configured_ball_model or str(_ensure_model(BALL_MODEL, BALL_MODEL_URL))
        ball_model = YOLO(ball_model_path)
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
    flows: dict[int, float] = {}
    previous_ball: Detection | None = None
    previous_gray: np.ndarray | None = None
    processed = 0
    cut_frames = []
    raw_people_counts = []
    pose_tracker = PoseTracker(width / height, max_gap_frames=max(3, round(fps * .7)))
    with ExitStack() as resources:
        resources.callback(capture.release)
        resources.callback(writer.release)
        landmarker = None if game_mode else resources.enter_context(mp.tasks.vision.PoseLandmarker.create_from_options(options))
        frame_no = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_no += 1
            if frame_no % stride:
                continue
            time_s = frame_no / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if game_mode and scene_cut(previous_gray, gray):
                cut_frames.append(frame_no)
                pose_tracker.reset()
                previous_ball = None
            players = []
            landmark_maps = []
            if court_vision:
                players, landmark_maps, ball = court_vision.detect(frame, frame_no, time_s, previous_ball)
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
                ball = _track_ball(ball_model, frame, previous_ball, frame_no, time_s, width, height)
            poses.extend(players)
            player_frames.append({"frame": frame_no, "time_s": time_s, "players": players})
            if ball:
                balls.append(ball)
                previous_ball = ball
            flows[frame_no] = _net_flow(previous_gray, gray, rim)
            previous_gray = gray
            for visible_pose, player in zip(landmark_maps, players):
                player_label = f"P{player.track_id}" if mode == "one_on_one" and player.track_id else None
                _draw(frame, visible_pose, None, None, handedness, player_label)
            _draw(frame, {}, ball, rim)
            if court:
                polygon = np.rint(np.asarray(court) * (width, height)).astype(np.int32)
                cv2.polylines(frame, [polygon], True, (255, 210, 80), 2)
            writer.write(frame)
            processed += 1
            if processed % 5 == 0:
                report(min(.88, .08 + .78 * frame_no / max(1, total)), f"Analyzing frame {frame_no:,} of {total:,}")
    observed_balls = list(balls)
    if game_mode:
        # Keep the measured inputs so scoring can be reviewed without rerunning inference.
        (output_dir / "observations.json").write_text(json.dumps({
            "fps": fps, "width": width, "height": height, "cuts": cut_frames,
            "balls": [asdict(b) for b in observed_balls],
            "players": [{"frame": f["frame"], "time_s": f["time_s"],
                         "players": [asdict(p) for p in f["players"]]} for f in player_frames],
        }, default=lambda value: float(value)))
    nonzero_flow = [v for v in flows.values() if v > 0]
    baseline = float(np.median(nonzero_flow)) if nonzero_flow else 1.0
    normalized_flow = {frame: value / max(baseline, 1e-5) for frame, value in flows.items()}
    # A marked rim is a fixed image coordinate: never reuse it across broadcast pans/cuts.
    scoring_rim = None if game_mode and profile in {"broadcast", "elevated"} else rim
    shots = []
    boundaries = [0, *cut_frames, frame_no + 1]
    for start, end in zip(boundaries, boundaries[1:]):
        segment_balls = _interpolate_track([b for b in balls if start <= b.frame < end], max(2, round(fps * .25)))
        segment_poses = [p for p in poses if start <= p.frame < end]
        segment_shots = analyze_shots(segment_balls, segment_poses, fps, scoring_rim, normalized_flow,
                                     aspect_ratio=width / height, handedness=handedness, game_mode=game_mode)
        for shot in segment_shots:
            if game_mode and profile in {"broadcast", "elevated"}:
                shot.evidence = ["Outcome unavailable: elevated/moving footage has no tracked rim."
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
        "mode": mode, "handedness": handedness, "game_summary": game_summary,
        "camera_profile": profile, "court_polygon": court,
        "vision": {"pose_model": Path(game_pose_path).name if game_pose_path else pose_path.name,
                   "ball_model": Path(ball_model_path).name, "device": device,
                   "ball_device": "cpu" if isinstance(ball_model, BasketballDetector) else device,
                   "input_size": court_vision.imgsz if court_vision else 640,
                   "ball_input_size": 640 if isinstance(ball_model, BasketballDetector) else (court_vision.imgsz if court_vision else 640),
                   "broadcast_role_filter": bool(isinstance(ball_model, BasketballDetector) and profile == "broadcast"),
                   "overlapping_crops": bool(court_vision and court_vision.tiled)},
        "right_handed": handedness == "right", "shots": [shot.to_dict() for shot in shots],
        "diagnostics": {"pose_frames": sum(bool(p["players"]) for p in player_frames),
                        "two_player_frames": sum(len(p["players"]) == 2 for p in player_frames),
                        "multi_player_frames": sum(len(p["players"]) > 2 for p in player_frames),
                        "max_players_visible": max((len(p["players"]) for p in player_frames), default=0),
                        "max_people_detected": max(raw_people_counts, default=0),
                        "scene_cuts": len(cut_frames), "scene_cut_times_s": [round(f / fps, 3) for f in cut_frames],
                        "court_filtered": court is not None,
                        "ball_detections": len(observed_balls),
                        "rim_marked": rim is not None},
        "limitations": [
            "Angles and distances are 2D image-plane estimates, not calibrated 3D measurements.",
            "Make/miss is inferred from visible ball/rim geometry and net motion; occlusion can lower confidence.",
            "Feedback is descriptive and should complement, not replace, coaching judgment.",
            "Shot candidates and outcomes require video review; passes, occlusion and camera movement can cause errors.",
        ],
        "annotated_video": "annotated.mp4",
    }
    if mode == "one_on_one":
        result["limitations"].extend(game_summary.get("limitations", []))
        if court is None:
            result["limitations"].append("No playing-area polygon supplied. Automatic player/referee filtering is used only with the default broadcast detector and can make mistakes; mark the court to further exclude the sidelines.")
        if profile in {"broadcast", "elevated"}:
            result["limitations"].append("Elevated camera profile is not court calibration. Pans and zooms affect projected trajectories and spacing. Make/miss is withheld because a fixed rim box cannot follow a moving camera; use courtside for a stationary clip.")
        result["limitations"].append("Game shots require raised-hand ball contact and an arc rising above the release shoulders. Fully occluded releases, flat arcs and underhand shots may be omitted; passes and slow-motion edits need manual review.")
    (output_dir / "result.json").write_text(json.dumps(result, indent=2))
    report(1.0, "Complete")
    return result
