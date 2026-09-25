"""Game pose backends; both return crop-pixel boxes, confidences and COCO-17 keypoints."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Fixed input shape (h, w) per requested size, matching Ultralytics' rect letterbox
# for 16:9 full frames at 1280 and the 60%-width side crops at 960.
EXPORT_SHAPES = {1280: (736, 1280), 960: (928, 960)}
POSE_CONF, POSE_IOU, POSE_MAX_DET = .3, .65, 60


def export_path(models_dir: Path, pose_model: str, size: int) -> Path:
    h, w = EXPORT_SHAPES[size]
    return models_dir / f"{pose_model}-{h}x{w}-fp16.onnx"


def letterbox_shape(height: int, width: int, size: int, stride: int = 32) -> tuple[int, int]:
    """The padded input shape Ultralytics' rect letterbox produces for this image."""
    r = min(size / height, size / width)
    new_h, new_w = round(height * r), round(width * r)
    return new_h + (size - new_h) % stride, new_w + (size - new_w) % stride


def letterbox(image: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and centre-pad exactly as Ultralytics' LetterBox does for prediction."""
    height, width = image.shape[:2]
    r = min(shape[0] / height, shape[1] / width)
    new_w, new_h = round(width * r), round(height * r)
    dw, dh = (shape[1] - new_w) / 2, (shape[0] - new_h) / 2
    if (width, height) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - .1), round(dh + .1)
    left, right = round(dw - .1), round(dw + .1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, r, (left, top)


def decode(output: np.ndarray, gain: float, pad: tuple[int, int], height: int, width: int):
    """Ultralytics single-class NMS and coordinate scaling for a raw (1, 56, N) pose head."""
    import torch
    import torchvision

    prediction = output[0].T
    prediction = prediction[prediction[:, 4] > POSE_CONF]
    if not len(prediction):
        return None
    cx, cy, bw, bh = prediction[:, 0], prediction[:, 1], prediction[:, 2], prediction[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    scores = prediction[:, 4]
    keep = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), POSE_IOU)[:POSE_MAX_DET].numpy()
    boxes, scores = boxes[keep], scores[keep]
    keypoints = prediction[keep, 5:].reshape(-1, 17, 3).copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    keypoints[..., 0] = ((keypoints[..., 0] - pad[0]) / gain).clip(0, width)
    keypoints[..., 1] = ((keypoints[..., 1] - pad[1]) / gain).clip(0, height)
    return boxes, scores, keypoints


class TorchPose:
    """Ultralytics YOLO pose on the PyTorch device (MPS on Apple silicon)."""

    backend = "torch"

    def __init__(self, model, device: str):
        self.model, self.device = model, device

    def infer(self, image: np.ndarray, size: int, side: bool = False):
        result = self.model.predict(image, imgsz=size, conf=POSE_CONF, iou=POSE_IOU,
                                    max_det=POSE_MAX_DET, device=self.device, verbose=False)[0]
        if result.keypoints is None or result.boxes is None:
            return None
        # Reading tensors back also synchronizes MPS, so callers time the
        # completed GPU work rather than only asynchronous submission.
        return (result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                result.keypoints.data.cpu().numpy())


def _session(path: Path, size: int, compute_units: str):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    cache_dir = path.parent / "coreml-cache" / path.stem / compute_units
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(path), sess_options=options, providers=[
        ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": compute_units,
                                     "RequireStaticInputShapes": "1",
                                     "ModelCacheDirectory": str(cache_dir)}),
        "CPUExecutionProvider"])
    if "CoreMLExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"Core ML provider did not initialize for {path.name}.")
    return session, session.get_inputs()[0].name, EXPORT_SHAPES[size]


class CoreMLPose:
    """The same YOLO pose weights as fixed-shape fp16 ONNX, run by Core ML.

    The Neural Engine only executes fp16 programs; an fp32 export silently runs
    on the CPU. Images whose letterbox shape differs from an export (non-16:9
    footage) use the PyTorch fallback so every input matches the original path.
    """

    backend = "coreml"

    def __init__(self, paths: dict[int, Path], fallback: TorchPose, compute_units: str = "CPUAndNeuralEngine",
                 side_units: str | None = None):
        self.fallback = fallback
        self.compute_units = compute_units
        self.fallback_calls = 0
        self.sessions = {size: _session(path, size, compute_units) for size, path in paths.items()}
        # Side crops can run on a second compute unit while the full frame is on
        # this one; they only use the 960 export.
        self.side_units = side_units if side_units and side_units != compute_units and 960 in paths else None
        self.side_sessions = {960: _session(paths[960], 960, self.side_units)} if self.side_units else {}

    @property
    def parallel_sides(self) -> bool:
        return bool(self.side_sessions)

    def exported(self, image: np.ndarray, size: int) -> bool:
        """Whether this image runs on Core ML rather than the PyTorch fallback."""
        entry = self.sessions.get(size)
        return entry is not None and letterbox_shape(*image.shape[:2], size) == entry[2]

    def infer(self, image: np.ndarray, size: int, side: bool = False):
        height, width = image.shape[:2]
        entry = (self.side_sessions if side and self.side_sessions else self.sessions).get(size)
        if entry is None or letterbox_shape(height, width, size) != entry[2]:
            self.fallback_calls += 1
            return self.fallback.infer(image, size)
        session, input_name, shape = entry
        padded, gain, pad = letterbox(image, shape)
        blob = cv2.dnn.blobFromImage(padded, 1 / 255., swapRB=True)
        return decode(session.run(None, {input_name: blob})[0], gain, pad, height, width)
