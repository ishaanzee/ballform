"""Basketball-trained RF-DETR inference (pinned ONNX artifact; no hosted API)."""
from __future__ import annotations

import hashlib
import logging
import platform
from pathlib import Path

import cv2
import numpy as np

MODEL_FILENAME = "basketball-rfdetr.onnx"
MODEL_URL = "https://huggingface.co/ortizeg/basketball-rf-detr-m-640/resolve/ffc7d08a5b8ee950681d922b3b1d13ef1fe66870/rfdetr_m_640.onnx"
MODEL_SHA256 = "708789b50c42b5265cced64276a8beb1b7f294d324f954d359fd8a2d01f5a939"
# Export retains Roboflow COCO category IDs, with unused supercategory zero.
BALL_CLASSES = {1, 2}
PLAYER_CLASSES = {4, 5, 6, 7, 8}
# Dataset order: player, player-in-possession, jump-shot, layup-dunk, shot-block.
POSSESSION_CLASS = 5
RIM_CLASS = 10
# Shot-context classes, recorded per frame as evidence for shot candidates and types.
SHOT_EVENT_CLASSES = {2: "ball_in_basket", 6: "jump_shot", 7: "layup_dunk", 8: "shot_block"}
REFEREE_CLASS = 9
# The transformer runs ~3x faster on the GPU than on the Neural Engine
# (M3 Pro: 46 ms CPUAndGPU vs 119 ms CPUAndNeuralEngine). ALL is similar per
# call but Core ML re-specializes it for longer on every session load.
COREML_COMPUTE_UNITS = {"ALL", "CPUAndGPU", "CPUAndNeuralEngine", "CPUOnly"}


def preprocess(frame):
    rgb = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mean = np.asarray([.485, .456, .406], dtype=np.float32)
    std = np.asarray([.229, .224, .225], dtype=np.float32)
    return np.ascontiguousarray(((rgb - mean) / std).transpose(2, 0, 1)[None])


def decode(boxes, logits, threshold=.25):
    """Native RF-DETR sigmoid/top-300 query-class decode, normalized xyxy."""
    boxes, logits = np.asarray(boxes)[0], np.asarray(logits)[0]
    if boxes.shape != (300, 4) or logits.shape != (300, 11):
        raise ValueError("Unexpected basketball RF-DETR output schema")
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    scores = probabilities.ravel()
    selected = np.argsort(scores)[-300:][::-1]
    objects = []
    for flat_index in selected:
        score = float(scores[flat_index])
        if score < threshold:
            break
        query, category = divmod(int(flat_index), 11)
        if category not in BALL_CLASSES | PLAYER_CLASSES | {REFEREE_CLASS, 10}:
            continue
        cx, cy, w, h = boxes[query]
        xyxy = tuple(float(v) for v in np.clip([cx-w/2, cy-h/2, cx+w/2, cy+h/2], 0, 1))
        objects.append((category, score, xyxy))
    return objects


def _load_fast_detector(path):
    """The MLX + Metal RF-DETR port (optional package `fast_rfdetr`); raises ImportError if absent."""
    from fast_rfdetr import FastBasketballDetector
    # Batch 2 is the side-crop pair; tracing it now keeps the first real call warm.
    return FastBasketballDetector(str(path), warmup_batch_sizes=(1, 2))


class BasketballDetector:
    """RF-DETR basketball detector.

    Backends: "mlx" runs the fast_rfdetr MLX/Metal port on the GPU (~1.7x faster
    per call than Core ML, same outputs to 1e-3); "coreml" and "cpu" run the
    ONNX file through ONNX Runtime. "auto" prefers mlx on Apple silicon, then
    Core ML, then CPU, recording why in fallback_reason.
    """

    def __init__(self, path, backend="auto", compute_units="CPUAndGPU"):
        if backend not in {"auto", "mlx", "cpu", "coreml"}:
            raise ValueError("Basketball detector backend must be 'auto', 'mlx', 'cpu', or 'coreml'.")
        if compute_units not in COREML_COMPUTE_UNITS:
            raise ValueError(f"Core ML compute units must be one of {sorted(COREML_COMPUTE_UNITS)}.")
        self.requested_backend = backend
        self.compute_units = compute_units
        self.fallback_reason = None
        self._model_path = str(path)
        self._fast = None
        self.session = None
        apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        if backend == "mlx" or (backend == "auto" and apple_silicon):
            try:
                self._fast = _load_fast_detector(path)
                self.backend = "mlx"
                self.providers = ["MLX (Metal)"]
                return
            except Exception as exc:
                if backend == "mlx":
                    raise
                self.fallback_reason = f"MLX detector unavailable: {type(exc).__name__}: {exc}"
                logging.warning("%s; using ONNX Runtime", self.fallback_reason)
        self._init_onnx(backend)

    def _init_onnx(self, backend):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self._session_options = options
        path, compute_units = self._model_path, self.compute_units
        coreml_available = "CoreMLExecutionProvider" in ort.get_available_providers()
        chosen = ("coreml" if coreml_available and platform.system() == "Darwin"
                  and platform.machine() == "arm64" else "cpu") if backend == "auto" else backend
        if chosen == "coreml" and not coreml_available:
            raise RuntimeError("This ONNX Runtime installation does not include the Core ML provider.")
        try:
            providers = ["CPUExecutionProvider"]
            if chosen == "coreml":
                # A content-specific cache avoids recompilation on every job
                # and stale graphs if the ONNX file changes at the same path.
                model_path = Path(path)
                with model_path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()[:20]
                cache_dir = model_path.parent / "coreml-cache" / digest / compute_units
                cache_dir.mkdir(parents=True, exist_ok=True)
                providers.insert(0, ("CoreMLExecutionProvider", {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": compute_units,
                    "RequireStaticInputShapes": "1",
                    "ModelCacheDirectory": str(cache_dir),
                }))
            self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            if chosen == "coreml" and "CoreMLExecutionProvider" not in self.session.get_providers():
                raise RuntimeError("Core ML provider did not initialize for this model.")
        except Exception as exc:
            if backend != "auto" or chosen != "coreml":
                raise
            reason = f"Core ML initialization failed: {type(exc).__name__}: {exc}"
            self.fallback_reason = f"{self.fallback_reason}; {reason}" if self.fallback_reason else reason
            logging.warning("%s; using CPU detector", reason)
            self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            chosen = "cpu"
        self.backend = chosen
        self.providers = self.session.get_providers()

    def _fast_failed(self, exc):
        """Leave the MLX path for ONNX Runtime after a runtime failure, in auto mode only."""
        if self.requested_backend != "auto":
            raise exc
        self.fallback_reason = f"MLX inference failed: {type(exc).__name__}: {exc}"
        logging.warning("%s; using ONNX Runtime", self.fallback_reason)
        self._fast = None
        self._init_onnx("auto")

    def detect(self, frame):
        if self._fast is not None:
            try:
                return decode(*self._fast.raw(frame))
            except Exception as exc:
                self._fast_failed(exc)
        feed = {self.session.get_inputs()[0].name: preprocess(frame)}
        try:
            outputs = self.session.run(None, feed)
        except Exception as exc:
            if self.requested_backend != "auto" or self.backend != "coreml":
                raise
            import onnxruntime as ort
            self.fallback_reason = f"Core ML inference failed: {type(exc).__name__}: {exc}"
            logging.warning("%s; using CPU detector", self.fallback_reason)
            self.session = ort.InferenceSession(self._model_path, sess_options=self._session_options,
                                                providers=["CPUExecutionProvider"])
            self.backend = "cpu"
            self.providers = self.session.get_providers()
            outputs = self.session.run(None, feed)
        boxes = next(value for value in outputs if value.shape[-1] == 4)
        logits = next(value for value in outputs if value.shape[-1] == 11)
        return decode(boxes, logits)

    def detect_batch(self, frames):
        """Detect several images (e.g. the two side crops) in one GPU call when the backend allows it."""
        if self._fast is not None and 1 <= len(frames) <= 3:
            try:
                boxes, logits = self._fast.raw_batch(list(frames))
                return [decode(boxes[i:i + 1], logits[i:i + 1]) for i in range(len(frames))]
            except Exception as exc:
                self._fast_failed(exc)
        return [self.detect(frame) for frame in frames]
