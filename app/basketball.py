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


class BasketballDetector:
    def __init__(self, path, backend="auto", compute_units="CPUAndGPU"):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        if backend not in {"auto", "cpu", "coreml"}:
            raise ValueError("Basketball detector backend must be 'auto', 'cpu', or 'coreml'.")
        if compute_units not in COREML_COMPUTE_UNITS:
            raise ValueError(f"Core ML compute units must be one of {sorted(COREML_COMPUTE_UNITS)}.")
        self.requested_backend = backend
        self.compute_units = compute_units
        self.fallback_reason = None
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self._model_path = str(path)
        self._session_options = options
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
            self.fallback_reason = f"Core ML initialization failed: {type(exc).__name__}: {exc}"
            logging.warning("%s; using CPU detector", self.fallback_reason)
            self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            chosen = "cpu"
        self.backend = chosen
        self.providers = self.session.get_providers()

    def detect(self, frame):
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
