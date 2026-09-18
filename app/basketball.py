"""Basketball-trained RF-DETR inference (pinned ONNX artifact; no hosted API)."""
from __future__ import annotations

import cv2
import numpy as np

MODEL_FILENAME = "basketball-rfdetr.onnx"
MODEL_URL = "https://huggingface.co/ortizeg/basketball-rf-detr-m-640/resolve/ffc7d08a5b8ee950681d922b3b1d13ef1fe66870/rfdetr_m_640.onnx"
MODEL_SHA256 = "708789b50c42b5265cced64276a8beb1b7f294d324f954d359fd8a2d01f5a939"
# Export retains Roboflow COCO category IDs, with unused supercategory zero.
BALL_CLASSES = {1, 2}
PLAYER_CLASSES = {4, 5, 6, 7, 8}
REFEREE_CLASS = 9


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
    def __init__(self, path):
        import onnxruntime as ort
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        # ONNX CPU is portable and tested; YOLO pose independently uses Apple MPS.
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def detect(self, frame):
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: preprocess(frame)})
        boxes = next(value for value in outputs if value.shape[-1] == 4)
        logits = next(value for value in outputs if value.shape[-1] == 11)
        return decode(boxes, logits)
