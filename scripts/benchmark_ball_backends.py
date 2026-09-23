"""Compare CPU and Core ML basketball detections on the same video frames/crops.

Usage: uv run python scripts/benchmark_ball_backends.py /path/to/clip.mp4
This is a diagnostic; it does not change the app's default backend.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from app.basketball import BALL_CLASSES, BasketballDetector
from app.vision import iou, regions


def compare(reference, candidate):
    """Greedily match same-class boxes; count unmatched detections and box drift."""
    used = set()
    matched = 0
    minimum_iou = 1.0
    maximum_confidence_delta = 0.0
    for category, confidence, box in sorted(reference, key=lambda item: item[1], reverse=True):
        options = [(iou(box, other_box), index, other_confidence)
                   for index, (other_category, other_confidence, other_box) in enumerate(candidate)
                   if index not in used and category == other_category]
        overlap, index, other_confidence = max(options, default=(0.0, -1, 0.0))
        if overlap < .5:
            continue
        used.add(index)
        matched += 1
        minimum_iou = min(minimum_iou, overlap)
        maximum_confidence_delta = max(maximum_confidence_delta, abs(confidence - other_confidence))
    return matched, len(reference) - matched, len(candidate) - matched, minimum_iou, maximum_confidence_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "models/basketball-rfdetr.onnx")
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    cpu = BasketballDetector(args.model, backend="cpu")
    started = time.perf_counter()
    coreml = BasketballDetector(args.model, backend="coreml")
    load_seconds = time.perf_counter() - started
    counts = Counter()
    cpu_seconds = coreml_seconds = 0.0
    worst_iou = 1.0
    largest_confidence_delta = 0.0
    examples = []
    frame_no = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            for region_no, (x1, y1, x2, y2) in enumerate(regions(width, height, True)):
                crop = frame[y1:y2, x1:x2]
                started = time.perf_counter()
                reference = cpu.detect(crop)
                cpu_seconds += time.perf_counter() - started
                started = time.perf_counter()
                candidate = coreml.detect(crop)
                coreml_seconds += time.perf_counter() - started
                counts["comparisons"] += 1
                if region_no == 0:
                    cpu_side = not any(cls in BALL_CLASSES and conf >= .65 for cls, conf, _ in reference)
                    coreml_side = not any(cls in BALL_CLASSES and conf >= .65 for cls, conf, _ in candidate)
                    counts["side_crop_decision_disagreements"] += cpu_side != coreml_side
                matched, missed, added, minimum_iou, confidence_delta = compare(reference, candidate)
                counts["reference_detections"] += len(reference)
                counts["coreml_detections"] += len(candidate)
                counts["matched"] += matched
                counts["missed"] += missed
                counts["added"] += added
                worst_iou = min(worst_iou, minimum_iou)
                largest_confidence_delta = max(largest_confidence_delta, confidence_delta)
                if (missed or added) and len(examples) < 20:
                    examples.append({"frame": frame_no, "region": region_no, "cpu_count": len(reference),
                                     "coreml_count": len(candidate), "missed": missed, "added": added})
            frame_no += 1
            if frame_no % 30 == 0:
                print(f"Compared {frame_no} frames", flush=True)
    finally:
        capture.release()
    print(json.dumps({"frames": frame_no, "model_load_seconds": round(load_seconds, 2),
                      "cpu_detector_seconds": round(cpu_seconds, 2),
                      "coreml_detector_seconds": round(coreml_seconds, 2),
                      "detector_speedup": round(cpu_seconds / coreml_seconds, 2),
                      "worst_matched_iou": round(worst_iou, 4),
                      "largest_matched_confidence_delta": round(largest_confidence_delta, 4),
                      "counts": counts, "disagreement_examples": examples}, indent=2))


if __name__ == "__main__":
    main()
