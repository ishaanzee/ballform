"""Benchmark the two existing side ball crops on recorded trigger frames.

Usage:
  .venv/bin/python scripts/benchmark_side_crop_parallel.py \
      diagnostic/5on5clip3.mp4 /path/to/pipeline_timing.json

This is an isolated experiment: it does not change production inference.
The same pixels and ONNX model are used for serial and two-session parallel runs.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from app.basketball import BasketballDetector, MODEL_FILENAME
from app.vision import regions


def _same_objects(left, right, tolerance: float = 1e-3) -> bool:
    if len(left) != len(right):
        return False
    return all(a[0] == b[0] and abs(a[1] - b[1]) <= tolerance
               and all(abs(x - y) <= tolerance for x, y in zip(a[2], b[2]))
               for a, b in zip(left, right))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().parents[1] / "models" / MODEL_FILENAME)
    args = parser.parse_args()
    triggered = {item["frame"] for item in json.loads(args.trace.read_text())["frames"]
                 if item["side_ball_crops_triggered"]}
    if not triggered:
        raise SystemExit("Trace has no side-crop trigger frames.")

    first = BasketballDetector(args.model)
    second = BasketballDetector(args.model, backend=first.backend)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot decode {args.video}")
    serial_s = parallel_s = shared_s = 0.0
    compared = mismatched = shared_mismatched = 0
    warmed = False
    with ThreadPoolExecutor(max_workers=2) as workers:
        frame_no = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_no += 1
            if frame_no not in triggered:
                continue
            height, width = frame.shape[:2]
            crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in
                     list(regions(width, height, True))[1:]]
            if len(crops) != 2:
                raise RuntimeError("Expected two side crops")
            if not warmed:
                first.detect(crops[0])
                second.detect(crops[1])
                warmed = True

            def serial():
                start = time.perf_counter()
                results = [first.detect(crop) for crop in crops]
                return results, time.perf_counter() - start

            def parallel():
                start = time.perf_counter()
                left = workers.submit(first.detect, crops[0])
                right = workers.submit(second.detect, crops[1])
                results = [left.result(), right.result()]
                return results, time.perf_counter() - start

            def shared_parallel():
                start = time.perf_counter()
                left = workers.submit(first.detect, crops[0])
                right = workers.submit(first.detect, crops[1])
                results = [left.result(), right.result()]
                return results, time.perf_counter() - start

            variants = [serial, parallel, shared_parallel]
            # Rotate order to reduce bias from warming or thermal drift.
            variants = variants[compared % 3:] + variants[:compared % 3]
            outcomes = {variant.__name__: variant() for variant in variants}
            serial_result, s_s = outcomes["serial"]
            parallel_result, p_s = outcomes["parallel"]
            shared_result, shared_frame_s = outcomes["shared_parallel"]
            serial_s += s_s
            parallel_s += p_s
            shared_s += shared_frame_s
            mismatched += not all(_same_objects(a, b) for a, b in
                                  zip(serial_result, parallel_result))
            shared_mismatched += not all(_same_objects(a, b) for a, b in
                                         zip(serial_result, shared_result))
            compared += 1
    capture.release()
    print(json.dumps({"frames": compared, "backend": first.backend,
                      "serial_seconds": round(serial_s, 3),
                      "parallel_seconds": round(parallel_s, 3),
                      "parallel_speedup": round(serial_s / parallel_s, 3),
                      "detection_mismatch_frames": mismatched,
                      "shared_session_parallel_seconds": round(shared_s, 3),
                      "shared_session_speedup": round(serial_s / shared_s, 3),
                      "shared_session_mismatch_frames": shared_mismatched}, indent=2))


if __name__ == "__main__":
    main()
