"""Export game pose weights as fixed-shape fp16 ONNX for Core ML's Neural Engine.

Usage (export-only packages stay out of the app's dependencies):
  YOLO_AUTOINSTALL=False uv run --with onnx --with onnxslim --with onnxconverter-common \\
      python scripts/export_pose_coreml.py [yolo26s-pose yolo26m-pose]

The Neural Engine runs only fp16 programs; an fp32 export silently runs on the
CPU instead. The app uses these files automatically when they exist.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxconverter_common import float16
from ultralytics import YOLO

from app.pose import EXPORT_SHAPES, export_path

MODELS = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=["yolo26s-pose"])
    args = parser.parse_args()
    for name in args.models:
        for size, shape in EXPORT_SHAPES.items():
            exported = Path(YOLO(MODELS / f"{name}.pt").export(
                format="onnx", imgsz=shape, dynamic=False, simplify=True, verbose=False))
            model = onnx.load(exported)
            # Stale fp32 value_info breaks the converter's Resize casts.
            del model.graph.value_info[:]
            destination = export_path(MODELS, name, size)
            onnx.save(float16.convert_float_to_float16(model, keep_io_types=True), destination)
            exported.unlink()
            print(f"{destination.name}: input {shape}")


if __name__ == "__main__":
    main()
