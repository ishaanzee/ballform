# Third-party notices

Ballform depends on third-party open-source software. Dependency versions are recorded in `uv.lock`; consult each distribution for its complete license text.

- Ultralytics and its YOLO models: AGPL-3.0 or a separately purchased enterprise license.
- MediaPipe: Apache License 2.0.
- ONNX Runtime: MIT License.
- OpenCV: Apache License 2.0, with bundled third-party notices.
- PyTorch and TorchVision: BSD-style licenses.
- FastAPI, Uvicorn, NumPy, QRCode, and their transitive dependencies retain their respective licenses.

The project downloads `pose_landmarker_lite.task` from Google's MediaPipe model storage and `yolo11n.pt` from Ultralytics for individual form analysis. Game analysis downloads YOLO26s-pose or YOLO26m-pose from Ultralytics. Those files are excluded from this repository and remain subject to their upstream terms.

Game ball/player-role detection uses [ortizeg/basketball-rf-detr-m-640](https://huggingface.co/ortizeg/basketball-rf-detr-m-640), published under Apache-2.0, based on [Roboflow RF-DETR](https://github.com/roboflow/rf-detr). Artifact revision: `ffc7d08a5b8ee950681d922b3b1d13ef1fe66870`; SHA-256: `708789b50c42b5265cced64276a8beb1b7f294d324f954d359fd8a2d01f5a939`. The model publisher documents preprocessing and provenance in [object-detection-eval](https://github.com/ortizeg/object-detection-eval). Training data attribution: [ego-playground / basketball-player-detection-3](https://universe.roboflow.com/ego-playground/basketball-player-detection-3-ycjdo-lacpg), CC BY 4.0, as documented by the publisher. Weights download separately; the dataset is not redistributed here. Ballform adds its own tracking, confidence filtering and shot analysis, so upstream model metrics are not Ballform accuracy claims.
