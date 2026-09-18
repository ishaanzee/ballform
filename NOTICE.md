# Third-party notices

Ballform depends on third-party open-source software. Dependency versions are recorded in `uv.lock` and `frontend/package-lock.json`; consult each distribution for its complete license text.

- Ultralytics and its YOLO models: AGPL-3.0 or a separately purchased enterprise license.
- MediaPipe: Apache License 2.0.
- OpenCV: Apache License 2.0, with bundled third-party notices.
- PyTorch and TorchVision: BSD-style licenses.
- FastAPI, Uvicorn, NumPy, QRCode, and their transitive dependencies retain their respective licenses.
- Next.js, React, React DOM, and tsx: MIT License.
- TypeScript: Apache License 2.0.

The project downloads `pose_landmarker_lite.task` from Google's MediaPipe model storage and `yolo11n.pt` from Ultralytics on first use. Those files are excluded from this repository and remain subject to their upstream terms.
