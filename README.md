# Ballform

Local-first basketball shot analysis for uploaded iPhone video. Ballform uses MediaPipe Pose for right-arm mechanics, a small YOLO model for basketball tracking, and rim-plane/net-motion evidence for make/miss classification.

## Run on Apple Silicon

Python 3.11 or 3.12 is required (MediaPipe does not support the system's Python 3.14 build).

```bash
uv sync --extra dev --python 3.12
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. On the first analysis, the MediaPipe pose model and YOLO weights download into local caches. Later runs are offline. `ffmpeg` is recommended and is used automatically to produce a browser-friendly H.264 review video.

## Upload directly from an iPhone with Tailscale

Tailscale is the recommended transport. It works across isolated school, guest, and home networks without using a phone hotspot, while inference and stored footage remain on the Mac.

1. Install [Tailscale for macOS](https://tailscale.com/download/mac) and [Tailscale for iOS](https://tailscale.com/download/ios).
2. Sign into the same Tailscale account on both devices and switch both connections on.
3. Start Ballform in required-Tailscale mode:

```bash
uv run ballform-share --network tailscale
```

The command verifies that Tailscale is connected, discovers the Mac's private Tailscale address, and prints a QR code containing a temporary Ballform access token. Scan it with the iPhone Camera app, open the page in Safari, then choose an existing clip from Photos or tap **Record a new clip**. Upload progress and analysis status stay visible on the phone; if Safari reloads, it reconnects to the current job.

Keep Tailscale connected and the terminal open during the transfer. Press `Ctrl+C` to shut down phone access and invalidate the pairing link. If macOS asks whether Python may accept incoming connections, choose **Allow**. This mode:

- listens only while the command is running;
- requires the random token embedded in the QR link for every job/video API request;
- makes no cloud or analytics requests;
- carries phone-to-Mac traffic inside Tailscale's encrypted connection;
- serializes analyses so concurrent phone uploads do not compete for the GPU.

Running `uv run ballform-share` without a `--network` option automatically prefers Tailscale when it is connected and otherwise falls back to local Wi-Fi. Force ordinary local networking with `--network lan`; the legacy `ballform-lan` command remains available.

If required-Tailscale mode reports that Tailscale is stopped, open the app on both devices and enable it before retrying. A campus policy that explicitly blocks VPN software may still prevent this approach; that requires the future cloud-upload worker described in the roadmap.

## Open-source license

Ballform is licensed under **AGPL-3.0-only** because it integrates the AGPL-licensed Ultralytics package and model. MediaPipe is Apache-2.0. Downloaded model weights, uploaded footage, generated reports, caches, and pairing credentials are excluded from git. See `LICENSE`, `NOTICE.md`, `CONTRIBUTING.md`, and `SECURITY.md` before distributing a modified service.

## Capture guidance

- Record at 60 fps if possible; 1080p is enough.
- Keep the shooter, ball flight, rim, and net in frame from gather through result.
- Put the phone 3–6 ft high and avoid digital zoom or moving the camera.
- Side and rear-oblique views work. A side view is best for release/arc; rear-oblique is better for elbow alignment.
- Drag a tight box around the rim in the preview before analysis.

## What the numbers mean

All joint and launch angles are **2D image-plane estimates**. They are useful for comparing attempts recorded from the same camera position, not as calibrated 3D biomechanics. The outcome classifier reports its confidence and the exact evidence it used.

## Architecture

- `app/analyzer.py`: video decoding, pose/ball inference, net optical flow, annotation
- `app/scoring.py`: shot segmentation, release detection, metrics, outcome evidence
- `app/main.py`: upload/job API
- `app/lan.py`: tokenized LAN/Tailscale sharing and QR pairing
- `web/`: dependency-free upload and review interface

Set `BALLFORM_YOLO_MODEL` to another Ultralytics detection checkpoint if desired. It must include COCO class 32 (`sports ball`). The default checkpoint is stored in `models/` and ignored by git.
