# Ballform

Free, open-source basketball video analysis that runs on the user's own computer. Choose **Shooting form** for left- or right-arm mechanics, or **1-on-1 game** for projected shooter–defender separation and contest review. Ballform uses MediaPipe Pose, YOLO basketball detection, and rim/net evidence for estimated make/miss classification. Uploaded footage and model inference stay local.

## Run locally

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). After cloning the repository, run:

```bash
uv sync --extra dev --python 3.12
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. No Node.js, frontend build, cloud account, environment file, or access token is required for local use. On first analysis, the MediaPipe pose model and YOLO weights download into `models/`; later runs can work offline. Install `ffmpeg` for browser-friendly H.264 review videos.

Uploaded clips and generated reports are stored under `data/jobs/`. Stop the server with `Ctrl+C`.

## Upload from an iPhone with Tailscale

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

If required-Tailscale mode reports that Tailscale is stopped, open the app on both devices and enable it before retrying.

## Open-source license

Ballform is licensed under **AGPL-3.0-only** because it integrates the AGPL-licensed Ultralytics package and model. MediaPipe is Apache-2.0. Downloaded model weights, uploaded footage, generated reports, caches, and pairing credentials are excluded from git. See `LICENSE`, `NOTICE.md`, `CONTRIBUTING.md`, and `SECURITY.md` before distributing a modified service.

## Capture guidance

- Record at 60 fps if possible; 1080p is enough.
- Keep the shooter, nearby defenders, ball flight, rim, and net in frame from gather through result. In 5-on-5, a half-court view usually preserves more player detail than a distant full-court view.
- Put the phone 3–6 ft high and avoid digital zoom or moving the camera.
- Side and rear-oblique views work. A side view is best for release/arc; rear-oblique is better for elbow alignment.
- Drag a tight box around the rim in the preview before analysis.

## What the numbers mean

All joint and launch angles are **2D image-plane estimates**. They are useful for comparing attempts recorded from the same camera position, not as calibrated 3D biomechanics. The outcome classifier reports its confidence and the exact evidence it used.

### 1-on-1 game review

Select **1-on-1 game** before uploading. The mode accepts isolated 1-on-1 clips or crowded 5-on-5 footage. It detects up to ten visible poses, associates the shooter by ball-to-wrist proximity, groups jersey appearance into two teams, and chooses the nearest visible opponent as the primary defender. Persistent player IDs carry that matchup into the pre-release separation measurement. Mark the rim for outcome estimates. Each detected shot offers a release review button, underlying measurements, evidence quality, and an exportable JSON report.

- **Separation:** projected distance between player hip centers divided by the shooter's shoulder-to-hip torso length.
- **Contest clearance:** projected distance from the ball to the nearest visible defender wrist, in the same torso units. This is a hand-contest proxy, not a measurement of a blocked shooting lane.
- **Separation change:** change over approximately 0.5 seconds before release, when conservative player matching and stable body scale permit comparison. Positive values mean more projected space.
- **Shot-space score (0–100):** 65% separation component plus 35% contest-clearance component. Separation maps 0.5–3 torso lengths to 0–100; clearance maps 0.15–1.5 to 0–100, both clipped at the endpoints. Separation change is descriptive and does not affect the score.

The formula and thresholds are **unvalidated review heuristics**, not make probability, expected points, a professional player grade, or a claim of optimal shot selection. Evidence quality is also heuristic, not a statistical probability. No score is produced when shooter ownership is ambiguous, teams cannot be separated by jersey appearance, a nearby unidentified player could be the defender, required landmarks are occluded, or ball evidence is weak. Missing measurements remain unavailable, never zero.

Footage is sampled at up to 30 FPS; exact release timing still has sampling and detection uncertainty. Distances are aspect-corrected but not court-calibrated: camera angle, depth, player overlap and movement affect the numbers. Compare attempts from a consistent camera setup and review detected shots and outcomes in the video. Passes can still be mistaken for shots. Game mode does not publish arm mechanics across players without reliable persistent identity. Use the separate form mode for individual biomechanics review.

Game mode uses MediaPipe's full pose model and its documented [`num_poses` option](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarkerOptions) to request up to ten players. Track IDs are motion- and jersey-assisted; team grouping uses torso colour in the current lighting. Similar uniforms, severe overlap, a tiny player image, cuts, or moving cameras can still break identity. The analyzer reports its selection evidence and withholds uncertain matchups. Professional-use accuracy still needs evaluation on labeled representative game footage.

## Architecture

- `app/analyzer.py`: video decoding, pose/ball inference, net optical flow, annotation
- `app/scoring.py`: shot segmentation, release detection, metrics, outcome evidence
- `app/game.py`: conservative 1-on-1 association, projected measurements, transparent shot-space score
- `app/tracking.py`: multi-player track IDs and jersey-colour descriptors
- `app/main.py`: upload/job API
- `app/lan.py`: tokenized LAN/Tailscale sharing and QR pairing
- `web/`: dependency-free local upload and review interface

Set `BALLFORM_YOLO_MODEL` to another Ultralytics detection checkpoint if desired. It must include COCO class 32 (`sports ball`). The default checkpoint is stored in `models/` and ignored by git.

Run regression checks with `uv run pytest -q` and `node --check web/app.js`.
