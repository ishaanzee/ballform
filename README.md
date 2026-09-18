# Ballform

Free, open-source basketball video analysis that runs on the user's own computer. Choose **Single-person shooting form** for left- or right-arm mechanics, or **Game / 1-on-1 / 5-on-5** for projected shooter–defender separation and contest review. Individual form uses MediaPipe; game footage uses multi-person YOLO pose and a basketball-trained RF-DETR detector. Uploaded footage and model inference stay local.

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

## NBA broadcasts and elevated pickup footage

1. Select **Game / 1-on-1 / 5-on-5**, then **NBA / elevated broadcast** (or **Pickup / elevated wide view**). Auto uses the wide broadcast pipeline for game mode and the close-up pipeline for form mode; it is a preset, not automatic camera calibration.
2. Prefer a short, continuous half-court possession at the original resolution. Avoid clips with replays, intro graphics and multiple camera angles when possible.
3. Optionally scrub to the game action, choose **Mark playing area**, and click 3–8 corners around the visible court in order. The polygon must be convex. Leave spectators and benches outside. This filters by players' feet, so their heads can extend above the selected area. Use **Clear current selection** to redraw. NBA broadcast mode also filters players/referees automatically; a polygon is especially useful for pickup footage or if sideline people slip through.
4. This polygon stays fixed in image coordinates: choose a region covering play throughout the clip. Split clips with substantial pans or changing camera views and mark each separately. It is an inclusion mask, not a measurement in feet/meters.
5. Click **Analyze game**. Review the annotated player IDs, each shot's evidence and the exported JSON. Both hands are checked automatically in game mode.

Wide game mode uses **YOLO11m-pose** for multi-person keypoints at a 1280-pixel input size plus two overlapping 960-pixel crop passes. Duplicate people are merged back into original-frame coordinates. Small players are checked by torso pixel size rather than requiring their torso to fill 4% of the entire image. **Stationary courtside** uses the game pose model at 960 pixels without extra crops. Pose inference follows the [Ultralytics model documentation](https://docs.ultralytics.com/models/yolo11/) and [prediction API](https://docs.ultralytics.com/modes/predict/).

Ball detection uses a [basketball-trained RF-DETR Medium checkpoint](https://huggingface.co/ortizeg/basketball-rf-detr-m-640), with the publisher's 640-pixel RGB/ImageNet preprocessing and sigmoid/top-300 decoding. When full-frame ball evidence is weak, wide mode checks overlapping crops. Broadcast mode also matches poses to this detector's player/referee classes to exclude officials and spectators automatically. Pickup/elevated and courtside modes do not apply that NBA-trained role filter; use a playing-area polygon there. The checkpoint is pinned to a revision and SHA-256 checked on download. Its training data covers a small number of NBA games; generalization to new arenas and pickup footage is not guaranteed.

The two game checkpoints total approximately **170 MB on disk**. Inference uses more memory than that; sequential, single-frame/crop inference bounds the working set. Pose inference selects Apple GPU (MPS) when available, otherwise CPU; RF-DETR uses the portable ONNX CPU runtime. These models are intended for local M3 Pro 36 GB use; wide-view analysis is offline processing, not a promise of real-time playback. Run `uv sync --extra dev` after updating to install the free ONNX runtime. No paid service is needed. First game analysis downloads weights; subsequent analysis works offline.

Jersey colour uses dominant fabric colour to reduce interference from numbers. In crowded frames, a third appearance group can remain unassigned. Team assignment remains a colour heuristic even when the detector filters referees. Obscured jerseys, similar pickup shirts and spectators wearing team colours can still prevent a reliable matchup. The court polygon can further restrict the region if the automatic broadcast filtering includes someone from the sideline.

Abrupt camera cuts reset player and ball tracking and split shot segmentation so arcs are not joined across edits. Broadcast/elevated mode withholds make/miss because a manually marked rim is fixed in the image and cannot follow a pan or zoom. A fixed-camera clip can use **Stationary courtside** and a marked rim for outcome estimates. Shot-space scores remain projected, uncalibrated measurements at every camera angle.

Game release detection checks both hands and requires ball contact at or above shoulder height followed by an arc rising at least 0.75 torso lengths above the release shoulders, reducing false shots from dribbles. Pre-release hand–ball observations support possession identity, so a background hand crossing the airborne ball does not automatically become the shooter. Fully occluded releases, flat arcs and underhand attempts can be omitted. Passes and slow-motion edits still need review; recorded clip time is not necessarily game time.

## What the numbers mean

All joint and launch angles are **2D image-plane estimates**. They are useful for comparing attempts recorded from the same camera position, not as calibrated 3D biomechanics. The outcome classifier reports its confidence and the exact evidence it used.

### 1-on-1 game review

Select **Game / 1-on-1 / 5-on-5** before uploading. The mode accepts isolated 1-on-1 clips or crowded 5-on-5 footage. It detects multiple people, filters them through an optional playing-area polygon, associates the shooter using raised-hand contact and recent possession, and groups jersey appearance into two teams. Defender selection prioritizes a nearby opposing raised-hand contest, then falls back to the nearest projected opposing hip center. This is a primary-contester estimate, not the player's tactical defensive assignment. Persistent player IDs carry that matchup into the pre-release separation measurement. Each detected shot offers a release review button, underlying measurements, evidence quality, and an exportable JSON report. Detected-person counts can include officials; they are not a verified count of players on court.

- **Separation:** projected distance between player hip centers divided by the shooter's shoulder-to-hip torso length.
- **Contest clearance:** projected distance from the ball to the nearest visible defender wrist, in the same torso units. This is a hand-contest proxy, not a measurement of a blocked shooting lane.
- **Separation change:** change over approximately 0.5 seconds before release, when conservative player matching and stable body scale permit comparison. Positive values mean more projected space.
- **Shot-space score (0–100):** 65% separation component plus 35% contest-clearance component. Separation maps 0.5–3 torso lengths to 0–100; clearance maps 0.15–1.5 to 0–100, both clipped at the endpoints. Separation change is descriptive and does not affect the score.

The formula and thresholds are **unvalidated review heuristics**, not make probability, expected points, a professional player grade, or a claim of optimal shot selection. Evidence quality is also heuristic, not a statistical probability. No score is produced when shooter ownership is ambiguous, teams cannot be separated by jersey appearance, a nearby unidentified player could be the defender, required landmarks are occluded, or ball evidence is weak. Missing measurements remain unavailable, never zero.

If exactly one defender hand is unobserved, the point score stays unavailable but the report supplies **possible score bounds**: separation's contribution alone is the lower bound; adding the visible hand's contest contribution gives the upper bound. The unseen hand could reduce clearance anywhere within that range. These bounds assume the measured separation, visible hand and matchup are correct; they are **not** a statistical confidence interval. Visible-hand clearance is labeled separately from complete contest clearance, and any supported separation measurement remains available.

Footage is sampled at up to 30 FPS; exact release timing still has sampling and detection uncertainty. Distances are aspect-corrected but not court-calibrated: camera angle, depth, player overlap and movement affect the numbers. Compare attempts from a consistent camera setup and review detected shots and outcomes in the video. Passes can still be mistaken for shots. Game mode does not publish arm mechanics across players without reliable persistent identity. Use the separate form mode for individual biomechanics review.

Track IDs are motion- and jersey-assisted; team grouping uses torso colour in the current lighting. Similar uniforms, severe overlap, a tiny player image, cuts, or moving cameras can still break identity. The analyzer reports its selection evidence and withholds uncertain matchups. Professional-use accuracy still needs evaluation on labeled representative game footage. A larger generic model improves detection, but it has not been fine-tuned or calibrated against NBA shot outcomes.

## Architecture

- `app/analyzer.py`: video decoding, pose/ball inference, net optical flow, annotation
- `app/scoring.py`: shot segmentation, release detection, metrics, outcome evidence
- `app/game.py`: conservative 1-on-1 association, projected measurements, transparent shot-space score
- `app/tracking.py`: multi-player track IDs and jersey-colour descriptors
- `app/vision.py`: wide-view multi-person inference, overlapping crops, court filtering and cut detection
- `app/basketball.py`: local basketball-trained ball and player/referee detection
- `app/main.py`: upload/job API
- `app/lan.py`: tokenized LAN/Tailscale sharing and QR pairing
- `web/`: dependency-free local upload and review interface

Set `BALLFORM_YOLO_MODEL` to another Ultralytics detection checkpoint if desired. It must include COCO class 32 (`sports ball`). Set `BALLFORM_GAME_POSE_MODEL` to another Ultralytics COCO-17 pose checkpoint to experiment with larger models. Defaults are stored in `models/` and ignored by git. Larger checkpoints are not automatically more accurate on your footage; compare the annotated output.

The upload API also accepts `camera=auto|broadcast|elevated|courtside` and optional `court=[[x,y],...]` as a JSON form field with normalized coordinates. Existing clients using `mode=one_on_one` continue to work and get the wide-view pipeline by default. The JSON report records the profile, model names, devices, crop usage, court polygon, scene cuts and detected-person counts. Game jobs also save `observations.json` beside `result.json` under `data/jobs/<job_id>/`: these are the detected balls and tracked poses used for scoring, allowing review without repeating model inference. Overriding `BALLFORM_YOLO_MODEL` selects a COCO ball model and disables the specialized broadcast role filter.

Run regression checks with `uv run pytest -q` and `node --check web/app.js`.
