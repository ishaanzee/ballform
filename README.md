# Ballform

Basketball video analysis with a **Next.js / React / TypeScript frontend** and a **Python vision backend**. Choose **Shooting form** for left- or right-arm mechanics, or **1-on-1 game** for projected shooter–defender separation and contest review. Both modes use the same existing analysis pipeline. Run everything locally, or deploy the frontend to Vercel and connect an HTTPS analysis worker.

## Run locally

Use Node.js 22+ and Python 3.11 or 3.12. Start the backend in one terminal:

```bash
uv sync --extra dev --python 3.12
uv run uvicorn app.main:app --reload
```

In another terminal:

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Open <http://localhost:3000>. The default frontend environment points directly to <http://localhost:8000>. Localhost origins on port 3000 are allowed by the backend by default. No access token is needed for this local development setup unless you configure one.

On first analysis, the MediaPipe pose model and YOLO weights download into local caches. Later local runs are offline. Install `ffmpeg` for browser-friendly H.264 review videos. The original standalone interface remains at <http://127.0.0.1:8000> for the existing phone-sharing workflow.

## Deploy the Next.js frontend to Vercel

1. Import this repository into Vercel.
2. Set **Root Directory** to **`frontend`** and select the **Next.js** framework preset. Build command: `npm run build`. Leave the output directory at its default.
3. Set **`NEXT_PUBLIC_API_URL`** to your analysis worker's public HTTPS base URL, such as `https://analysis.example.com`. This is an address, **not a secret**. Never put the access token in a `NEXT_PUBLIC_` variable.
4. Deploy. Configure the backend's `BALLFORM_CORS_ORIGINS` with the exact Vercel production origin, then restart the backend. For previews, explicitly add each origin that should work.
5. Open the app and enter the backend access token. It stays in the tab's session storage. Upload a clip, choose a mode, and analyze.

The frontend can build without the backend URL; it displays a setup notice and disables analysis until configured. `NEXT_PUBLIC_API_URL` is baked into the browser build, so changing it requires a redeploy.

Videos upload **directly from the browser to the Python service**; they are never proxied through Vercel functions. [Vercel limits function request/response bodies to 4.5 MB](https://vercel.com/docs/functions/limitations), while Ballform accepts up to 750 MB and runs long-lived model jobs. Deploying the frontend alone does not provide video analysis. An HTTPS frontend also needs an HTTPS backend; a localhost or plain HTTP worker cannot serve a public Vercel deployment.

### Deploy the analysis service

The root `Dockerfile` packages the existing FastAPI worker, CPU vision dependencies, and FFmpeg. Deploy it on a container host with HTTPS, writable persistent storage, and enough memory for the vision models. Build it from the **repository root**, not `frontend/`.

Configure these backend variables (see `backend.env.example`):

| Variable | Value |
| --- | --- |
| `BALLFORM_REQUIRE_TOKEN` | `1` (the Docker default; refuses startup without a token) |
| `BALLFORM_ACCESS_TOKEN` | A long random private token, entered in the frontend by trusted users |
| `BALLFORM_CORS_ORIGINS` | Exact frontend origins, comma-separated, e.g. `https://your-project.vercel.app` |
| `BALLFORM_JOBS_DIR` | `/data/jobs` |
| `BALLFORM_MODELS_DIR` | `/data/models` |
| `PORT` | Host-supplied port, or `8000` |

Mount a persistent volume at `/data` writable by the container's `ballform` user. The healthcheck path is `/healthz`. First analysis downloads model weights; allow outbound access for that download. Your host/reverse proxy must allow the upload size and upload duration you intend to support. Configure access logs to omit query strings: video playback uses the access token in its URL because native video elements cannot set a custom authorization header.

Run **one instance and one Uvicorn worker**: the job queue and progress are process-local. Completed reports survive restarts when `/data` persists, but interrupted or queued jobs must be submitted again. Do not enable horizontal autoscaling without first adding a shared job queue and object storage. The shared token is suitable for a trusted individual/team, not public multi-tenant access: holders can access all jobs. Uploaded footage remains on the backend until you remove it.

Example local container smoke test (set the token in your shell first):

```bash
docker build -t ballform-worker .
docker run --rm -p 8000:8000 \
  -e BALLFORM_ACCESS_TOKEN \
  -e BALLFORM_CORS_ORIGINS=http://localhost:3000 \
  -v ballform-data:/data ballform-worker
```

## Optional: original local iPhone interface with Tailscale

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

If required-Tailscale mode reports that Tailscale is stopped, open the app on both devices and enable it before retrying. The Vercel frontend plus hosted analysis service above is the alternative when you need access without a private network.

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

### 1-on-1 game review

Select **1-on-1 game** before uploading. Use a fixed sideline or slightly angled camera with both players and their hands visible. Mark the rim for outcome estimates. Each detected shot offers a release review button, underlying measurements, evidence quality, and an exportable JSON report.

- **Separation:** projected distance between player hip centers divided by the shooter's shoulder-to-hip torso length.
- **Contest clearance:** projected distance from the ball to the nearest visible defender wrist, in the same torso units. This is a hand-contest proxy, not a measurement of a blocked shooting lane.
- **Separation change:** change over approximately 0.5 seconds before release, when conservative player matching and stable body scale permit comparison. Positive values mean more projected space.
- **Shot-space score (0–100):** 65% separation component plus 35% contest-clearance component. Separation maps 0.5–3 torso lengths to 0–100; clearance maps 0.15–1.5 to 0–100, both clipped at the endpoints. Separation change is descriptive and does not affect the score.

The formula and thresholds are **unvalidated review heuristics**, not make probability, expected points, a professional player grade, or a claim of optimal shot selection. Evidence quality is also heuristic, not a statistical probability. No score is produced when shooter ownership is ambiguous, an extra player is detected, required landmarks are occluded, or ball evidence is weak. Missing measurements remain unavailable, never zero.

Footage is sampled at up to 30 FPS; exact release timing still has sampling and detection uncertainty. Distances are aspect-corrected but not court-calibrated: camera angle, depth, player overlap and movement affect the numbers. Compare attempts from a consistent camera setup and review detected shots and outcomes in the video. Passes can still be mistaken for shots. Game mode does not publish arm mechanics across players without reliable persistent identity. Use the separate form mode for individual biomechanics review.

The multi-person configuration uses MediaPipe's documented [`num_poses` option](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarkerOptions). It allows a third detection so crowded frames can be rejected, rather than assigning that person as the defender. This is a review aid; professional-use accuracy needs evaluation on labeled representative game footage.

## Architecture

- `app/analyzer.py`: video decoding, pose/ball inference, net optical flow, annotation
- `app/scoring.py`: shot segmentation, release detection, metrics, outcome evidence
- `app/game.py`: conservative 1-on-1 association, projected measurements, transparent shot-space score
- `app/main.py`: upload/job API
- `app/lan.py`: tokenized LAN/Tailscale sharing and QR pairing
- `frontend/`: Next.js App Router, React upload/review UI, typed API client (Vercel root directory)
- `web/`: original standalone interface for local LAN/Tailscale compatibility
- `Dockerfile`: independently hosted Python analysis worker

Set `BALLFORM_YOLO_MODEL` to another Ultralytics detection checkpoint if desired. It must include COCO class 32 (`sports ball`). The default checkpoint is stored in `models/` and ignored by git.

Run backend regression checks with `uv run pytest -q`. In `frontend/`, run `npm test`, `npm run typecheck`, and `npm run build`.
