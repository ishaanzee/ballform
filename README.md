# ballform

Ballform is a basketball video analyzer for people who want more than “nice shot.”

It looks at a single shooter’s mechanics, or at the little geometry of a game possession: who had the ball, who was the primary defender, how much space existed at release, whether the contest got tighter, and whether the shot got up cleanly. It runs on your machine. Your clips do not get uploaded to somebody else’s dashboard, and there is no subscription hiding behind the demo.

The numbers are useful review signals, not a scouting department in a box. Compare clips from the same camera. Watch the annotated video. If the tracker is unsure, the report should say so.

## see it on a real clip [Full Demo](https://www.ishaanmehta.dev/projects/basketball)

<a href="https://www.youtube.com/watch?v=6gkVcpQtEcA"><img src="https://img.youtube.com/vi/6gkVcpQtEcA/0.jpg" alt="Before: raw shooting clip" width="640" height="360"></a>
raw shooting clip

<a href="https://www.youtube.com/watch?v=uuiYnWBlgpQ"><img src="https://img.youtube.com/vi/uuiYnWBlgpQ/maxresdefault.jpg" alt="After: Ballform review" width="640" height="360"></a>
after ballform

![Before shot details](docs/demo/shot-details-before.png)
simple metrics

![After shot details](docs/demo/shot-details-after.png)
advanced metrics

## get it running

You need Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv):

```bash
uv sync --extra dev --python 3.12
uv run uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Thats it. `ffmpeg` is worth installing if you want clean H.264 review videos in the browser.

The first run downloads the pose and detection weights into `models/`. They are cached after that. Uploads, reports, observations, and rendered videos live in `data/jobs/`. Stop the server with `Ctrl+C`.

## getting a clip that does not sabotage the model

For form work, keep the shooting arm, ball, feet, rim, and net visible. A side view is best for release and arc; a rear-oblique view is better for alignment. 60 fps and 1080p are a good target. A phone three to six feet high is usually enough. Do not digitally zoom halfway through the possession.

For game footage, short continuous half-court possessions are the sweet spot. NBA skycam and elevated pickup clips are supported, but replays, cuts, graphics, extreme zooms, and a rim that disappears behind the broadcast edit are still hard problems. A five-on-five clip is fine even when the UI says 1-on-1; the analyzer still tries to identify the ball carrier and the primary contesting defender.

The court polygon is optional. If you draw one, keep it convex and cover the playable area, not the benches. It stays fixed in image coordinates, so split a clip when the camera changes to another angle.

## camera choices

`Auto` is a preset: close-up pipeline for form, wide pipeline for game mode.

`Stationary courtside` is the cleanest choice when you want make/miss. Mark a tight rim box in the preview.

`NBA / elevated broadcast` is for player and ball tracking in a wide view. It intentionally does not claim make/miss from a rim box that is frozen while the camera pans.

`Moving broadcast + tracked rim` is the extra option for a continuous pan or moderate zoom. Scrub to any frame where the hoop is clear, mark the rim, and let the local CSRT tracker follow it forward and backward from that timestamp. It can score the rim crossing, use net motion as supporting evidence, and place the green make pulse over the moving hoop. A hard cut, a lost track, or an implausible tracker jump ends outcome scoring instead of producing a confident-looking lie.

The make classifier wants a visible downward crossing through the rim. If the ball vanishes at the hoop, it can call a **likely make** only when the descending path projects through the rim and localized net motion arrives afterward. Net movement by itself never turns an airball into a make. Green animation means the analyzer found a verified or likely make; it is not a broadcast replay graphic.

## what game mode actually measures

The game pass uses multi-person pose detection, a basketball-trained detector, motion and jersey appearance to keep track of people. It chooses a shooter from recent hand/ball contact and release evidence, then chooses the nearest plausible opposing contest. Persistent IDs make the pre-release comparison less random, but crowded frames, similar jerseys, tiny players, officials, and occlusion can still break identity.

The shot-space score is deliberately transparent:

```text
65% separation score + 35% contest-clearance score
```

Separation is projected hip-to-hip distance in shooter torso lengths. Contest clearance is projected ball-to-defender-wrist distance in the same units. Separation change over roughly half a second is reported as context, not secretly folded into the grade. The score is a 0–100 review heuristic, not make probability, expected points, or a professional player grade. Missing evidence stays missing; it does not become a zero.

Form mode reports 2D image-plane estimates such as release timing, launch angle, elbow angle, and upper-arm elevation. These are good for comparing your own reps from the same setup. They are not calibrated 3D biomechanics.

## phone upload, still local

If the clip is on an iPhone, install [Tailscale for macOS](https://tailscale.com/download/mac) and [Tailscale for iOS](https://tailscale.com/download/ios), sign into the same account, then run:

```bash
uv run ballform-share --network tailscale
```

Scan the QR code in the terminal with the iPhone camera. Safari opens a private pairing link to the Mac. The video still runs on the Mac and stays on the Mac; Tailscale is only the encrypted pipe. Keep the terminal and Tailscale open while uploading. `--network lan` is available if both devices are on the same Wi-Fi. The older `ballform-lan` command remains available.

## models, memory, and reality

Game mode defaults to YOLO26s-pose and offers YOLO26m-pose in the model selector. YOLO26s was about 11% faster on the current 10-second sample, with similar shot detection; compare more clips before treating that as a general result. Only one analysis runs at a time. Game analysis also uses overlapping wide-view crops and a basketball-trained RF-DETR Medium detector. The full-frame pose view and both side crops retain their original input sizes; no frames or crop views are skipped. When side-crop ball detection is needed, it runs alongside pose inference. The app is intended for Apple silicon with 36 GB RAM (16 GB should work with smaller clips). The downloaded JSON report now breaks out pose, ball, rim tracking, frame processing, review-overlay, and encoding times. Pose and ball times are subsets of frame processing and can overlap, so do not add them to the total. Wide-view analysis is offline processing, not real-time playback.

On Apple-silicon Macs, the basketball detector now uses ONNX Runtime's Core ML provider by default, running it on CPU + GPU (`MLComputeUnits=CPUAndGPU`) without INT8 quantization or changing the input size. This transformer ran about 2.6x faster per call on the GPU than on the Neural Engine (46 ms vs 119 ms on an M3 Pro). On the 10-second moving-broadcast sample, the whole analysis took 80.1s versus 98.5s with CPU + Neural Engine, with an identical shot result; frame processing fell from 56.9s to 45.8s. `ALL` processed frames about as fast but took 46s instead of 25s to load. Set `BALLFORM_COREML_UNITS` to `ALL`, `CPUAndNeuralEngine` or `CPUOnly` to compare; each setting keeps its own cache. Other systems use CPU, and an automatic Core ML initialization or inference failure falls back to CPU. To force the previous path, start the app with `BALLFORM_BALL_BACKEND=cpu uv run uvicorn app.main:app --reload`; `BALLFORM_BALL_BACKEND=coreml` explicitly requests Core ML and fails rather than falling back. The report records the requested and actual backend. The first run compiles a cache under `models/coreml-cache/` and can take longer to start; later runs reuse it. Core ML may still leave some operations on CPU. The detector can be compared on every frame and crop with `uv run python scripts/benchmark_ball_backends.py /path/to/clip.mp4`. On the current 10-second sample, the cached Core ML run took 101.82s versus 172.68s with CPU, with identical shot and diagnostic report fields; compare more clips before generalizing that result.

On a Mac using MPS pose and the Core ML basketball detector, game mode now keeps one additional frame in flight. A second detector session examines the next frame while the current frame's pose, tracking, and annotation finish; tracking and scoring still consume frames in source order. This preserves the full input resolution and analyzed frame rate. Set `BALLFORM_FRAME_PIPELINE=1` when starting the server to disable the overlap, or `BALLFORM_FRAME_PIPELINE=2` to request it explicitly. The report records the actual pipeline depth and any fallback. On the same 10-second moving-broadcast clip, the two-frame pipeline took 83.50s versus 102.48s with one frame; both runs produced identical observation files and shot results. More in-flight frames are not automatically faster because detector calls can contend for the same compute hardware.

The server loads the game pose model and both detector sessions once and reuses them for every job, instead of reloading them per analysis; Core ML otherwise spends about 12s preparing each detector session on every load. On startup it preloads and warms the default models in the background (about 10s); a job submitted during that time waits until preloading finishes. On the 10-second sample this cut a job from 80.1s to about 55s with identical observations. Set `BALLFORM_PRELOAD=0` to load models on the first job instead. Each report lists any models loaded during that job under `vision.models_loaded_this_job`.

Game pose can run on the Neural Engine while the basketball detector uses the GPU. Export the pose weights once:

```bash
YOLO_AUTOINSTALL=False uv run --with onnx --with onnxslim --with onnxconverter-common \
    python scripts/export_pose_coreml.py yolo26s-pose yolo26m-pose
```

This writes fixed-shape fp16 ONNX files to `models/`, which the app then uses automatically. fp16 matters: the Neural Engine only runs fp16 programs, and an fp32 model silently runs on the CPU at about the same speed as CPU-only. The exports fit 16:9 footage. Other aspect ratios keep using PyTorch pose call by call, so every input matches the original preprocessing. On the 10-second sample, frame processing fell from 45.0s to 30.0s and the job from 54.0s to 38.9s. The shot, make frame, shooter and defender were unchanged. fp16 keypoints moved the shot-space score from 97.9 to 97.4 (separation 2.997 vs 2.977 torso lengths) and changed which track ID labelled the same ball handler on 18 end-of-clip frames. Set `BALLFORM_POSE_BACKEND=torch` to use PyTorch/MPS pose, or `coreml` to fail rather than fall back. Reports record `vision.pose_backend` and any fallback.

The ball-handler highlight is decoded after all frames are analyzed rather than frame by frame. Each tracked player, plus "nobody", is scored on every frame from wrist contact with the ball, a low ball beside the body (mid-dribble), and the basketball detector's player-in-possession class. The most consistent sequence over the whole clip wins (`app/possession.py`). Keeping a handler is free. Picking up a loose ball is cheap, so a catch-and-shoot still registers. Taking the ball from another player costs more, so a few frames of a defender's hand near the ball do not relabel the dribbler. Future frames also let passes switch on the catch, and the handler ends while a shot or pass is in flight. `observations.json` keeps the previous frame-by-frame result as `handler_online`, and `BALLFORM_HANDLER=online` restores it for the review video.

Wrist contact counts in full only when the ball is within about 0.4 torso lengths of a wrist, and fades to nothing at 0.9. A loose ball bouncing past a player's hand in the image, such as after a make, therefore no longer reads as possession. When two hands are near the ball, the clearly closer one takes the credit. Players the detector sees but pose estimation misses, usually because a teammate or defender hides them, also compete for the ball. When one of them is holding it, the highlight shows nobody rather than moving to the visible neighbour.

After all frames are analyzed, player tracks are stitched within each camera segment. The frame-by-frame tracker can split one player into several IDs when the ball or arms hide the jersey colour it matches on, or when the player crouches. Two fragments join when they never appear on the same frame and every switch between them is a short, plausible continuation: at most 0.5s, a small jump and a similar body scale. Stitching is deliberately conservative. A missed join leaves a duplicate label, but a wrong one would swap two players. Player labels in the review video are drawn after stitching, and reports record the count under `diagnostics.tracks_stitched`. A camera cut is confirmed one sampled frame late, so the cut's first frame is now re-tracked with fresh IDs instead of carrying the previous shot's.

You can experiment with other local Ultralytics checkpoints:

```bash
export BALLFORM_YOLO_MODEL=/path/to/model.pt
```

The replacement ball model needs COCO class 32 (`sports ball`). A bigger generic checkpoint is not automatically better on NBA broadcasts or pickup footage, so compare its annotated output before trusting it.

## limits worth knowing before you trust a number

Camera movement changes apparent distances. Jerseys can look alike. Players overlap. A ball can be hidden for exactly the frames that matter. Passes and slow-motion edits can resemble shots. Cuts reset tracking. The analyzer reports evidence and confidence, but this project has not been calibrated against a labeled NBA shot-outcome dataset. Treat it like a sharp review assistant, not an oracle.

## code map

`app/analyzer.py` handles decoding, inference, tracking, net flow, and annotated video. `app/scoring.py` segments arcs and computes form/outcome evidence. `app/game.py` handles shooter/defender association and the shot-space score. `app/tracking.py` owns multi-player IDs and jersey descriptors. `app/vision.py` handles wide-view detection, crops, court filtering, and cuts. `app/main.py` is the local upload/job API. `app/lan.py` handles the tokenized LAN/Tailscale link.

Run the checks with:

```bash
uv run pytest -q
node --check web/app.js
```

Ballform is AGPL-3.0-only because it integrates the AGPL-licensed Ultralytics package and model. MediaPipe is Apache-2.0. See `LICENSE`, `NOTICE.md`, `CONTRIBUTING.md`, and `SECURITY.md` before distributing a modified service.
