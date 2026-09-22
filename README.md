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

Game mode defaults to YOLO26s-pose and offers YOLO26m-pose in the model selector. YOLO26s was about 11% faster on the current 10-second sample, with similar shot detection; compare more clips before treating that as a general result. Only one analysis runs at a time. Game analysis also uses overlapping wide-view crops and a basketball-trained RF-DETR Medium detector. The pipeline processes frames and crops sequentially and is intended for Apple silicon with 36 GB RAM (16 GB should work with smaller clips). Apple GPU (MPS) is used for pose when available; the portable ONNX ball detector runs on CPU. Wide-view analysis is offline processing, not real-time playback.

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
