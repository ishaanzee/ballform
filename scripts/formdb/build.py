#!/usr/bin/env python3
"""Build app/form_db.json, the player free-throw signatures that form-match mode compares against.

    P="uv run --inexact python scripts/formdb/build.py"
    $P fetch <youtube id> [--sections '*300-420']      # 1080p video into eval/staging/formdb/video
    $P scan <video> --player "Shai Gilgeous-Alexander" --hand right [--source youtube:<id>]
    $P sheet "Shai Gilgeous-Alexander"                  # contact sheet of every rep, to review
    $P drop "Shai Gilgeous-Alexander" 3 7               # remove reps that are not his free throws
    $P build                                            # write app/form_db.json, print the identification check

scan finds free throws in a video of one player's free throws (a per-player compilation):
a pose model samples the video three times a second for a standing person with a wrist above
the head who stood on the same spot for the second before, then MediaPipe follows that person
on an upscaled crop and app/form_signature.py measures the rep. Only side views of a shooter who stays on the spot are kept. Every other
player in the video who shoots a set shot from the side is kept too, so review the sheet.

Reps live in eval/staging/formdb/reps/<player>.json (not in git). Only numbers derived from
the poses reach app/form_db.json; no footage is stored.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app import form_match  # noqa: E402
from app.form_signature import FEATURES, shot_signatures  # noqa: E402
from app.models import PoseFrame  # noqa: E402

STAGING = ROOT / "eval" / "staging" / "formdb"
from app.analyzer import MATCH_POSE_MODEL as HEAVY, MATCH_POSE_MODEL_URL as HEAVY_URL  # noqa: E402
MP_NAMES = {0: "nose", 11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
            15: "left_wrist", 16: "right_wrist", 23: "left_hip", 24: "right_hip", 25: "left_knee",
            26: "right_knee", 27: "left_ankle", 28: "right_ankle"}
BONES = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
         ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
         ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")]
SCAN_HZ = 3.0
BEFORE_S, AFTER_S = 2.2, 1.6      # window followed around a raised-arm sighting
MAX_SHOULDER_WIDTH = form_match.MAX_SIDE_SHOULDER_WIDTH
MAX_DRIFT = 0.3                   # torso lengths the ankles may move in the 0.8 s before release


def slug(player: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", player.lower()).strip("-")


def reps_path(player: str) -> Path:
    return STAGING / "reps" / f"{slug(player)}.json"


def load_reps(player: str) -> dict:
    path = reps_path(player)
    return json.loads(path.read_text()) if path.exists() else {"player": player, "hand": "right", "reps": []}


def save_reps(data: dict) -> None:
    path = reps_path(data["player"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fetch(video_id: str, sections: str | None) -> int:
    dest = STAGING / "video"
    dest.mkdir(parents=True, exist_ok=True)
    name = video_id + ("_" + re.sub(r"[^0-9-]", "", sections) if sections else "")
    cmd = ["yt-dlp", "--no-warnings", "--no-playlist", "-q", "-f", "bv*[vcodec^=avc1][height<=1080]", "-N", "16",
           "-o", str(dest / f"{name}.mp4"), f"https://www.youtube.com/watch?v={video_id}"]
    if sections:
        cmd[1:1] = ["--download-sections", sections, "--force-keyframes-at-cuts"]
    code = subprocess.run(cmd).returncode
    print(dest / f"{name}.mp4" if code == 0 else "download failed")
    return code


def raised_arm_sightings(video: Path) -> list[tuple[float, tuple[int, int, int, int]]]:
    """(time, person box in pixels) where a standing, fully visible person has a wrist above the nose."""
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "models" / "yolo26s-pose.pt"))
    cap = cv2.VideoCapture(str(video))
    fps, width, height = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    step, found, index = max(1, round(fps / SCAN_HZ)), [], 0
    people: dict[int, np.ndarray] = {}   # sampled frame index -> every person box
    while cap.grab():
        if index % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            result = model.predict(frame, imgsz=1280, conf=0.4, verbose=False)[0]
            people[index] = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
            if result.keypoints is not None and result.keypoints.conf is not None:
                for box, xy, conf in zip(result.boxes.xyxy.cpu().numpy(), result.keypoints.xy.cpu().numpy(),
                                         result.keypoints.conf.cpu().numpy()):
                    x1, y1, x2, y2 = box
                    tall = (y2 - y1) / height
                    if not 0.15 <= tall <= 0.7 or x1 < 4 or x2 > width - 4 or y2 > height - 4:
                        continue
                    if min(conf[[0, 5, 6, 11, 12, 15, 16]]) < 0.5:   # nose, shoulders, hips, ankles
                        continue
                    torso = np.linalg.norm((xy[5] + xy[6]) / 2 - (xy[11] + xy[12]) / 2)
                    if any(conf[w] > 0.4 and xy[0][1] - xy[w][1] > 0.15 * torso for w in (9, 10)):
                        # A free-throw shooter stood on the same spot in the two samples before.
                        if all(_overlaps(box, people.get(index - k * step)) for k in (2, 3)):
                            found.append((index / fps, tuple(int(v) for v in box)))
        index += 1
    cap.release()
    return found


def _overlaps(box: np.ndarray, others: np.ndarray | None, least: float = 0.5) -> bool:
    """Whether some box in ``others`` covers at least ``least`` of ``box`` by intersection over union."""
    if others is None or not len(others):
        return False
    x1, y1 = np.maximum(box[0], others[:, 0]), np.maximum(box[1], others[:, 1])
    x2, y2 = np.minimum(box[2], others[:, 2]), np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]) - inter
    return bool((inter / union).max() >= least)


def group(sightings: list) -> list[tuple[float, tuple[int, int, int, int]]]:
    """First sighting of each raised arm: the same place within 1.5 s is the same raise."""
    events: list[list] = []
    for when, box in sightings:
        cx = (box[0] + box[2]) / 2
        for event in events:
            last_t, last_box = event[-1]
            if when - last_t <= 1.5 and abs(cx - (last_box[0] + last_box[2]) / 2) < (last_box[2] - last_box[0]):
                event.append((when, box))
                break
        else:
            events.append([(when, box)])
    return [event[0] for event in events]


def crop_poses(video: Path, start: float, end: float, crop: tuple[int, int, int, int]):
    """MediaPipe poses of the person in ``crop`` (pixels) from start to end; landmarks are
    normalized to the crop. Returns (poses, crop aspect ratio)."""
    import mediapipe as mp
    if not HEAVY.exists():
        import urllib.request
        urllib.request.urlretrieve(HEAVY_URL, HEAVY)  # noqa: S310
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HEAVY), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1, min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3, min_tracking_confidence=0.3)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = max(0, round(start * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    x1, y1, x2, y2 = crop
    poses = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i in range(round((end - start) * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            scale = 720 / (y2 - y1)
            image = cv2.resize(frame[y1:y2, x1:x2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                int(i * 1000 / fps))
            if result.pose_landmarks:
                marks = result.pose_landmarks[0]
                poses.append(PoseFrame(first + i, (first + i) / fps,
                                       {name: (marks[k].x, marks[k].y, marks[k].visibility or 0.0)
                                        for k, name in MP_NAMES.items()}))
    cap.release()
    return poses, (x2 - x1) / (y2 - y1)


def measure(video: Path, when: float, box: tuple[int, int, int, int], hand: str) -> tuple[dict | None, str]:
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    w, h = box[2] - box[0], box[3] - box[1]
    pad = max(0.8 * w, 0.3 * h)
    crop = (max(0, int(box[0] - pad)), max(0, int(box[1] - 0.25 * h)),
            min(width, int(box[2] + pad)), min(height, int(box[3] + 0.08 * h)))
    poses, aspect = crop_poses(video, when - BEFORE_S, when + AFTER_S, crop)
    reps = [r for r in shot_signatures(poses, aspect, hand) if when - 0.8 <= r["release_s"] <= when + 0.5]
    if not reps:
        return None, "no set-shot motion"
    rep = reps[0]
    if rep["shoulder_width"] > MAX_SHOULDER_WIDTH:
        return None, f"not a side view (shoulder width {rep['shoulder_width']:.2f} torso)"
    track = [f for f, t in zip(rep["skeleton"], form_match.SKELETON_TIMES) if f and -0.8 <= t <= 0]
    if len(track) < 12:
        return None, "pose lost before release"
    ankles = np.asarray([[(f["left_ankle"][k] + f["right_ankle"][k]) / 2 for k in (0, 1)] for f in track])
    drift = float(np.linalg.norm(ankles.max(axis=0) - ankles.min(axis=0)))
    if drift > MAX_DRIFT:
        return None, f"shooter moving ({drift:.2f} torso)"
    stored = [[round(p.time_s, 4), {k: [round(v[0], 4), round(v[1], 4)] for k, v in p.landmarks.items()}] for p in poses]
    return {**rep, "crop": list(crop), "aspect": aspect, "poses": stored}, "ok"


def remeasure(rep: dict, hand: str) -> dict | None:
    """The rep measured again from its stored poses with the current app/form_signature.py."""
    poses = [PoseFrame(i, t, {k: (x, y, 1.0) for k, (x, y) in marks.items()}) for i, (t, marks) in enumerate(rep["poses"])]
    found = [r for r in shot_signatures(poses, rep["aspect"], hand) if abs(r["release_s"] - rep["release_s"]) < 0.4]
    return {**rep, **found[0]} if found and found[0]["shoulder_width"] <= MAX_SHOULDER_WIDTH else None


def scan(video: Path, player: str, hand: str, source: str | None) -> int:
    data = load_reps(player)
    data["hand"] = hand
    source = source or video.stem
    events = group(raised_arm_sightings(video))
    print(f"{len(events)} raised-arm sightings in {video.name}")
    kept = 0
    for when, box in events:
        rep, why = measure(video, when, box, hand)
        print(f"  {when:8.2f} s  box {box}  {why}")
        if rep is None:
            continue
        if any(r["source"] == source and abs(r["release_s"] - rep["release_s"]) < 0.5 for r in data["reps"]):
            continue
        data["reps"].append({**rep, "source": source, "video": str(video)})
        kept += 1
    save_reps(data)
    print(f"kept {kept}; {player} now has {len(data['reps'])} reps. Review: build.py sheet \"{player}\"")
    return 0


def sheet(player: str) -> int:
    """Set-point and release frames of every rep with the measured skeleton drawn on."""
    data = load_reps(player)
    data["reps"] = [remeasure(rep, data["hand"]) or rep for rep in data["reps"]]
    tiles = []
    for index, rep in enumerate(data["reps"]):
        x1, y1, x2, y2 = rep["crop"]
        cap = cv2.VideoCapture(rep["video"])
        fps = cap.get(cv2.CAP_PROP_FPS)
        for label, when in (("set", rep["set_s"]), ("release", rep["release_s"]), ("after", rep["release_s"] + 0.3)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(when * fps))
            ok, frame = cap.read()
            tile = cv2.resize(frame[y1:y2, x1:x2], (200, 260)) if ok else np.zeros((260, 200, 3), np.uint8)
            cv2.putText(tile, f"#{index} {label}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(tile, f"#{index} {label}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            tiles.append(tile)
        cap.release()
        f = rep["features"]
        print(f"#{index} {rep['source']} {rep['release_s']:.2f}s {rep.get('arm_side', '?')}  " + "  ".join(
            f"{k.rsplit('_', 1)[0]} {f[k]}" for k in FEATURES))
    if not tiles:
        print("no reps")
        return 1
    per_row = 12
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    out = STAGING / "sheets" / f"{slug(player)}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]))
    print(out)
    return 0


def drop(player: str, indexes: list[int]) -> int:
    data = load_reps(player)
    data["reps"] = [r for i, r in enumerate(data["reps"]) if i not in set(indexes)]
    save_reps(data)
    print(f"{player}: {len(data['reps'])} reps left")
    return 0


def _same_shot(a: dict, b: dict) -> bool:
    return all(a["features"][k] is not None and b["features"][k] is not None and abs(a["features"][k] - b["features"][k]) < 0.5
               for k in ("set_elbow_deg", "release_arm_deg", "knee_bend_deg"))


def build(min_reps: int) -> int:
    players = [json.loads(p.read_text()) for p in sorted((STAGING / "reps").glob("*.json"))]
    for player in players:
        reps = [r for r in (remeasure(rep, player["hand"]) for rep in player["reps"]) if r]
        # Compilations overlap: the same free throw from two sources measures the same.
        player["reps"] = [r for i, r in enumerate(reps) if not any(_same_shot(r, e) for e in reps[:i])]
    players = [{"name": p["player"], "hand": p["hand"], "reps": p["reps"]} for p in players]
    db = form_match.build_db(players, min_reps)
    form_match.DB_PATH.write_text(json.dumps(db, separators=(",", ":")))
    print(f"wrote {form_match.DB_PATH.relative_to(ROOT)}")
    for side, view in db["views"].items():
        print(f"{side} arm: " + ", ".join(f"{p['name']} ({p['reps']})" for p in view["players"]))
        if len(view["players"]) < 2:
            continue
        for name, feature in view["features"].items():
            print(f"  {name:<20} weight {feature['weight']:.2f}  between players {feature['scale']:.3f}  "
                  f"within a player {feature['within']:.3f}")
        # Identification check: each rep, left out of its own player's template, should match that player.
        names = {p["name"] for p in view["players"]}
        hits = top3 = total = 0
        for player in players:
            for i, rep in enumerate(player["reps"]):
                if rep["arm_side"] != side or player["name"] not in names:
                    continue
                # The same players as the templates, so a player at the minimum stays a candidate.
                others = [{**p, "reps": [r for j, r in enumerate(p["reps"]) if p is not player or j != i]}
                          for p in players if p["name"] in names]
                result = form_match.match([rep], form_match.build_db(others, 1))
                ranked = [m["name"] for m in result["matches"]] if result else []
                total += 1
                hits += ranked[:1] == [player["name"]]
                top3 += player["name"] in ranked[:3]
        if total:
            print(f"  leave-one-rep-out identification: top-1 {hits}/{total}, top-3 {top3}/{total} "
                  f"(chance top-1 {total / len(names):.0f}/{total})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch"); f.add_argument("video_id"); f.add_argument("--sections")
    s = sub.add_parser("scan"); s.add_argument("video", type=Path); s.add_argument("--player", required=True)
    s.add_argument("--hand", choices=["left", "right"], default="right"); s.add_argument("--source")
    sh = sub.add_parser("sheet"); sh.add_argument("player")
    d = sub.add_parser("drop"); d.add_argument("player"); d.add_argument("indexes", type=int, nargs="+")
    b = sub.add_parser("build"); b.add_argument("--min-reps", type=int, default=3, help="reps a player needs on a camera side")
    args = parser.parse_args()
    if args.command == "fetch":
        return fetch(args.video_id, args.sections)
    if args.command == "scan":
        return scan(args.video, args.player, args.hand, args.source)
    if args.command == "sheet":
        return sheet(args.player)
    if args.command == "drop":
        return drop(args.player, args.indexes)
    return build(args.min_reps)


if __name__ == "__main__":
    sys.exit(main())
