#!/usr/bin/env python3
"""Contact sheets of clip frames for labeling, each tile captioned with frame and time.

    uv run --inexact python scripts/labeling/frames.py <job id or clip> --overview --out eval/draft-frames/<job>_overview.jpg
    uv run --inexact python scripts/labeling/frames.py <job> --at 3.2 --window 0.6 --out eval/draft-frames/<job>_release.jpg
    uv run --inexact python scripts/labeling/frames.py <job> --at 4.4 --window 0.5 --step 2 \\
        --crop 1100,200,1500,500 --out eval/draft-frames/<job>_rim.jpg

--at/--window: consecutive frames from at - window/2 to at + window/2 (every --step
frames). --crop x1,y1,x2,y2 (pixels in the 1920x1080 frame) crops and upscales each
tile to at least 640 px wide. --overview: one frame every 0.25 s over the whole clip.
A sheet is at most ~2000 px on its longer side; more tiles go to extra pages
(<out>_p2.jpg, ...). Times are frame / fps, the same clock Ballform and labels use.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = ROOT / "data" / "jobs"
MAX_SIDE = 2000
CAPTION = 34


def resolve_clip(clip: str) -> Path:
    path = Path(clip)
    if path.exists():
        return path
    job = JOBS_DIR / clip / "input.mp4"
    if job.exists():
        return job
    raise SystemExit(f"no clip at {clip} and no job {clip} under data/jobs")


def read_frames(path: Path, numbers: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    wanted = sorted(set(numbers))
    out = {}
    if not wanted:
        return out
    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    position = wanted[0]
    for number in wanted:
        while position < number:
            cap.grab()
            position += 1
        ok, image = cap.read()
        position += 1
        if ok:
            out[number] = image
    cap.release()
    return out


def tile(image: np.ndarray, number: int, fps: float, width: int, crop) -> np.ndarray:
    if crop:
        x1, y1, x2, y2 = crop
        image = image[y1:y2, x1:x2]
    h, w = image.shape[:2]
    height = max(1, round(h * width / w))
    interp = cv2.INTER_CUBIC if width > w else cv2.INTER_AREA
    body = cv2.resize(image, (width, height), interpolation=interp)
    bar = np.zeros((CAPTION, width, 3), np.uint8)
    text = f"f{number}  {number / fps:.3f}s"
    cv2.putText(bar, text, (8, CAPTION - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, body])


def layout(count: int, aspect: float, min_width: int, overview: bool) -> tuple[int, int, int]:
    """(columns, rows per page, tile width) keeping each page within MAX_SIDE."""
    if overview:
        cols = 5 if aspect >= 1.2 else 7
        width = MAX_SIDE // cols - 4
        tile_h = width / aspect + CAPTION
        rows = max(1, min(math.ceil(count / cols), int(MAX_SIDE // (tile_h + 4))))
        return cols, rows, width
    width = max(min_width, 320)
    cols = max(1, min(count, MAX_SIDE // (width + 4)))
    tile_h = width / aspect + CAPTION
    rows = max(1, int(MAX_SIDE // (tile_h + 4)))
    return cols, rows, width


def sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    canvas = np.full((rows * (h + 4) + 4, cols * (w + 4) + 4, 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = 4 + r * (h + 4), 4 + c * (w + 4)
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="job id or clip path")
    parser.add_argument("--at", type=float, help="centre time in seconds")
    parser.add_argument("--window", type=float, default=0.6, help="total span in seconds (default 0.6)")
    parser.add_argument("--step", type=int, default=1, help="every Nth frame (default 1)")
    parser.add_argument("--crop", help="x1,y1,x2,y2 in source pixels; tiles are upscaled")
    parser.add_argument("--width", type=int, default=640, help="tile width for --at sheets (default 640)")
    parser.add_argument("--overview", action="store_true", help="one frame every 0.25 s over the whole clip")
    parser.add_argument("--every", type=float, default=0.25, help="overview spacing in seconds (default 0.25)")
    parser.add_argument("--out", type=Path, required=True, help="output .jpg (eval/draft-frames/<job>_<tag>.jpg)")
    args = parser.parse_args(argv)
    path = resolve_clip(args.clip)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    crop = None
    if args.crop:
        try:
            crop = [int(float(v)) for v in args.crop.split(",")]
            assert len(crop) == 4
        except (ValueError, AssertionError):
            raise SystemExit("--crop must be x1,y1,x2,y2") from None
        crop = [max(0, min(crop[0], width - 2)), max(0, min(crop[1], height - 2)),
                max(0, min(crop[2], width)), max(0, min(crop[3], height))]
        if crop[2] - crop[0] < 8 or crop[3] - crop[1] < 8:
            raise SystemExit("--crop box is empty")
    if args.overview:
        numbers = sorted({min(count - 1, round(k * args.every * fps)) for k in range(int(count / fps / args.every) + 1)})
    else:
        if args.at is None:
            raise SystemExit("give --at <seconds> or --overview")
        first = max(0, round((args.at - args.window / 2) * fps))
        last = min(count - 1, round((args.at + args.window / 2) * fps))
        numbers = list(range(first, last + 1, max(1, args.step)))
    frames = read_frames(path, numbers)
    numbers = [n for n in numbers if n in frames]
    if not numbers:
        raise SystemExit("no frames in that span")
    cw, ch = (crop[2] - crop[0], crop[3] - crop[1]) if crop else (width, height)
    cols, rows, tile_w = layout(len(numbers), cw / ch, args.width, args.overview)
    per_page = cols * rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pages = [numbers[i:i + per_page] for i in range(0, len(numbers), per_page)]
    for p, page in enumerate(pages):
        out = args.out if p == 0 else args.out.with_name(f"{args.out.stem}_p{p + 1}{args.out.suffix}")
        image = sheet([tile(frames[n], n, fps, tile_w, crop) for n in page], cols)
        cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{out}  frames f{page[0]}-f{page[-1]} ({page[0] / fps:.3f}-{page[-1] / fps:.3f} s), "
              f"{len(page)} tiles, {image.shape[1]}x{image.shape[0]}")
    scale = cw / tile_w
    print(f"tile pixels x {scale:.2f} = source pixels" + (f" (offset {crop[0]},{crop[1]})" if crop else "")
          + "; use source pixels for --crop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
