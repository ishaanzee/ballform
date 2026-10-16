#!/usr/bin/env python3
"""Read the broadcast scorebug through a video and line it up with ESPN play-by-play.

Build the timeline (needs Apple Vision through pyobjc; ~1 min for a 15 min video):

    uv run --inexact --with pyobjc-framework-Vision --with pyobjc-framework-Quartz \\
        python scripts/labeling/scorebug.py eval/staging/video/<yt id>.mp4 --event <espn id>

Match plays to video times (no OCR needed):

    uv run --inexact python scripts/labeling/scorebug.py match eval/staging/scorebug/<yt id>.json <espn id>

The bug is found by its text, not a fixed crop: a couple of dozen frames are read
whole, the period token (1ST, Q4, OT...) and game clock (9:59, 58.4) are located, and
every sampled frame is then read only inside the box around them. Frames with no
clock (replays, cutaways, graphics) get conf "none". With --event the scores are
checked against the play-by-play: conf "high" means period, clock and a score that
ESPN shows at that point of the game were all read; "low" means something is missing
or does not fit. OCR results are cached per frame next to the timeline, so a rerun
(or a finer --step) only reads new frames.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pbp  # noqa: E402

OUT_DIR = pbp.STAGING / "scorebug"
LOOKALIKES = str.maketrans({"З": "3", "з": "3", "Э": "3", "э": "3", "н": "h", "О": "O", "о": "o", "Т": "T", "т": "t", "Н": "H", "К": "K",
                            "к": "k", "Е": "E", "е": "e", "С": "C", "с": "c", "Р": "P", "р": "p", "А": "A",
                            "а": "a", "В": "B", "М": "M", "Х": "X", "І": "I", "Ι": "I", "ı": "i", "：": ":",
                            "∶": ":", "꞉": ":", ";": ":", "‚": ",", "·": ".", "•": "."})
PERIOD = re.compile(r"^(?:([1-4Il])(?:ST|ND|RD|TH)|Q([1-4])|([1-4])Q|([2-9])?OT([2-9])?)$", re.I)
MMSS = re.compile(r"^(\d{1,2}):(\d{2})$")
TENTHS = re.compile(r"^(\d{1,2})\.(\d)$")
SHOT = re.compile(r"^:(\d{1,2})$")
INTEGER = re.compile(r"^\d{1,3}$")
PIECES = re.compile(r"\d{1,2}:\d{2}|\d{1,2}\.\d|:\d{1,2}|\d+|[A-Za-z]+")


# ---------------------------------------------------------------- parsing (pure)

def parse_period(token: str) -> int | None:
    """'1ST' / 'IST' / 'Q4' / '4Q' / '4TH' / 'OT' / '2OT' / 'OT2' -> period number (OT = 5)."""
    found = PERIOD.match(token.translate(LOOKALIKES).strip())
    if not found:
        return None
    ordinal, q, q2, ot_pre, ot_post = found.groups()
    if ordinal:
        return 1 if ordinal in "Il" else int(ordinal)
    if q or q2:
        return int(q or q2)
    return 4 + int(ot_pre or ot_post or 1)


def parse_game_clock(token: str) -> float | None:
    """A scorebug game clock: '9:59' -> 599, '11:43' -> 703, '58.4' -> 58.4. A shot clock
    written ':17' is not a game clock."""
    token = token.translate(LOOKALIKES).strip()
    found = MMSS.match(token)
    if found:
        minutes, seconds = int(found.group(1)), int(found.group(2))
        value = minutes * 60 + seconds
        return float(value) if seconds < 60 and value <= 720 else None
    found = TENTHS.match(token)
    if found:
        return int(found.group(1)) + int(found.group(2)) / 10
    return None


@dataclass
class Token:
    text: str
    conf: float
    box: tuple[float, float, float, float]  # x, y, w, h in pixels, top-left origin

    @property
    def cx(self) -> float:
        return self.box[0] + self.box[2] / 2

    @property
    def cy(self) -> float:
        return self.box[1] + self.box[3] / 2

    @property
    def h(self) -> float:
        return self.box[3]


def tokens_of(observations) -> list[Token]:
    """Split OCR lines ('83 DEN 94 4TH') into tokens, each with its share of the line's box."""
    out = []
    for text, conf, (x, y, w, h) in observations:
        text = text.translate(LOOKALIKES)
        n = max(1, len(text))
        for word in re.finditer(r"\S+", text):
            pieces = [(word.group(), word.start())]
            if parse_period(word.group()) is None and parse_game_clock(word.group()) is None:
                pieces = [(m.group(), word.start() + m.start()) for m in PIECES.finditer(word.group())]
            for piece, start in pieces:
                out.append(Token(piece, conf, (x + w * start / n, y, w * len(piece) / n, h)))
    return out


def _dist(a: Token, b: Token) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def reading_order(a: Token, b: Token) -> tuple[Token, Token]:
    """Away first: left of the other on one row, or above it when stacked."""
    if abs(a.cy - b.cy) < .6 * max(a.h, b.h):
        return (a, b) if a.cx <= b.cx else (b, a)
    return (a, b) if a.cy < b.cy else (b, a)


@dataclass
class Reading:
    period: int | None = None
    clock: float | None = None
    clock_text: str = ""
    away: int | None = None
    home: int | None = None
    shot_clock: float | None = None
    conf: str = "none"
    numbers: list = field(default_factory=list)  # (value, cx, cy, h) score candidates

    def complete(self) -> bool:
        return None not in (self.period, self.clock, self.away, self.home)


def parse_reading(observations, valid_scores=None, scores_for=None) -> Reading:
    """Period, game clock, scores and shot clock from one frame's OCR observations.

    valid_scores: optional set of (away, home) states from the play-by-play, and
    scores_for(period, clock) the states ESPN shows around that point of the game; when
    given, the score pair is one ESPN shows (in either order), preferring nearby ones."""
    tokens = tokens_of(observations)
    reading = Reading()
    clocks = [t for t in tokens if MMSS.match(t.text) and parse_game_clock(t.text) is not None]
    periods = [t for t in tokens if parse_period(t.text) is not None]
    tenths = [t for t in tokens if TENTHS.match(t.text)]
    used = set()
    clock = None
    if clocks:
        if periods and len(clocks) > 1:
            clock = min(clocks, key=lambda c: min(_dist(c, p) for p in periods))
        else:
            clock = max(clocks, key=lambda c: c.h)
    elif tenths:
        ordered = sorted(tenths, key=lambda t: parse_game_clock(t.text), reverse=True)
        if periods:
            ordered.sort(key=lambda c: min(_dist(c, p) for p in periods))
        clock = ordered[0]
        rest = [t for t in tenths if t is not clock]
        if rest:  # a shot clock under 5 s shows tenths too; the game clock is the larger
            other = rest[0]
            if parse_game_clock(other.text) > parse_game_clock(clock.text):
                clock, other = other, clock
            reading.shot_clock = parse_game_clock(other.text)
            used.add(id(other))
    if clock is None:
        return reading
    used.add(id(clock))
    reading.clock = parse_game_clock(clock.text)
    reading.clock_text = clock.text
    if periods:
        period = min(periods, key=lambda p: _dist(p, clock))
        reading.period = parse_period(period.text)
        used.add(id(period))
    shots = [t for t in tokens if SHOT.match(t.text) and int(t.text[1:]) <= 24]
    if shots:
        shot = min(shots, key=lambda s: _dist(s, clock))
        reading.shot_clock = float(shot.text[1:])
        used.add(id(shot))
    numbers = [t for t in tokens if INTEGER.match(t.text) and id(t) not in used and int(t.text) < 200]
    reading.numbers = [(int(t.text), t.cx, t.cy, t.h) for t in numbers]
    pair = None
    if len(numbers) >= 2:
        candidates = []
        for i, a in enumerate(numbers):
            for b in numbers[i + 1:]:
                first, second = reading_order(a, b)
                size = min(a.h, b.h)
                candidates.append((first, second, size))
        allowed = []
        if scores_for is not None and reading.period is not None:
            allowed.append(scores_for(reading.period, reading.clock))
        if valid_scores is not None:
            allowed.append(valid_scores)
        for states in allowed:
            fits = [(f, s, z) for f, s, z in candidates
                    if (int(f.text), int(s.text)) in states or (int(s.text), int(f.text)) in states]
            if fits:
                f, s, _ = max(fits, key=lambda c: c[2])
                pair = (f, s) if (int(f.text), int(s.text)) in states else (s, f)
                break
        if pair is None:
            # Scores are the biggest digits in a bug; break ties by nearness to the clock.
            f, s, _ = max(candidates, key=lambda c: (round(c[2] / max(clock.h, 1), 1),
                                                     -(_dist(c[0], clock) + _dist(c[1], clock))))
            pair = (f, s)
    if pair:
        reading.away, reading.home = int(pair[0].text), int(pair[1].text)
        used.update({id(pair[0]), id(pair[1])})
        if reading.shot_clock is None:
            small = [t for t in numbers if id(t) not in used and int(t.text) <= 24
                     and abs(t.cy - clock.cy) < 2 * clock.h]
            if small:
                reading.shot_clock = float(min(small, key=lambda t: _dist(t, clock)).text)
    reading.conf = "high" if reading.complete() else "low"
    return reading


# ---------------------------------------------------------------- play-by-play checks (pure)

def score_states(all_plays) -> list[tuple[int, float, int, int]]:
    """(period, clock_s, away, home) after every scoring change, starting from 0-0."""
    states = [(1, 720., 0, 0)]
    for p in all_plays:
        if (p.away_score, p.home_score) != states[-1][2:]:
            states.append((p.period, p.clock_s, p.away_score, p.home_score))
    return states


def scores_near(states, period: int, clock: float, before: float = 20., after: float = 6.) -> set:
    """Scores ESPN shows from `before` s of game clock earlier to `after` s later (the bug
    can trail the official score by a few seconds)."""
    current = (0, 0)
    found = set()
    for p, c, a, h in states:
        if p < period or (p == period and c > clock + before):
            current = (a, h)
        elif p == period and c >= clock - after:
            found.add((a, h))
    found.add(current)
    return found


# ---------------------------------------------------------------- OCR (Apple Vision)

def _vision_ocr(images):
    import Quartz
    import Vision
    from Foundation import NSData
    import cv2
    results = []
    for img in images:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        h, w = rgba.shape[:2]
        data = NSData.dataWithBytes_length_(rgba.tobytes(), rgba.nbytes)
        provider = Quartz.CGDataProviderCreateWithCFData(data)
        cg = Quartz.CGImageCreate(w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
                                  Quartz.kCGImageAlphaNoneSkipLast, provider, None, False,
                                  Quartz.kCGRenderingIntentDefault)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        handler.performRequests_error_([request], None)
        found = []
        for obs in request.results() or []:
            candidate = obs.topCandidates_(1)[0]
            b = obs.boundingBox()  # normalized, bottom-left origin
            found.append((str(candidate.string()), round(float(candidate.confidence()), 2),
                          (b.origin.x * w, (1 - b.origin.y - b.size.height) * h, b.size.width * w, b.size.height * h)))
        results.append(found)
    return results


def _ocr_frames(job):
    """Worker: decode the given frames in order and OCR each (crop or whole frame)."""
    import cv2
    path, frames, region = job
    cap = cv2.VideoCapture(path)
    out = {}
    position = None
    for frame in frames:
        if position is None or frame < position or frame - position > 240:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            position = frame
        while position < frame:
            cap.grab()
            position += 1
        ok, image = cap.read()
        position += 1
        if not ok:
            out[frame] = []
            continue
        x1, y1, x2, y2 = region or (0, 0, image.shape[1], image.shape[0])
        found = _vision_ocr([image[y1:y2, x1:x2]])[0]
        out[frame] = [(t, c, (round(x + x1, 1), round(y + y1, 1), round(w, 1), round(h, 1)))
                      for t, c, (x, y, w, h) in found]
    return out


def ocr_parallel(path: str, frames: list[int], region, workers: int) -> dict[int, list]:
    if not frames:
        return {}
    frames = sorted(frames)
    workers = max(1, min(workers, len(frames)))
    size = math.ceil(len(frames) / workers)
    jobs = [(path, frames[i:i + size], region) for i in range(0, len(frames), size)]
    out = {}
    with ProcessPoolExecutor(len(jobs)) as pool:
        for part in pool.map(_ocr_frames, jobs):
            out.update(part)
    return out


def locate(observations: dict[int, list], width: int, height: int):
    """The bug's box from whole-frame OCR: around game clocks that sit next to a period token."""
    anchors, boxes = [], []
    for frame, obs in observations.items():
        tokens = tokens_of(obs)
        periods = [t for t in tokens if parse_period(t.text) is not None]
        clocks = [t for t in tokens if parse_game_clock(t.text) is not None]
        for c in clocks:
            near = [p for p in periods if _dist(p, c) < 8 * max(c.h, 10)]
            if near:
                anchors.append((frame, c))
    if not anchors:
        return None
    cx = sorted(c.cx for _, c in anchors)[len(anchors) // 2]
    cy = sorted(c.cy for _, c in anchors)[len(anchors) // 2]
    h = sorted(c.h for _, c in anchors)[len(anchors) // 2]
    near_frames = {f for f, c in anchors if abs(c.cx - cx) < 4 * h and abs(c.cy - cy) < 2 * h}
    for frame in near_frames:
        for t in tokens_of(observations[frame]):
            if abs(t.cx - cx) < .2 * width and abs(t.cy - cy) < max(5 * h, .08 * height) and t.h < 3 * h:
                boxes.append(t.box)
    x1 = min(b[0] for b in boxes) - 2 * h
    y1 = min(b[1] for b in boxes) - 1.5 * h
    x2 = max(b[0] + b[2] for b in boxes) + 2 * h
    y2 = max(b[1] + b[3] for b in boxes) + 1.5 * h
    return (int(max(0, x1)), int(max(0, y1)), int(min(width, x2)), int(min(height, y2)))


# ---------------------------------------------------------------- timeline

def video_info(path: Path) -> tuple[float, int, int, int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, count, width, height


def smooth(samples: list[dict], step: float) -> None:
    """Fill a missing period from neighbours on the same running clock, and drop isolated
    period misreads."""
    for i, s in enumerate(samples):
        if s["clock"] is None:
            continue
        neighbours = [samples[j] for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(samples)
                      and samples[j]["clock"] is not None and samples[j]["period"] is not None
                      and abs(samples[j]["clock"] - s["clock"]) <= 3 * step + 1]
        periods = [n["period"] for n in neighbours]
        if periods and (s["period"] is None or (s["period"] not in periods and len(set(periods)) == 1
                                                 and len(periods) >= 2)):
            s["period"] = max(set(periods), key=periods.count)
            s["conf"] = "low"


def segments(samples: list[dict], step: float) -> list[dict]:
    """Runs of samples showing one period with a continuous clock (a highlight package
    cuts between plays, which makes the clock jump)."""
    runs, current = [], []
    for s in samples:
        if s["clock"] is None or s["period"] is None:
            if current:
                runs.append(current)
            current = []
            continue
        if current:
            last = current[-1]
            gap = s["t"] - last["t"]
            drop = last["clock"] - s["clock"]
            if (s["period"] != last["period"] or gap > 2.5 * step or drop < -1.0
                    or drop > gap + 1.5):
                runs.append(current)
                current = []
        current.append(s)
    if current:
        runs.append(current)
    out = []
    for run in runs:
        scores = [(s["away"], s["home"]) for s in run if s["away"] is not None]
        out.append({"start": run[0]["t"], "end": run[-1]["t"], "period": run[0]["period"],
                    "clock_start": run[0]["clock"], "clock_end": run[-1]["clock"],
                    "score_start": scores[0] if scores else None, "score_end": scores[-1] if scores else None,
                    "samples": len(run), "high": sum(s["conf"] == "high" for s in run)})
    return out


def build(args) -> int:
    started = time.time()
    video = args.video.resolve()
    fps, count, width, height = video_info(video)
    if count <= 0:
        raise SystemExit(f"cannot read {video}")
    duration = count / fps
    out = args.out or OUT_DIR / f"{video.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out.with_suffix(".ocr.json")
    stat = video.stat()
    key = {"path": str(video), "size": stat.st_size}
    cache = {}
    if cache_path.exists() and not args.fresh:
        cache = json.loads(cache_path.read_text())
        if cache.get("video") != key:
            cache = {}
    region = tuple(args.region) if args.region else (None if args.full else cache.get("region"))
    timings = {}
    if region is None and not args.full:
        t0 = time.time()
        probe = [int(count * (0.03 + 0.94 * k / 23)) for k in range(24)]
        whole = ocr_parallel(str(video), probe, None, args.workers)
        region = locate(whole, width, height)
        timings["locate"] = time.time() - t0
        if region is None:
            print("WARNING: no scorebug found in 24 probe frames; reading whole frames (slow)", file=sys.stderr)
        else:
            print(f"scorebug located at x {region[0]}-{region[2]}, y {region[1]}-{region[3]} "
                  f"({timings['locate']:.1f} s)")
    if cache.get("region") != (list(region) if region else None):
        cache = {"video": key, "region": list(region) if region else None, "frames": {}}
    frames_cache = cache.setdefault("frames", {})
    wanted = []
    t = 0.
    while t < duration - 0.05:
        wanted.append(int(round(t * fps)))
        t += args.step
    todo = [f for f in wanted if str(f) not in frames_cache]
    t0 = time.time()
    if todo:
        print(f"reading {len(todo)} frames ({len(wanted) - len(todo)} cached) with {args.workers} workers...")
        found = ocr_parallel(str(video), todo, region, args.workers)
        for frame, obs in found.items():
            frames_cache[str(frame)] = obs
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        os.replace(tmp, cache_path)
    timings["ocr"] = time.time() - t0
    valid = states = scores_for = None
    game = None
    if args.event:
        summary = pbp.fetch(args.event)
        game = pbp.game_info(summary, args.event)
        all_plays = pbp.plays(summary)
        states = score_states(all_plays)
        valid = {(a, h) for _, _, a, h in states}

        def scores_for(period, clock):
            return scores_near(states, period, clock)
    samples = []
    for frame in wanted:
        obs = frames_cache.get(str(frame), [])
        r = parse_reading(obs, valid, scores_for)
        if r.clock is not None and states is not None and r.complete():
            near = scores_near(states, r.period, r.clock)
            if (r.away, r.home) not in near:
                r.conf = "low"
        samples.append({"t": round(frame / fps, 3), "frame": frame, "period": r.period, "clock": r.clock,
                        "clock_text": r.clock_text, "away": r.away, "home": r.home, "shot_clock": r.shot_clock,
                        "conf": r.conf, "text": " | ".join(o[0] for o in obs)[:160]})
    smooth(samples, args.step)
    segs = segments(samples, args.step)
    result = {"video": str(video), "fps": fps, "frames": count, "duration": round(duration, 3), "step": args.step,
              "region": list(region) if region else None, "event": args.event,
              "game": game.tag if game else None, "segments": segs, "samples": samples}
    out.write_text(json.dumps(result, indent=0))
    conf = [s["conf"] for s in samples]
    print(f"{len(samples)} samples: {conf.count('high')} high, {conf.count('low')} low, {conf.count('none')} no bug; "
          f"{len(segs)} segments")
    for seg in segs:
        score = "-".join(map(str, seg["score_start"])) if seg["score_start"] else "?"
        print(f"  {seg['start']:7.1f}-{seg['end']:7.1f} s  Q{seg['period']} {pbp.format_clock(seg['clock_start']):>5}"
              f" -> {pbp.format_clock(seg['clock_end']):>5}  score {score:>8}  ({seg['high']}/{seg['samples']} high)")
    print(f"timeline: {out}  (ocr cache {cache_path.name}); locate {timings.get('locate', 0):.1f} s, "
          f"ocr {timings['ocr']:.1f} s, total {time.time() - started:.1f} s")
    return 0


# ---------------------------------------------------------------- matching

def clock_time(seg_samples: list[dict], clock: float) -> tuple[float | None, str]:
    """Video time when the bug clock first reads `clock` within one segment's samples.
    Interpolates on a running clock; returns (time, how)."""
    prev = None
    for s in seg_samples:
        if s["clock"] is None:
            continue
        if s["clock"] <= clock:
            if prev is None:
                return s["t"], "segment starts at or below the play clock"
            if prev["clock"] - s["clock"] > 0:
                frac = (prev["clock"] - clock) / (prev["clock"] - s["clock"])
                return round(prev["t"] + frac * (s["t"] - prev["t"]), 2), "interpolated"
            return s["t"], "clock stopped"
        prev = s
    return None, "clock never reaches the play clock in this segment"


def segment_cuts(video: str, segs: list[dict], cache_path: Path, workers: int = 6) -> dict[str, list[float]]:
    """Camera cuts (seconds) in and just around each segment, from clips.py's detector on
    ffmpeg-decoded small frames; cached next to the timeline."""
    import clips
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get("video") != video:
        cache = {"video": video, "segments": {}}
    todo = [g for g in segs if f"{g['start']}-{g['end']}" not in cache["segments"]]

    def scan(seg):
        start = max(0., seg["start"] - 1.2)
        fps, first, frames = clips.small_frames(Path(video), start, seg["end"] - start + 1.4)
        rows = clips.analyze_cuts(frames, fps, first)
        return f"{seg['start']}-{seg['end']}", [r["time"] for r in rows if clips.is_cut(r)]

    if todo:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool:
            for key, found in pool.map(scan, todo):
                cache["segments"][key] = found
        cache_path.write_text(json.dumps(cache))
    return cache["segments"]


def safe_start(t: float, seg: dict, cuts: list[float], length: float = 7.0, lead: float = 4.0):
    """A clip start that keeps [start, start + length) clear of cuts with the shot about
    `lead` s in. Returns (start, cut-free span start, span end)."""
    before = [c for c in cuts if c <= t]
    after = [c for c in cuts if c > t]
    lo = before[-1] + 0.05 if before else seg["start"] - 0.5
    hi = after[0] - 0.05 if after else seg["end"] + 0.5
    start = max(lo, t - lead)
    if start + length > hi:
        start = max(lo, hi - length)
    return round(start, 2), lo, hi


def match(args) -> int:
    timeline = json.loads(Path(args.timeline).read_text())
    event = str(args.pbp)
    if event.endswith(".json"):
        event = Path(event).stem
    summary = pbp.fetch(event)
    game = pbp.game_info(summary, event)
    all_plays = pbp.plays(summary)
    samples = timeline["samples"]
    segs = timeline["segments"]
    shots = [p for p in all_plays if p.field_goal or (args.free_throws and p.free_throw)]
    covered = []
    cut_map = None
    if not args.no_cuts and Path(timeline["video"]).exists():
        wanted = []
        for p in shots:
            for seg in segs:
                if (seg["period"] == p.period and seg["clock_end"] - args.slack <= p.clock_s
                        <= seg["clock_start"] + 1.0 and seg not in wanted):
                    wanted.append(seg)
        t0 = time.time()
        cut_map = segment_cuts(timeline["video"], wanted, Path(args.timeline).with_suffix(".cuts.json"))
        print(f"checked {len(wanted)} segments for camera cuts in {time.time() - t0:.1f} s")
    elif not args.no_cuts:
        print(f"source video {timeline['video']} is gone: cut-free spans not checked")
    print(f"{game.tag} ESPN event {event}; video {Path(timeline['video']).name} "
          f"({timeline['duration']:.0f} s, {len(segs)} bug segments)")
    print("time = video seconds when the bug clock reads the play clock; ESPN's clock is often 1-4 s off the "
          "release, so find the shot in the frames. clip start = a 7 s clip with no camera cut and the shot "
          "~4 s in. Check every cut clip with clips.py cuts anyway.")
    for p in shots:
        hits = []
        for seg in segs:
            if seg["period"] != p.period:
                continue
            if not (seg["clock_end"] - args.slack <= p.clock_s <= seg["clock_start"] + 1.0):
                continue
            seg_samples = [s for s in samples if seg["start"] <= s["t"] <= seg["end"]]
            t, how = clock_time(seg_samples, p.clock_s)
            if t is None:
                t, how = seg["end"], "play clock just below the segment's last reading"
            before = (p.away_score - (p.points or 0) * (p.made and p.team == game.away),
                      p.home_score - (p.points or 0) * (p.made and p.team == game.home))
            scores = {(s["away"], s["home"]) for s in seg_samples if s["away"] is not None}
            ok = bool(scores & {before, (p.away_score, p.home_score)})
            hits.append([t, how, seg, ok])
        result = "made" if p.made else "missed"
        dist = "" if p.distance is None else f"{p.distance:g}{'*' if p.distance_from_xy else ''} ft"
        head = (f"#{p.index:<4} Q{p.period} {p.clock:>5} {p.team:<3} {result:<6} {dist:>6} {p.zone or '':<17} "
                f"{p.text[:80]}")
        if not hits:
            if not args.covered_only:
                print(f"{head}\n        no footage")
            continue
        covered.append(p)
        print(head)
        for t, how, seg, ok in hits:
            cuts = cut_map.get(f"{seg['start']}-{seg['end']}") if cut_map is not None else None
            if cuts is None:
                start, lo, hi = safe_start(t, seg, [])
                span = f"segment {seg['start']:.1f}-{seg['end']:.1f} s (cuts not checked)"
            else:
                start, lo, hi = safe_start(t, seg, cuts)
                inside = [c for c in cuts if seg["start"] - 1.2 <= c <= seg["end"] + 1.2]
                span = (f"no cut {lo:.2f}-{hi:.2f} s (segment {seg['start']:.1f}-{seg['end']:.1f} s, cuts "
                        f"{', '.join(f'{c:.2f}' for c in inside) or 'none found'})")
            room = hi - start
            warn = "" if room >= 6.99 else f"  WARNING: only {room:.1f} s without a cut; use a shorter clip or skip"
            print(f"        time {t:7.2f} s ({how}); {span}; bug Q{seg['period']} "
                  f"{pbp.format_clock(seg['clock_start'])}->{pbp.format_clock(seg['clock_end'])}, score "
                  f"{'matches' if ok else 'DOES NOT match'} pbp; clip start {start:.2f}{warn}")
    fg = [p for p in all_plays if p.field_goal]
    got = [p for p in covered if p.field_goal]
    misses = [p for p in got if not p.made]
    rim = [p for p in got if p.distance is not None and p.distance <= 4]
    print(f"\n{len(got)} of {len(fg)} field goal attempts have footage: {len(misses)} misses, "
          f"{len(got) - len(misses)} makes, {len(rim)} within 4 ft. Segments are 1-step accurate: "
          f"run clips.py cuts on every clip.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "match":
        parser = argparse.ArgumentParser(prog="scorebug.py match", description="match ESPN plays to video times")
        parser.add_argument("timeline", type=Path)
        parser.add_argument("pbp", help="ESPN event id (or its cached eval/staging/pbp/<id>.json)")
        parser.add_argument("--covered-only", action="store_true", help="only plays with footage")
        parser.add_argument("--free-throws", action="store_true", help="include free throws")
        parser.add_argument("--no-cuts", action="store_true", help="skip the camera-cut scan of the source")
        parser.add_argument("--slack", type=float, default=2.0,
                            help="seconds the play clock may sit below a segment's last reading (default 2)")
        return match(parser.parse_args(argv[1:]))
    if argv and argv[0] == "build":
        argv = argv[1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--event", help="ESPN event id: check scores against the play-by-play (recommended)")
    parser.add_argument("--step", type=float, default=1.0, help="seconds between samples (default 1.0)")
    parser.add_argument("--workers", type=int, default=6, help="parallel OCR processes (default 6)")
    parser.add_argument("--out", type=Path, help="timeline path (default eval/staging/scorebug/<video>.json)")
    parser.add_argument("--region", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="read only this box instead of locating the bug")
    parser.add_argument("--full", action="store_true", help="read whole frames (slow; for odd layouts)")
    parser.add_argument("--fresh", action="store_true", help="ignore the OCR cache")
    return build(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
