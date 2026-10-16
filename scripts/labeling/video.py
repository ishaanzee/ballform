#!/usr/bin/env python3
"""Find and download the NBA channel's full-game highlights upload of a game.

    uv run --inexact python scripts/labeling/video.py find 401810953
    uv run --inexact python scripts/labeling/video.py download fED_h7XPWGY eval/staging/video

find searches YouTube (yt-dlp ytsearch10, flat playlist) and keeps only uploads by the
official NBA channel whose title names both teams' nicknames in caps and the game's
date, e.g. "WIZARDS at LAKERS | FULL GAME HIGHLIGHTS | March 30, 2026" or
"EXTENDED: HEAT at ROCKETS ...". Fan re-uploads (other channels) are never picked.
The last line is "BEST <youtube id> ..." or "NONE ...".

download fetches 1080p60 H.264 video only (format 312, HLS; else the best avc1 at or
below 1080p) with 16 concurrent fragments and retries, to <dest_dir>/<id>.mp4.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pbp  # noqa: E402

NBA_CHANNEL_ID = "UCWJ2lWNubArHWmf3FIHbfcQ"
FIELDS = "%(id)s\t%(channel)s\t%(channel_id)s\t%(duration)s\t%(title)s"
FORMAT = "312/bv*[vcodec^=avc1][height<=1080][fps>30]/bv*[vcodec^=avc1][height<=1080]"


def search(query: str, count: int = 10) -> list[dict]:
    out = subprocess.run(["yt-dlp", "--flat-playlist", "--no-warnings", "--print", FIELDS,
                          f"ytsearch{count}:{query}"], capture_output=True, text=True, timeout=120)
    found = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        vid, channel, channel_id, duration, title = parts
        try:
            seconds = float(duration)
        except ValueError:
            seconds = 0.
        found.append({"id": vid, "channel": channel, "channel_id": channel_id, "duration": seconds, "title": title})
    return found


def date_words(date: str) -> list[str]:
    day = datetime.strptime(date, "%Y-%m-%d")
    return [f"{day:%B} {day.day}, {day.year}", f"{day:%b} {day.day}, {day.year}", f"{day:%B} {day.day} {day.year}"]


def upload_date(video_id: str) -> str:
    out = subprocess.run(["yt-dlp", "--no-warnings", "--skip-download", "--print", "%(upload_date)s",
                          f"https://www.youtube.com/watch?v={video_id}"], capture_output=True, text=True, timeout=120)
    return out.stdout.strip()


def judge(video: dict, game: pbp.Game) -> tuple[bool, str]:
    title = video["title"]
    if video["channel_id"] != NBA_CHANNEL_ID and video["channel"] != "NBA":
        return False, f"channel {video['channel']!r} is not NBA"
    if video["channel_id"] and video["channel_id"] != NBA_CHANNEL_ID:
        return False, "a channel named NBA that is not the official one"
    for name in (game.away_name, game.home_name):
        if name.upper() not in title:
            return False, f"title lacks {name.upper()}"
    if not re.search(r"EXTENDED|FULL GAME HIGHLIGHTS", title, re.I):
        return False, "not an EXTENDED / FULL GAME HIGHLIGHTS upload"
    if any(words.lower() in title.lower() for words in date_words(game.date)):
        return True, "date in title"
    if re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                 r"\s+\d{1,2},?\s+\d{4}", title):
        return False, "title names another date"
    uploaded = upload_date(video["id"])
    if uploaded:
        day = datetime.strptime(game.date, "%Y-%m-%d")
        ok = [(day + timedelta(days=k)).strftime("%Y%m%d") for k in range(0, 3)]
        if uploaded in ok:
            return True, f"no date in title; uploaded {uploaded}"
        return False, f"no date in title; uploaded {uploaded}, not within 2 days of the game"
    return False, "no date in title and upload date unknown"


def find(event: str) -> int:
    game = pbp.game_info(pbp.fetch(event), event)
    away, home = game.away_name.upper(), game.home_name.upper()
    words = date_words(game.date)[0]
    queries = [f"{away} at {home} FULL GAME HIGHLIGHTS {words}", f"EXTENDED {away} at {home} {words}",
               f"NBA {away} at {home} highlights {words}"]
    print(f"{game.tag}: searching YouTube for the NBA channel upload ({away} at {home}, {words})")
    with ThreadPoolExecutor(len(queries)) as pool:
        results = list(pool.map(search, queries))
    seen, candidates = set(), []
    for batch in results:
        for video in batch:
            if video["id"] not in seen:
                seen.add(video["id"])
                candidates.append(video)
    good = []
    for video in candidates:
        ok, why = judge(video, game)
        flag = "OK  " if ok else "skip"
        print(f"  {flag} {video['id']}  {video['duration'] / 60:5.1f} min  {video['channel'][:22]:<22}  "
              f"{video['title'][:90]}  ({why})")
        if ok:
            good.append(video)
    if not good:
        print(f"NONE: no NBA-channel full-game upload found for {game.tag}; pick another game")
        return 1
    best = max(good, key=lambda v: (v["duration"], "EXTENDED" in v["title"].upper()))
    print(f"BEST {best['id']}  {best['duration'] / 60:.1f} min  {best['title']}")
    return 0


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True)
    data = json.loads(out.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    return {"codec": stream.get("codec_name"), "width": stream.get("width"), "height": stream.get("height"),
            "fps": stream.get("r_frame_rate"), "duration": float((data.get("format") or {}).get("duration", 0))}


def download(video_id: str, dest: Path, sections: str | None = None) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{video_id}.mp4"
    if target.exists() and target.stat().st_size > 0 and not sections:
        print(f"already downloaded: {target}  {probe(target)}")
        return 0
    cmd = ["yt-dlp", "--no-warnings", "--no-playlist", "-f", FORMAT, "-N", "16", "--retries", "15",
           "--fragment-retries", "30", "--retry-sleep", "fragment:exp=1:10", "--socket-timeout", "20",
           "--no-progress", "--no-part", "-o", str(dest / f"{video_id}.%(ext)s"), "--remux-video", "mp4"]
    if sections:
        cmd += ["--download-sections", sections]
    cmd.append(f"https://www.youtube.com/watch?v={video_id}")
    started = time.time()
    for attempt in range(3):
        result = subprocess.run(cmd)
        if result.returncode == 0 and target.exists():
            break
        print(f"yt-dlp failed (attempt {attempt + 1}); retrying", file=sys.stderr)
        time.sleep(3)
    else:
        print(f"download failed: {video_id}", file=sys.stderr)
        return 1
    info = probe(target)
    print(f"downloaded {target} in {time.time() - started:.0f} s: {info['codec']} {info['width']}x{info['height']} "
          f"{info['fps']} fps, {info['duration']:.1f} s, {target.stat().st_size / 1e6:.0f} MB")
    if info["codec"] != "h264" or info["height"] != 1080:
        print("WARNING: not 1080p H.264; clips.py will scale it to 1920x1080 59.94 fps anyway", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("find", help="find the NBA channel upload of an ESPN event")
    f.add_argument("event")
    d = sub.add_parser("download", help="download 1080p60 H.264 video only")
    d.add_argument("video_id")
    d.add_argument("dest_dir", type=Path)
    d.add_argument("--sections", help="only these parts, yt-dlp --download-sections syntax, e.g. '*300-420'")
    args = parser.parse_args(argv)
    if args.command == "find":
        return find(args.event)
    return download(args.video_id, args.dest_dir, args.sections)


if __name__ == "__main__":
    sys.exit(main())
