#!/usr/bin/env python3
"""ESPN play-by-play for one NBA game: every shot with its distance and zone.

    uv run --inexact python scripts/labeling/pbp.py 401810953             # shot table
    uv run --inexact python scripts/labeling/pbp.py 401810953 --play 57   # details + note prefix
    uv run --inexact python scripts/labeling/pbp.py --fit 401810953 401810217 ...

The summary JSON (site.api.espn.com .../nba/summary?event=<id>) is fetched once and
cached in eval/staging/pbp/<id>.json.

ESPN coordinates, worked out with --fit over 7139 field goals in 54 games whose
text states "N-foot": whole feet on the half court of the basket shot at, x across
the court 0-50 with the rim at x=25, y toward half court with the rim at y=1 (the
backboard is about y=0 and the baseline y=-4.25). Distance to (25, 1) matches the
stated feet with mean error 0.38 ft and no bias (rim at y=0 is 0.67 ft and +0.66 ft
too long); e.g. a 5-foot layup at x=25 y=6 and a 26-foot three at x=45 y=17. For
zones, (x - 25, y - 1 + 5.25) is the point in app/court.py's frame. Free throws carry junk coordinates
(about -2.1e8), which are dropped. Big values like x=-134 y=211 in old label notes
are NBA-stats coordinates (tenths of a foot from the rim), not ESPN ones.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import court as courtlib  # noqa: E402

STAGING = ROOT / "eval" / "staging"
PBP_DIR = STAGING / "pbp"
JOBS_DIR = ROOT / "data" / "jobs"
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"

# ESPN abbreviations that differ from the NBA ones the eval CSVs use.
ESPN_TO_NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS",
               "PHO": "PHX", "BRK": "BKN", "NJ": "BKN", "CHO": "CHA"}

# The rim in ESPN coordinates (feet). See --fit.
COORD_X0 = 25.0
COORD_Y0 = 1.0

NBA = courtlib.template("nba")
SHOT_WORDS = re.compile(r"\b(layup|dunk|tip|putback|alley oop|finger roll)\b", re.I)
CONTEXT_TYPES = re.compile(r"rebound|foul|block|goaltend|violation|jump ?ball", re.I)
DISTANCE = re.compile(r"(\d+)-(?:foot|ft)\b", re.I)
CLOCK = re.compile(r"^(?:(\d{1,2}):(\d{2})(?:\.(\d))?|:?(\d{1,2})(?:\.(\d{1,2}))?)$")


def parse_clock(text: str) -> float | None:
    """Game clock text to seconds left in the period: '9:59' -> 599, '58.4' -> 58.4, '0.2' -> 0.2,
    '12:00' -> 720, ':07' -> 7. None if it is not a clock."""
    text = (text or "").strip()
    found = CLOCK.match(text)
    if not found:
        return None
    minutes, seconds, tenth, under, frac = found.groups()
    if minutes is not None:
        value = int(minutes) * 60 + int(seconds)
        if int(seconds) > 59 or value > 720:
            return None
        return value + (int(tenth) / 10 if tenth else 0.)
    value = float(under + ("." + frac if frac else ""))
    return value if value < 60 else None


def format_clock(seconds: float) -> str:
    if seconds >= 60:
        return f"{int(seconds // 60)}:{int(seconds % 60):02d}"
    return f"{seconds:.1f}"


def parse_distance(text: str) -> int | None:
    """The stated distance in a play text ('26-foot three point jumper' -> 26), or None."""
    found = DISTANCE.search(text or "")
    return int(found.group(1)) if found else None


def valid_xy(x, y) -> bool:
    return x is not None and y is not None and -10 <= x <= 60 and -10 <= y <= 100


def court_point(x: float, y: float) -> tuple[float, float]:
    """ESPN (x, y) as a point in app/court.py's half-court frame (x from the lane's centre
    line, y from the baseline, rim at (0, 5.25))."""
    return x - COORD_X0, y - COORD_Y0 + NBA.dims["rim_y"]


def xy_distance(x: float, y: float) -> float:
    return math.hypot(x - COORD_X0, y - COORD_Y0)


def zone_of(x: float, y: float, points: int | None = None) -> tuple[str | None, bool]:
    """Shot zone from ESPN coordinates with court.zone() on the NBA template.

    Returns (zone, overridden). Coordinates are whole feet, so a shot on the
    three-point line can land on the wrong side; when ESPN's points attempted (exact)
    disagree, the zone keeps its side of the arc from the 2-or-3 call instead and
    overridden is True."""
    point = court_point(x, y)
    zone = courtlib.zone(point, NBA)
    if zone is None or points not in (2, 3):
        return zone, False
    is_three = zone in ("corner_three", "above_break_three")
    if points == 3 and not is_three:
        return ("corner_three" if point[1] <= NBA.three_break_y else "above_break_three"), True
    if points == 2 and is_three:
        return "midrange", True
    return zone, False


@dataclass
class Play:
    index: int
    period: int
    clock: str
    clock_s: float
    away_score: int
    home_score: int
    team: str
    text: str
    type: str
    x: float | None
    y: float | None
    shooting: bool
    made: bool | None
    points: int | None
    distance: float | None = None
    distance_from_xy: bool = False
    zone: str | None = None
    zone_overridden: bool = False
    shooter: str = ""
    shooter_id: str = ""
    sequence: str = ""

    @property
    def free_throw(self) -> bool:
        return self.type.lower().startswith("free throw")

    @property
    def field_goal(self) -> bool:
        return self.shooting and not self.free_throw

    @property
    def score(self) -> str:
        return f"{self.away_score}-{self.home_score}"

    @property
    def xy(self) -> str:
        return f"x={self.x:g} y={self.y:g}" if self.x is not None else "no x/y"


@dataclass
class Game:
    event: str
    away: str
    home: str
    away_name: str
    home_name: str
    date: str
    start_utc: str
    network: str

    @property
    def tag(self) -> str:
        return f"{self.away}@{self.home} {self.date}"


def fetch(event: str, refresh: bool = False) -> dict:
    """The ESPN summary for an event, from the cache when present."""
    event = str(event).strip()
    if not event.isdigit():
        raise SystemExit(f"ESPN event ids are numbers, not {event!r}")
    path = PBP_DIR / f"{event}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(ESPN_URL.format(event), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read())
            break
        except Exception as exc:  # network hiccups: retry with backoff
            error = exc
            time.sleep(1.5 * (attempt + 1))
    else:
        raise SystemExit(f"could not fetch ESPN event {event}: {error}")
    if not data.get("plays"):
        raise SystemExit(f"ESPN event {event} has no play-by-play (wrong id, or not played yet)")
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=PBP_DIR, suffix=".tmp")
    with os.fdopen(handle, "w") as out:
        json.dump(data, out)
    os.replace(tmp, path)
    return data


def nba_abbr(espn: str) -> str:
    return ESPN_TO_NBA.get(espn, espn)


def game_info(summary: dict, event: str = "") -> Game:
    competition = summary["header"]["competitions"][0]
    teams = {c["homeAway"]: c["team"] for c in competition["competitors"]}
    start = competition.get("date", "")
    try:
        utc = datetime.strptime(start, "%Y-%m-%dT%H:%MZ")
    except ValueError:
        utc = datetime.strptime(start[:10], "%Y-%m-%d") + timedelta(hours=23)
    try:
        from zoneinfo import ZoneInfo
        from datetime import timezone
        local = utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("America/New_York"))
    except Exception:
        local = utc - timedelta(hours=5)
    networks = []
    for b in competition.get("broadcasts", []) or []:
        name = (b.get("media") or {}).get("shortName")
        if name and name not in networks:
            networks.append(name)
    return Game(event or summary["header"].get("id", ""), nba_abbr(teams["away"]["abbreviation"]),
                nba_abbr(teams["home"]["abbreviation"]), teams["away"]["name"], teams["home"]["name"],
                local.strftime("%Y-%m-%d"), start, ", ".join(networks))


def athletes(summary: dict) -> dict[str, tuple[str, str, str]]:
    found = {}
    for team in summary.get("boxscore", {}).get("players", []):
        abbr = nba_abbr(team["team"]["abbreviation"])
        for block in team.get("statistics", []):
            for entry in block.get("athletes", []):
                a = entry["athlete"]
                found[a["id"]] = (a.get("displayName", ""), a.get("jersey", ""), abbr)
    return found


def plays(summary: dict) -> list[Play]:
    competition = summary["header"]["competitions"][0]
    team_by_id = {c["team"]["id"]: nba_abbr(c["team"]["abbreviation"]) for c in competition["competitors"]}
    people = athletes(summary)
    output = []
    for index, raw in enumerate(summary.get("plays", [])):
        clock = raw.get("clock", {}).get("displayValue", "")
        coordinate = raw.get("coordinate") or {}
        x, y = coordinate.get("x"), coordinate.get("y")
        if not valid_xy(x, y):
            x = y = None
        shooting = bool(raw.get("shootingPlay"))
        points = raw.get("pointsAttempted") if shooting else None
        participants = raw.get("participants") or []
        shooter_id = participants[0]["athlete"]["id"] if participants and shooting else ""
        play = Play(index, raw.get("period", {}).get("number", 0), clock, parse_clock(clock) or 0.,
                    raw.get("awayScore", 0), raw.get("homeScore", 0),
                    team_by_id.get((raw.get("team") or {}).get("id"), ""),
                    " ".join(raw.get("text", "").split()), " ".join(raw.get("type", {}).get("text", "").split()),
                    x, y, shooting, bool(raw.get("scoringPlay")) if shooting else None, points,
                    shooter_id=shooter_id, sequence=str(raw.get("sequenceNumber", "")))
        if shooter_id in people:
            name, jersey, abbr = people[shooter_id]
            play.shooter = f"#{jersey} {abbr} ({name})" if jersey else f"{abbr} ({name})"
        if play.field_goal:
            play.distance = parse_distance(play.text)
            if play.distance is None and x is not None and SHOT_WORDS.search(play.text + " " + play.type):
                play.distance = round(xy_distance(x, y))
                play.distance_from_xy = True
            if x is not None:
                play.zone, play.zone_overridden = zone_of(x, y, points)
        output.append(play)
    return output


def with_context(all_plays: list[Play], seconds: float = 3.0, free_throws: bool = True) -> list[tuple[Play, bool]]:
    """Field goals (and free throws), plus the rebounds, fouls and blocks within `seconds` of
    game clock of a field goal in the same period. Returns (play, is_context)."""
    shots = [p for p in all_plays if p.field_goal]
    keyed = {}
    for p in shots:
        keyed.setdefault(p.period, []).append(p.clock_s)
    rows = []
    for p in all_plays:
        if p.field_goal or (p.free_throw and free_throws):
            rows.append((p, False))
        elif CONTEXT_TYPES.search(p.type) and any(abs(p.clock_s - c) <= seconds for c in keyed.get(p.period, [])):
            rows.append((p, True))
    return rows


def note_prefix(game: Game, play: Play) -> str:
    """The start of a label's notes for this play (README format)."""
    return (f"{game.tag} ESPN event {game.event} Q{play.period} {play.clock} '{play.text}' "
            f"({play.type}; {play.xy}), away-home {play.score}.")


def describe(game: Game, play: Play) -> str:
    lines = [f"play #{play.index}: Q{play.period} {play.clock} {play.team} {play.text}",
             f"  ESPN type: {play.type}; {play.xy}; points attempted: {play.points}; "
             f"{'made' if play.made else 'missed'}; score after: away-home {play.score}"]
    if play.field_goal:
        dist = "" if play.distance is None else f"{play.distance:g}"
        how = " (from coordinates: the text gives no feet)" if play.distance_from_xy else ""
        lines.append(f"  distance_ft: {dist or '(leave empty)'}{how}")
        lines.append(f"  zone: {play.zone or '(no coordinates)'}"
                     + (" (2-or-3 call overrides the coordinates)" if play.zone_overridden else ""))
        lines.append(f"  shooter: {play.shooter} (add the jersey colour from the frames)")
    lines.append(f"  notes start: {note_prefix(game, play)}")
    return "\n".join(lines)


def print_table(game: Game, rows: list[tuple[Play, bool]]) -> None:
    print(f"{game.tag}  ESPN event {game.event}  away {game.away} ({game.away_name}) at home "
          f"{game.home} ({game.home_name})  tip {game.start_utc} UTC  TV: {game.network}")
    print("dist: stated feet; * = from coordinates (layup/dunk text with no feet). zone ! = the 2-or-3 "
          "call overrode coordinates on the line. Context rows (rebounds/fouls) start with '.'.")
    print(f"{'#':>4} {'Q':>2} {'clock':>6} {'score':>8} {'team':<4} {'result':<6} {'dist':>4} "
          f"{'zone':<18} {'x':>3} {'y':>3}  {'ESPN type':<30} text")
    for p, context in rows:
        if context:
            print(f"{'.':>4} {p.period:>2} {p.clock:>6} {p.score:>8} {p.team:<4} {'':<6} {'':>4} {'':<18} "
                  f"{'':>3} {'':>3}  {p.type[:30]:<30} {p.text}")
            continue
        dist = "" if p.distance is None else f"{p.distance:g}" + ("*" if p.distance_from_xy else "")
        zone = (p.zone or "") + ("!" if p.zone_overridden else "")
        x = "" if p.x is None else f"{p.x:g}"
        y = "" if p.y is None else f"{p.y:g}"
        result = "made" if p.made else "missed"
        print(f"{p.index:>4} {p.period:>2} {p.clock:>6} {p.score:>8} {p.team:<4} {result:<6} {dist:>4} "
              f"{zone:<18} {x:>3} {y:>3}  {p.type[:30]:<30} {p.text}")


def fit(events: list[str]) -> None:
    """Fit the rim position (and a scale) in ESPN coordinates from the stated distances."""
    import numpy as np
    samples = []
    for event in events:
        for p in plays(fetch(event)):
            stated = parse_distance(p.text)
            if p.field_goal and stated is not None and p.x is not None:
                samples.append((p.x, p.y, stated, p.points))
    data = np.array([(x, y, d) for x, y, d, _ in samples], dtype=float)
    print(f"{len(data)} field goals with a stated distance and coordinates in {len(events)} games")
    best = None
    for x0 in np.arange(23, 27.01, .25):
        for y0 in np.arange(-6, 6.01, .25):
            err = np.hypot(data[:, 0] - x0, data[:, 1] - y0) - data[:, 2]
            score = float(np.mean(np.abs(err)))
            if best is None or score < best[0]:
                best = (score, x0, y0, float(np.mean(err)))
    print(f"best rim at x0={best[1]:.2f} y0={best[2]:.2f}: mean |coord dist - stated| {best[0]:.2f} ft, "
          f"bias {best[3]:+.2f} ft")
    for x0, y0 in ((25, 0), (25, 5.25), (25, -5.25)):
        err = np.hypot(data[:, 0] - x0, data[:, 1] - y0) - data[:, 2]
        within = float(np.mean(np.abs(err) <= 1.0))
        print(f"  rim at ({x0}, {y0}): mean |err| {np.mean(np.abs(err)):.2f} ft, bias {np.mean(err):+.2f}, "
              f"{within:.0%} within 1 ft")
    print(f"  x range {data[:, 0].min():g}..{data[:, 0].max():g}, y range {data[:, 1].min():g}..{data[:, 1].max():g}")
    threes = [(x, y) for x, y, d, pts in samples if pts == 3]
    twos = [(x, y) for x, y, d, pts in samples if pts == 2]
    wrong3 = sum(courtlib.zone(court_point(x, y), NBA) not in ("corner_three", "above_break_three") for x, y in threes)
    wrong2 = sum(courtlib.zone(court_point(x, y), NBA) in ("corner_three", "above_break_three") for x, y in twos)
    print(f"  2-or-3 from coordinates alone: {wrong3}/{len(threes)} threes and {wrong2}/{len(twos)} twos "
          f"land on the wrong side of the arc")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("event", nargs="*", help="ESPN event id (several with --fit)")
    parser.add_argument("--play", type=int, nargs="+", help="print details and the note prefix for these play numbers")
    parser.add_argument("--period", type=int, help="only this period")
    parser.add_argument("--no-free-throws", action="store_true", help="hide free throws")
    parser.add_argument("--refresh", action="store_true", help="refetch instead of using the cache")
    parser.add_argument("--fit", action="store_true", help="fit ESPN's coordinate convention over the given events")
    args = parser.parse_args(argv)
    if not args.event:
        parser.error("give an ESPN event id")
    if args.fit:
        fit(args.event)
        return 0
    event = args.event[0]
    summary = fetch(event, args.refresh)
    game = game_info(summary, event)
    all_plays = plays(summary)
    if args.play:
        by_index = {p.index: p for p in all_plays}
        for number in args.play:
            if number not in by_index:
                raise SystemExit(f"no play #{number}")
            print(describe(game, by_index[number]))
        return 0
    rows = with_context(all_plays, free_throws=not args.no_free_throws)
    if args.period:
        rows = [(p, c) for p, c in rows if p.period == args.period]
    print_table(game, rows)
    shots = [p for p in all_plays if p.field_goal]
    print(f"{len(shots)} field goal attempts ({sum(bool(p.made) for p in shots)} made), "
          f"{sum(p.free_throw for p in all_plays)} free throws. Game tag for notes: {game.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
