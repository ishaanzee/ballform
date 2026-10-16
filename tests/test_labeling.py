import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "labeling"))

import pbp  # noqa: E402
import rows  # noqa: E402
import scorebug  # noqa: E402


def test_zone_from_espn_coordinates():
    # The rim is (25, 1) in ESPN coordinates; zones use app/court.py's NBA template.
    assert pbp.zone_of(25, 6) == ("paint", False)            # 5-foot layup
    assert pbp.zone_of(45, 17) == ("above_break_three", False)  # 26-foot three
    assert pbp.zone_of(2, 7) == ("corner_three", False)       # 24-foot corner three
    assert pbp.zone_of(25, 21, 2) == ("midrange", False)      # 20 ft straight on
    assert pbp.zone_of(17, 14)[0] == "paint"                  # elbow, on the lane line
    assert pbp.zone_of(35, 10)[0] == "midrange"               # outside the lane, short of the line
    # On the arc the exact 2-or-3 call wins over whole-foot coordinates.
    assert pbp.zone_of(25, 24, 3) == ("above_break_three", True)
    assert pbp.zone_of(25, 25, 2) == ("midrange", True)
    assert round(pbp.xy_distance(25, 6)) == 5 and round(pbp.xy_distance(45, 17)) == 26


def test_clock_parsing():
    assert pbp.parse_clock("9:59") == 599
    assert pbp.parse_clock("12:00") == 720
    assert pbp.parse_clock("58.4") == pytest.approx(58.4)
    assert pbp.parse_clock("0.2") == pytest.approx(0.2)
    assert pbp.parse_clock("12:61") is None and pbp.parse_clock("abc") is None
    assert scorebug.parse_game_clock("9:59") == 599
    assert scorebug.parse_game_clock("58.4") == pytest.approx(58.4)
    assert scorebug.parse_game_clock(":17") is None  # a shot clock
    assert [scorebug.parse_period(t) for t in ("1ST", "IST", "ЗRD", "4TH", "Q4", "OT", "2OT", "4тн")] == \
        [1, 1, 3, 4, 4, 5, 6, 4]
    assert scorebug.parse_period("BONUS") is None


def test_distance_parsing():
    assert pbp.parse_distance("Austin Reaves misses 26-foot three point jumper") == 26
    assert pbp.parse_distance("Jake LaRavia makes 2-foot running dunk (LeBron James assists)") == 2
    assert pbp.parse_distance("LeBron James makes driving layup") is None


def box(x, y, w=40, h=26):
    return (x, y, w, h)


def test_scorebug_reading():
    # A local-RSN bug as Vision reads it: logo text glued to the home score, colon shot clock.
    obs = [("3RD", 1.0, box(1446, 940)), ("50", 1.0, box(1534, 878)), ("TKERS 72", 0.5, box(1601, 876, 165)),
           ("8:17", 1.0, box(1541, 942, 75)), (":17", 1.0, box(1682, 940))]
    r = scorebug.parse_reading(obs)
    assert (r.period, r.clock, r.away, r.home, r.shot_clock, r.conf) == (3, 497, 50, 72, 17, "high")
    assert scorebug.parse_reading([("CRYPTO.COM", 1.0, box(0, 0))]).conf == "none"
    # One line holding both scores and the period.
    r = scorebug.parse_reading([("83 DEN 94 4TH", 1.0, box(1200, 950, 300)), ("9:59", 1.0, box(1520, 950, 70))])
    assert (r.period, r.clock, r.away, r.home) == (4, 599, 83, 94)


def test_notes_must_start_with_the_game():
    assert rows.notes_problem("WAS@LAL 2026-03-30 ESPN event 401810953 Q2 ...") is None
    assert rows.notes_problem("FIXED youtube:dQw4w9WgXcQ pickup run ...") is None
    assert rows.notes_problem("WSH@LAL 2026-03-30 ...")  # ESPN abbreviation, not the NBA one
    assert rows.notes_problem("Same clip. Q2 10:29 ...")


def test_row_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(rows, "JOBS_DIR", tmp_path / "jobs")
    job = tmp_path / "jobs" / "0a1b2c3d4e5f"
    job.mkdir(parents=True)
    (job / "input.mp4").write_bytes(b"")
    (job / "settings.json").write_text("{}")
    good = {"job_id": "0a1b2c3d4e5f", "release_s": "3.45", "outcome": "missed", "shot_type": "jump shot",
            "distance_ft": "26", "zone": "above_break_three", "shooter": "#15 LAL white", "source": "espn_pbp",
            "notes": "WAS@LAL 2026-03-30 ESPN event 401810953 Q1 6:45 'x, y' (Jump Shot; x=37 y=24)."}
    assert rows.row_problems(good) == []
    assert rows.row_problems(dict(good, job_id="ffffffffffff"))  # no such job
    assert rows.row_problems(dict(good, outcome="blocked"))
    assert rows.row_problems(dict(good, source=""))  # distance needs its source
    no_shot = {k: "" for k in rows.HEADER} | {"job_id": "0a1b2c3d4e5f", "notes": good["notes"]}
    assert rows.row_problems(no_shot) == []
    assert rows.row_problems(dict(no_shot, outcome="made"))
    path = tmp_path / "rows.csv"
    rows.write_rows(path, [good, no_shot])
    assert len(rows.check(path, quiet=True)) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text(",".join(rows.HEADER) + "\n0a1b2c3d4e5f,3.4,made,,,,,,WAS@LAL 2026-03-30 a, b\n")
    with pytest.raises(SystemExit, match="10 columns"):
        rows.check(bad, quiet=True)
