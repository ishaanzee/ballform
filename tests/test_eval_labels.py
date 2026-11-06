"""Integrity checks on the dev label files. Holdout files are never opened here."""
import re
from collections import defaultdict
from pathlib import Path

import pytest

from app.evaluation import game_of, read_labels

EVAL = Path(__file__).resolve().parents[1] / "eval"
DEV_FILES = ["labels.csv", "fixed.csv"]
# Old notes use ESPN's team codes; the same game must not show up under both spellings.
ESPN_TO_NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def clips(name):
    by_job = defaultdict(list)
    for label in read_labels(EVAL / name):
        by_job[label.job_id].append(label)
    return by_job


@pytest.mark.parametrize("name", DEV_FILES)
def test_each_attempt_is_labeled_once(name):
    for job, rows in clips(name).items():
        releases = [row.release_s for row in rows]
        assert len(releases) == len(set(releases)), f"{job}: duplicate release times {releases}"


@pytest.mark.parametrize("name", DEV_FILES)
def test_no_shot_rows_stand_alone(name):
    for job, rows in clips(name).items():
        if any(row.release_s is None for row in rows):
            assert len(rows) == 1, f"{job}: a no-shot row next to shot rows"


@pytest.mark.parametrize("name", DEV_FILES)
def test_a_clip_names_one_game(name):
    for job, rows in clips(name).items():
        games = {game_of(row.notes) for row in rows} - {""}
        assert len(games) <= 1, f"{job}: rows name different games {sorted(games)}"


@pytest.mark.parametrize("name", DEV_FILES)
def test_a_game_has_one_spelling(name):
    # game_of() groups clips by its exact text, so "MEM@NO" and "MEM@NOP" on one date would split a game.
    spellings = defaultdict(set)
    for rows in clips(name).values():
        for row in rows:
            game = game_of(row.notes)
            found = re.fullmatch(r"([A-Z]+)@([A-Z]+) (\S+)", game)
            if found:
                away, home, date = found.groups()
                spellings[(ESPN_TO_NBA.get(away, away), ESPN_TO_NBA.get(home, home), date)].add(game)
    split = {key: sorted(names) for key, names in spellings.items() if len(names) > 1}
    assert not split, f"one game under several tags: {split}"


def test_label_fixes_2026_11_06_hold():
    # The three clips corrected from the frames on 2026-11-06 keep their rows and audit notes.
    rows = clips("labels.csv")
    got = {job: [(row.release_s, row.outcome, row.shot_type) for row in rows[job]]
           for job in ("1f3b385091c9", "483cfc8b3060", "3f4df5367418")}
    assert got["1f3b385091c9"] == [(7.97, "unclear", "layup")]
    assert got["483cfc8b3060"] == [(2.55, "missed", "floater"), (3.78, "missed", "layup"), (4.95, "made", "layup")]
    assert got["3f4df5367418"] == [(2.95, "missed", "jump shot"), (4.95, "made", "dunk")]
    for job, release in [("1f3b385091c9", 7.97), ("483cfc8b3060", 3.78), ("483cfc8b3060", 4.95), ("3f4df5367418", 4.95)]:
        (row,) = [row for row in rows[job] if row.release_s == release]
        assert "Corrected 2026-11-06 from the frames: " in row.notes
