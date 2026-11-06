import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "labeling"))

import scorebug  # noqa: E402

# Observations (text, conf, (x, y, w, h)) below are Apple Vision reads of 1920x1080 dev clips.


def inside(observations, region):
    """The observations a crop to region reads whole."""
    x1, y1, x2, y2 = region
    return [o for o in observations
            if x1 <= o[2][0] and o[2][0] + o[2][2] <= x2 and y1 <= o[2][1] and o[2][1] + o[2][3] <= y2]


def texts(line):
    return [t.text for t in scorebug.tokens_of([(line, 1.0, (0, 0, 300, 30))])]


def test_letter_read_for_the_one_of_1st():
    # re.I lets i and L match as well as I and l; 'iST' and 'LST' used to raise in int().
    for token in ("IST", "lST", "iST", "LST", "lst", "ıST", "İST"):
        assert scorebug.parse_period(token) == 1
    # Only 1 takes ST, so Indiana's abbreviation is not the 1st period.
    for token in ("IND", "iND", "LTH", "IRD"):
        assert scorebug.parse_period(token) is None
    assert texts("iST 5:44 :23") == ["iST", "5:44", ":23"]
    r = scorebug.parse_reading([("14", 1.0, (268, 991, 70, 45)), ("15", 1.0, (597, 991, 78, 45)),
                                ("LST 5:44 :23", 1.0, (686, 993, 260, 43))])
    assert (r.period, r.clock, r.away, r.home, r.shot_clock, r.conf) == (1, 344, 14, 15, 23, "high")


def test_spelled_out_periods():
    cases = {"THIRD QUARTER": 3, "First Quarter": 1, "2ND QTR": 2, "FOURTH QTR": 4, "THIRDQUARTER": 3,
             "OVERTIME": 5, "2ND OT": 6, "DOUBLE OVERTIME": 6, "TRIPLE OT": 7, "HALFTIME": 2, "HALF TIME": 2}
    assert {token: scorebug.parse_period(token) for token in cases} == cases
    for token in ("THIRD", "SECOND", "QUARTER", "QTR", "HALF", "QUARTERFINALS"):
        assert scorebug.parse_period(token) is None
    # A spelled period stays one token; everything else splits as before.
    assert texts("THIRD QUARTER") == ["THIRD QUARTER"]
    assert texts("THIRD QTR 3PT") == ["THIRD QTR", "3", "PT"]
    assert texts("QUARTERFINALS Emirates NBA CUP") == ["QUARTERFINALS", "Emirates", "NBA", "CUP"]
    assert texts("LIVE 1ST HALF O/U") == ["LIVE", "1ST", "HALF", "O", "U"]
    # '2ND OT' is the 6th period, not the 2nd or the 5th.
    r = scorebug.parse_reading([("101", 1.0, (200, 900, 60, 45)), ("99", 1.0, (300, 900, 60, 45)),
                                ("2ND OT 3:12", 1.0, (400, 900, 200, 45))])
    assert (r.period, r.clock, r.away, r.home) == (6, 192, 101, 99)


def test_blazers_bug():
    # The period is spelled out under the clock; the bug was not found at all before.
    obs = [("67", 1.0, (201, 896, 67, 48)), ("TO II!!", 0.3, (53, 965, 84, 20)),
           ("THIRD QUARTER", 1.0, (53, 1010, 154, 25)), ("65", 1.0, (301, 893, 70, 50)),
           ("TO 1III", 0.3, (282, 962, 95, 26)), ("8:09", 1.0, (307, 1005, 73, 34)), ("21", 1.0, (430, 1005, 39, 31)),
           ("mado", 0.3, (827, 772, 487, 142))]
    r = scorebug.parse_reading(obs)
    assert (r.period, r.clock, r.away, r.home, r.conf) == (3, 489, 67, 65, "high")
    assert scorebug.locate({0: obs}, 1920, 1080) == (0, 842, 537, 1080)


def test_halftime_only_on_a_stopped_clock():
    def reading(clock):
        return scorebug.parse_reading([("52", 1.0, (200, 900, 60, 45)), ("49", 1.0, (300, 900, 60, 45)),
                                       ("HALFTIME", 1.0, (60, 1000, 150, 25)), (clock, 1.0, (300, 1000, 70, 30))])
    assert reading("0.0").period == 2
    assert reading("8:24").period is None  # counting down the break, not game time


def test_wide_one_row_bug():
    # ESPN's national bug: the away score is 500 px left of the clock, past .2 of the width.
    # The box used to cut '48' to '8', and the reading came out 47-10 (the shot clock).
    espn = [("ESPN", 1.0, (482, 960, 102, 37)), ("MIN", 1.0, (608, 963, 95, 31)), ("48", 1.0, (742, 960, 103, 36)),
            ("20-10 (4th IN WEST)", 1.0, (611, 1020, 213, 24)), ("52", 1.0, (999, 924, 42, 28)),
            ("DEN", 1.0, (876, 963, 89, 31)), ("21-8 (3rd IN WEST)", 1.0, (874, 1021, 190, 26)),
            ("47", 1.0, (1027, 960, 86, 36)), ("2ND", 1.0, (1141, 966, 53, 31)), ("2:46", 1.0, (1233, 965, 115, 37)),
            ("10", 1.0, (1390, 966, 45, 31)), ("NBA + CHRISTMAS", 1.0, (1150, 1019, 285, 28))]
    region = scorebug.locate({0: espn}, 1920, 1080)
    assert region[0] < 742 and region[2] > 1435
    r = scorebug.parse_reading(inside(espn, region))
    assert (r.period, r.clock, r.away, r.home, r.shot_clock) == (2, 166, 48, 47, 10)
    # The NBA bar of the local feeds: the away score is 530 px from the clock.
    bar = [("114", 1.0, (257, 991, 92, 45)), ("BONUS", 1.0, (282, 1038, 53, 14)), ("110", 1.0, (586, 991, 89, 45)),
           ("BONUS", 1.0, (608, 1035, 64, 17)), ("4TН 4:20 :2", 1.0, (698, 993, 251, 42)),
           ("NICKEIL ALEXANDER-WALKER: 8 POINTS 2 FOULS", 1.0, (1108, 1002, 706, 28))]
    r = scorebug.parse_reading(inside(bar, scorebug.locate({0: bar}, 1920, 1080)))
    assert (r.period, r.clock, r.away, r.home) == (4, 260, 114, 110)
    # Vision may read the home score twice (alone and in a line running into the ticker).
    twice = [("108 Fous", 1.0, (197, 989, 161, 48)),
             ("105 FOLS 4TH 5:14 :24 DEVIN BOOKER: 24 POINTS 6/18 FIELD GOALS", 1.0, (488, 985, 1256, 48)),
             ("105 rzzs", 1.0, (525, 991, 162, 50)), ("4TH 5:14 :24", 1.0, (698, 998, 249, 42))]
    assert scorebug.locate({0: twice}, 1920, 1080)[0] < 197
    # A score-sized number in the ticker on the other side of the clock stays out.
    ticker = [("28", 1.0, (265, 991, 78, 45)), ("41", 1.0, (600, 991, 75, 45)), ("2ND 9:05 :08", 1.0, (692, 993, 257, 42)),
              ("59 FOLLOW @OKCTHUNDER ON", 1.0, (1232, 985, 440, 40))]
    region = scorebug.locate({0: ticker}, 1920, 1080)
    assert region[0] < 265 and region[2] < 1232


def test_anchors_in_two_places_find_no_bug():
    # Two probes show a bottom-left bug, two a HALFTIME graphic top right: the median position
    # is near neither, so there is no box (min() of no boxes used to raise).
    bug = [("14", 1.0, (60, 960, 70, 45)), ("15", 1.0, (160, 960, 70, 45)), ("1ST 5:44 :23", 1.0, (250, 962, 260, 43))]
    graphic = [("HALFTIME", 1.0, (1300, 300, 300, 50)), ("8:24", 1.0, (1350, 360, 120, 50))]
    assert scorebug.locate({0: bug, 1: bug, 2: graphic, 3: graphic}, 1920, 1080) is None
    assert scorebug.locate({0: bug, 1: bug, 2: graphic}, 1920, 1080)[0] == 0  # the bug is the majority


def test_stacked_bug_box_unchanged():
    # Both scores are in the box, so a number further along the clock's row is not taken.
    obs = [("CLE", 1.0, (1490, 876, 73, 36)), ("DEN", 1.0, (1490, 924, 78, 36)), ("71", 1.0, (1616, 873, 42, 39)),
           ("73", 1.0, (1613, 924, 50, 36)), ("6:31", 1.0, (1711, 879, 70, 34)), ("3RD :20", 1.0, (1694, 929, 109, 31))]
    assert scorebug.locate({0: obs}, 1920, 1080) == (1422, 822, 1871, 1011)
    assert scorebug.locate({0: obs + [("23", 1.0, (1100, 876, 40, 36))]}, 1920, 1080) == (1422, 822, 1871, 1011)
