import math

import pytest

from app.form_match import build_db, feedback, match, median_features
from app.form_signature import FEATURES, SKELETON_TIMES, shot_signatures
from app.models import PoseFrame

FPS = 30


def _ramp(t, start, end, a, b):
    u = min(1.0, max(0.0, (t - start) / (end - start)))
    return a + (b - a) * u * u * (3 - 2 * u)   # eased, as a joint moves


def shooter(set_elbow=70.0, release_elbow=175.0, start=1.0, duration=3.2, facing=1, hand="right", hold=0.8):
    """A side-on shooter: the arm hangs, lifts to the set point by start+0.4 s, extends by
    start+0.6 s, holds the follow-through and drops."""
    poses = []
    for frame in range(round(duration * FPS)):
        t = frame / FPS
        s = t - start
        if s < 0.6 + hold:
            lift = _ramp(s, 0.0, 0.4, 20, 100) if s < 0.4 else _ramp(s, 0.4, 0.6, 100, 150)
            elbow = _ramp(s, 0.0, 0.4, 150, set_elbow) if s < 0.4 else _ramp(s, 0.4, 0.6, set_elbow, release_elbow)
        else:
            lift, elbow = _ramp(s, 0.6 + hold, 1.0 + hold, 150, 20), _ramp(s, 0.6 + hold, 1.0 + hold, release_elbow, 150)
        knee_drop = 0.03 * math.exp(-((s - 0.25) / 0.12) ** 2)
        upper, fore = math.radians(lift), math.radians(lift + 180 - elbow)
        shoulder = (0.5, 0.4 + knee_drop)
        elbow_xy = (shoulder[0] + 0.12 * math.sin(upper), shoulder[1] + 0.12 * math.cos(upper))
        wrist = (elbow_xy[0] + 0.11 * math.sin(fore), elbow_xy[1] + 0.11 * math.cos(fore))
        points = {"nose": (0.52, 0.34 + knee_drop), f"{hand}_shoulder": shoulder, f"{hand}_elbow": elbow_xy,
                  f"{hand}_wrist": wrist, "left_hip": (0.5, 0.6 + knee_drop), "right_hip": (0.5, 0.6 + knee_drop),
                  "left_knee": (0.5 + 2 * knee_drop, 0.75), "right_knee": (0.5 + 2 * knee_drop, 0.75),
                  "left_ankle": (0.5, 0.9), "right_ankle": (0.5, 0.9)}
        other = "left" if hand == "right" else "right"
        points.update({f"{other}_shoulder": (0.51, shoulder[1]), f"{other}_elbow": (0.53, shoulder[1] + 0.1),
                       f"{other}_wrist": (0.56, shoulder[1] + 0.05)})
        poses.append(PoseFrame(frame, t, {k: (0.5 + facing * (x - 0.5), y, 0.9) for k, (x, y) in points.items()}))
    return poses


def test_signature_measures_set_point_and_release():
    reps = shot_signatures(shooter(set_elbow=70, release_elbow=175))
    assert len(reps) == 1
    features = reps[0]["features"]
    assert features["set_elbow_deg"] == pytest.approx(70, abs=4)
    assert features["release_elbow_deg"] == pytest.approx(175, abs=3)
    assert 1.5 <= reps[0]["release_s"] <= 1.62
    assert features["release_height"] > features["set_height"]
    assert features["follow_hold_s"] == pytest.approx(0.95, abs=0.2)
    assert features["knee_bend_deg"] < 178
    assert set(features) == set(FEATURES)
    assert len(reps[0]["skeleton"]) == len(SKELETON_TIMES)
    assert reps[0]["shoulder_width"] < 0.2


def test_signature_does_not_depend_on_which_way_the_shooter_faces():
    right = shot_signatures(shooter(facing=1))[0]["features"]
    left = shot_signatures(shooter(facing=-1))[0]["features"]
    assert left == pytest.approx(right, abs=1e-6)
    assert right["set_forward"] != 0


def test_left_handed_shooter_is_measured_on_the_left_arm():
    features = shot_signatures(shooter(set_elbow=90, hand="left"), handedness="left")[0]["features"]
    assert features["set_elbow_deg"] == pytest.approx(90, abs=4)


def test_each_free_throw_in_a_clip_is_a_rep_and_a_hanging_arm_is_none():
    first, second = shooter(duration=3.0), shooter(duration=3.0)
    for pose in second:
        pose.time_s += 3.0
        pose.frame += 90
    assert len(shot_signatures(first + second)) == 2
    assert shot_signatures(shooter(start=99)) == []


def test_hold_cut_off_by_the_end_of_the_clip_is_not_reported():
    features = shot_signatures(shooter(duration=1.9))[0]["features"]
    assert features["follow_hold_s"] is None


def _player(name, set_elbow, release_elbow):
    reps = [shot_signatures(shooter(set_elbow=set_elbow + d, release_elbow=release_elbow - d))[0] for d in (-3, 0, 3)]
    return {"name": name, "hand": "right", "reps": reps}


def test_match_ranks_the_closest_player_first_and_says_what_differs():
    db = build_db([_player("Folded", 60, 176), _player("Open", 110, 160)])
    assert db["features"]["set_elbow_deg"]["weight"] > 0.9
    mine = [shot_signatures(shooter(set_elbow=100, release_elbow=162))[0] for _ in range(3)]
    result = match(mine, db)
    assert [m["name"] for m in result["matches"]] == ["Open", "Folded"]
    assert result["matches"][0]["similarity"] > result["matches"][1]["similarity"]
    assert result["target"] == "Open"
    toward = match(mine, db, target="Folded")
    assert toward["target"] == "Folded"
    assert 1 <= len(toward["feedback"]) <= 3
    notes = feedback(median_features(mine), db["players"][0], db["features"], limit=len(FEATURES))
    note = next(n for n in notes if n["feature"] == "set_elbow_deg")
    assert note["text"].startswith("Fold your shooting arm more at the set point")
    assert "yours is 100°, Folded's is 6" in note["text"]
    assert len(toward["skeleton"]["theirs"]) == len(SKELETON_TIMES)


def test_match_needs_reps_and_a_database():
    assert match([], build_db([_player("A", 60, 176), _player("B", 110, 160)])) is None
    assert match(_player("A", 60, 176)["reps"], {"players": [], "features": {}}) is None


def test_match_warns_about_few_reps_and_a_view_that_is_not_side_on():
    db = build_db([_player("A", 60, 176), _player("B", 110, 160)])
    rep = dict(shot_signatures(shooter())[0], shoulder_width=0.6)
    notes = " ".join(match([rep], db)["notes"])
    assert "not a side view" in notes and "1 free throw." in notes
