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


def test_arm_hanging_straight_before_the_lift_is_not_the_release():
    poses = shooter(set_elbow=75)
    for pose in poses:
        if pose.time_s < 0.8:   # arm fully straight at the side before the routine
            x, y, v = pose.landmarks["right_shoulder"]
            pose.landmarks["right_elbow"], pose.landmarks["right_wrist"] = (x, y + 0.12, v), (x, y + 0.23, v)
    reps = shot_signatures(poses)
    assert len(reps) == 1 and reps[0]["features"]["set_elbow_deg"] == pytest.approx(75, abs=5)


def test_pose_glitch_that_looks_like_a_lift_is_not_a_rep():
    poses = shooter(hold=0.0, duration=2.0)
    for pose in poses:   # the arm is up for an instant only
        if pose.time_s > 1.65:
            x, y, v = pose.landmarks["right_shoulder"]
            pose.landmarks["right_elbow"], pose.landmarks["right_wrist"] = (x, y + 0.12, v), (x, y + 0.23, v)
    assert shot_signatures(poses) == []
    upside_down = shooter()
    for pose in upside_down:
        pose.landmarks["left_hip"], pose.landmarks["right_hip"] = (0.5, 0.2, 0.9), (0.5, 0.2, 0.9)
    assert shot_signatures(upside_down) == []


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
    assert db["views"]["near"]["features"]["set_elbow_deg"]["weight"] > 0.9
    mine = [shot_signatures(shooter(set_elbow=100, release_elbow=162))[0] for _ in range(3)]
    result = match(mine, db)
    assert [m["name"] for m in result["matches"]] == ["Open", "Folded"]
    assert result["matches"][0]["similarity"] > result["matches"][1]["similarity"]
    assert 1 <= len(result["matches"][1]["feedback"]) <= 3
    assert len(result["matches"][0]["skeleton"]) == len(SKELETON_TIMES)
    assert len(result["skeleton"]["yours"]) == len(SKELETON_TIMES)
    assert "feedback" not in match(mine, db, detail=1)["matches"][1]
    notes = feedback(median_features(mine), db["views"]["near"]["players"][0], db["views"]["near"]["features"], limit=len(FEATURES))
    note = next(n for n in notes if n["feature"] == "set_elbow_deg")
    assert note["text"].startswith("Fold your shooting arm more at the set point")
    assert "yours is 100°, Folded's is 6" in note["text"]


def test_near_and_far_arm_views_are_matched_separately():
    def far(player):
        return {**player, "reps": [dict(r, arm_side="far") for r in player["reps"]]}
    # Seen from the far side the two players' measurements are swapped.
    near = [_player("A", 60, 176), _player("B", 110, 160)]
    db = build_db([{**a, "reps": a["reps"] + far(b)["reps"]} for a, b in zip(near, near[::-1])])
    mine = _player("me", 62, 175)["reps"]
    assert shot_signatures(shooter())[0]["arm_side"] == "near"
    assert shot_signatures(shooter(facing=-1))[0]["arm_side"] == "far"
    assert shot_signatures(shooter(facing=-1, hand="left"), handedness="left")[0]["arm_side"] == "near"
    assert match(mine, db)["matches"][0]["name"] == "A"
    result = match(far({"reps": mine})["reps"], db)
    assert result["matches"][0]["name"] == "B" and result["arm_side"] == "far"
    assert not any("templates from this side" in note for note in result["notes"])
    fewer = build_db([{**a, "reps": a["reps"] + far(b)["reps"]} for a, b in zip(near, near[::-1])] + [_player("C", 80, 170)])
    assert "Only 2 of 3 players" in " ".join(match(far({"reps": mine})["reps"], fewer)["notes"])
    assert match(far({"reps": mine})["reps"], build_db(near)) is None   # no far-side templates


def test_match_needs_reps_and_a_database():
    assert match([], build_db([_player("A", 60, 176), _player("B", 110, 160)])) is None
    assert match(_player("A", 60, 176)["reps"], {"views": {"near": {"players": [], "features": {}}}}) is None


def test_match_warns_about_few_reps_and_a_view_that_is_not_side_on():
    db = build_db([_player("A", 60, 176), _player("B", 110, 160)])
    rep = dict(shot_signatures(shooter())[0], shoulder_width=0.6)
    notes = " ".join(match([rep], db)["notes"])
    assert "not a side view" in notes and "1 free throw." in notes
