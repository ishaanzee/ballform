import pytest

from app.game import analyze_game_shots
from app.models import Detection, PoseFrame, ShotResult


def player(x, wrist=None, hidden=None, time=1.):
    landmarks = {"left_shoulder": (x-.02, .4, .95), "right_shoulder": (x+.02, .4, .95),
                 "left_hip": (x-.02, .6, .95), "right_hip": (x+.02, .6, .95),
                 "left_wrist": (x-.03, .3, .95), "right_wrist": (x+.03, .3, .95)}
    if wrist:
        landmarks["right_wrist"] = (*wrist, .95)
    if hidden:
        landmarks[hidden] = (*landmarks[hidden][:2], .1)
    return PoseFrame(round(time*30), time, landmarks)


def run(players, prior=None, ball=None, aspect=1.):
    shot = ShotResult(1, .5, 1., 2., "unknown", 0., [], {})
    frames = [{"frame": 30, "time_s": 1., "players": players}]
    if prior is not None:
        frames.insert(0, {"frame": 15, "time_s": .5, "players": prior})
    summary = analyze_game_shots([shot], frames, [ball or Detection(30, 1., .33, .3, .9)], 30, aspect)
    return shot.game, summary


def test_open_space_scores_higher_than_close_contest():
    open_game, summary = run([player(.3), player(.85)])
    contested, _ = run([player(.3), player(.45, wrist=(.4, .3))])
    assert open_game["score"] > contested["score"]
    assert summary["scored_shots"] == 1
    assert open_game["metrics"]["separation_torso"] == 2.75


def test_aspect_corrects_horizontal_distances():
    game, _ = run([player(.3), player(.6)], aspect=2.)
    assert game["metrics"]["separation_torso"] == 3.


def test_ambiguous_shooter_abstains():
    game, summary = run([player(.3), player(.5, wrist=(.34, .3))])
    assert game["score"] is None
    assert summary["mean_score"] is None


@pytest.mark.parametrize("players", [[player(.3)], [player(.3), player(.6), player(.8)],
                                    [player(.3), player(.6, hidden="right_wrist")],
                                    [player(.3, hidden="left_hip"), player(.6)]])
def test_occlusion_and_extra_people_withhold_score(players):
    assert run(players)[0]["score"] is None


def test_stale_ball_withholds_score():
    assert run([player(.3), player(.7)], ball=Detection(20, .7, .33, .3, .9))[0]["score"] is None


def test_trend_matches_reordered_players():
    game, _ = run([player(.3), player(.8)], prior=[player(.7, time=.5), player(.3, time=.5)])
    assert game["metrics"]["separation_change_torso"] == .5


def test_missing_prior_does_not_invent_trend():
    assert run([player(.3), player(.7)])[0]["metrics"]["separation_change_torso"] is None


@pytest.mark.parametrize("field", ["x", "y", "confidence"])
def test_nonfinite_ball_withholds_score(field):
    ball = Detection(30, 1., .33, .3, .9)
    setattr(ball, field, float("nan"))
    assert run([player(.3), player(.7)], ball=ball)[0]["score"] is None


def test_landmark_visibility_reduces_confidence():
    clean, _ = run([player(.3), player(.7)])
    defender = player(.7)
    x, y, _ = defender.landmarks["left_hip"]
    defender.landmarks["left_hip"] = (x, y, .66)
    reduced, _ = run([player(.3), defender])
    assert reduced["score"] == clean["score"]
    assert reduced["confidence"] < clean["confidence"]


def test_zoom_scale_change_withholds_trend():
    prior = [player(.3, time=.5), player(.7, time=.5)]
    for p in prior:
        for name in ("left_shoulder", "right_shoulder"):
            x, _, confidence = p.landmarks[name]
            p.landmarks[name] = (x, .46, confidence)
    game, _ = run([player(.3), player(.8)], prior=prior)
    assert game["score"] is not None
    assert game["metrics"]["separation_change_torso"] is None
