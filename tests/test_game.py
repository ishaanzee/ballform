import math

import pytest

from app.game import analyze_game_shots
from app.models import Detection, PoseFrame, ShotResult


def player(x, wrist=None, hidden=None, time=1., appearance=None, track_id=None):
    landmarks = {"left_shoulder": (x-.02, .4, .95), "right_shoulder": (x+.02, .4, .95),
                 "left_hip": (x-.02, .6, .95), "right_hip": (x+.02, .6, .95),
                 "left_wrist": (x-.03, .3, .95), "right_wrist": (x+.03, .3, .95)}
    if wrist:
        landmarks["right_wrist"] = (*wrist, .95)
    if hidden:
        landmarks[hidden] = (*landmarks[hidden][:2], .1)
    return PoseFrame(round(time*30), time, landmarks, track_id=track_id, appearance=appearance)


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


def test_five_on_five_selects_ball_owner_and_nearest_opponent():
    offense = (.25, .35, .45)
    defense = (.78, .58, .18)
    players = [
        player(.30, wrist=(.33, .30), appearance=offense, track_id=1),
        player(.05, appearance=offense, track_id=2),
        player(.12, appearance=offense, track_id=3),
        player(.68, appearance=offense, track_id=4),
        player(.82, appearance=offense, track_id=5),
        player(.43, appearance=defense, track_id=6),
        player(.57, appearance=defense, track_id=7),
        player(.72, appearance=defense, track_id=8),
        player(.90, appearance=defense, track_id=9),
        player(.96, appearance=defense, track_id=10),
    ]
    game, summary = run(players)
    assert game["score"] is not None
    assert game["metrics"]["visible_players"] == 10
    assert game["players"] == {
        "ball_carrier_track_id": 1,
        "shooter_track_id": 1,
        "defender_track_id": 6,
    }
    assert game["metrics"]["separation_torso"] == .65
    assert summary["method"]["version"] == "shot-space-v3-broadcast"


def test_crowded_frame_with_indistinguishable_jerseys_withholds_matchup():
    same = (.4, .4, .4)
    players = [
        player(.30, wrist=(.33, .30), appearance=same, track_id=1),
        player(.55, appearance=same, track_id=2),
        player(.75, appearance=same, track_id=3),
    ]
    game, _ = run(players)
    assert game["score"] is None
    assert "jersey appearance" in " ".join(game["evidence"])


def test_multiplayer_trend_uses_track_ids_when_pose_order_changes():
    offense = (.2, .3, .4)
    defense = (.8, .6, .2)
    release = [
        player(.30, wrist=(.33, .30), appearance=offense, track_id=1),
        player(.80, appearance=offense, track_id=2),
        player(.50, appearance=defense, track_id=3),
        player(.92, appearance=defense, track_id=4),
    ]
    prior = [
        player(.45, time=.5, appearance=defense, track_id=3),
        player(.92, time=.5, appearance=defense, track_id=4),
        player(.80, time=.5, appearance=offense, track_id=2),
        player(.30, wrist=(.33, .30), time=.5, appearance=offense, track_id=1),
    ]
    game, _ = run(release, prior=prior)
    assert game["score"] is not None
    assert game["metrics"]["separation_change_torso"] == .25


def test_background_lowered_hand_cannot_become_shooter():
    shooter = player(.3, wrist=(.34, .3), track_id=1)
    background = player(.6, wrist=(.331, .3), track_id=2)
    for side in ('left', 'right'):
        background.landmarks[side+'_shoulder'] = (.6, .1, .95)
        background.landmarks[side+'_hip'] = (.6, .3, .95)
    game, _ = run([shooter, background])
    assert game['players']['shooter_track_id'] == 1


def test_raised_contesting_opponent_preferred_to_background_hip():
    shooter = player(.3, appearance=(.1,.2,.3), track_id=1)
    background = player(.45, appearance=(.8,.6,.3), track_id=2)
    for side in ('left', 'right'):
        background.landmarks[side+'_wrist'] = (.45, .65, .95)
    contester = player(.65, wrist=(.45,.3), appearance=(.8,.6,.3), track_id=3)
    game, _ = run([shooter, background, contester])
    assert game['players']['defender_track_id'] == 3


def test_occluded_hand_yields_bounds_not_a_fabricated_point_score():
    game, _ = run([player(.3), player(.7, hidden='right_wrist')])
    complete, _ = run([player(.3), player(.7)])
    assert game['score'] is None
    assert game['status'] == 'partial'
    assert game['score_range']['lower'] <= complete['score'] <= game['score_range']['upper']
    assert game['metrics']['contest_clearance_torso'] is None
    assert game['metrics']['visible_hand_clearance_torso'] is not None


def _crouch(pose, factor):
    """Shorten the projected torso by lowering the shoulders (posture, not zoom)."""
    for name in ("left_shoulder", "right_shoulder"):
        x, y, confidence = pose.landmarks[name]
        pose.landmarks[name] = (x, .6 - (.6 - y) * factor, confidence)
    return pose


def _tracked_frames(defender_x_before, times=(.4, .5, .6), crouch=None, shift=0.):
    """Four tracked players; release at 1.0 s, earlier frames at `times`."""
    offense, defense = (.2, .3, .4), (.8, .6, .2)
    def roster(time, defender_x, dx=0.):
        return [player(.30 + dx, wrist=(.33 + dx, .30), time=time, appearance=offense, track_id=1),
                player(.80 + dx, time=time, appearance=offense, track_id=2),
                player(defender_x + dx, time=time, appearance=defense, track_id=3),
                player(.95 + dx, time=time, appearance=defense, track_id=4)]
    frames = []
    for t in times:
        players = roster(t, defender_x_before, shift)
        if crouch:
            _crouch(players[2], crouch)
        frames.append({"frame": round(t * 30), "time_s": t, "players": players})
    frames.append({"frame": 30, "time_s": 1., "players": roster(1., .50)})
    return frames


def _run_frames(frames, balls=None):
    shot = ShotResult(1, .5, 1., 2., "unknown", 0., [], {})
    analyze_game_shots([shot], frames, balls or [Detection(30, 1., .33, .3, .9)], 30, 1.)
    return shot.game


def test_defender_crouching_does_not_void_the_trend():
    game = _run_frames(_tracked_frames(.45, crouch=.6))
    assert game["metrics"]["separation_change_torso"] == pytest.approx(.25)


def test_trend_uses_the_whole_window_not_one_exact_frame():
    game = _run_frames(_tracked_frames(.45, times=(.35, .65)))
    assert game["metrics"]["separation_change_torso"] == pytest.approx(.25)


def test_camera_pan_does_not_void_the_trend():
    game = _run_frames(_tracked_frames(.45, shift=.4))
    assert game["metrics"]["separation_change_torso"] == pytest.approx(.25)


def test_trend_needs_identity_continuity_through_release():
    frames = _tracked_frames(.45, times=(.4, .5, .6))
    gap = [{"frame": round(t * 30), "time_s": t, "players": []} for t in (.7, .75, .8, .85, .9, .95)]
    game = _run_frames(frames[:-1] + gap + frames[-1:])
    assert game["metrics"]["separation_change_torso"] is None


def test_defender_hand_hidden_on_release_frame_is_measured_from_an_adjacent_frame():
    release = [player(.3, track_id=1), player(.7, hidden="right_wrist", track_id=2)]
    later = [player(.3, time=1.033, track_id=1), player(.7, time=1.033, track_id=2)]
    frames = [{"frame": 30, "time_s": 1., "players": release}, {"frame": 31, "time_s": 1.033, "players": later}]
    balls = [Detection(30, 1., .33, .3, .9), Detection(31, 1.033, .33, .3, .9)]
    game = _run_frames(frames, balls)
    complete, _ = run([player(.3), player(.7)])
    assert game["status"] == "measured" and game["score"] == complete["score"]
    assert game["confidence"] < complete["confidence"]
    assert any("nearest frame where it was visible" in item for item in game["evidence"])


def _court_map():
    from app.camera_motion import CourtMap, follow_court
    from app.court import apply, fit, template
    from tests_support_court import synthetic_camera
    court_to_image = synthetic_camera()[2]
    ids = ["lane_base_left", "lane_base_right", "ft_left", "ft_right", "three_top", "half_right", "half_left"]
    image = apply(court_to_image, [template("nba").landmarks[key] for key in ids])
    calibration = fit([(key, (x / 1920, y / 1080)) for key, (x, y) in zip(ids, image)], 1920, 1080)
    frames = follow_court(None, list(range(30)), 0, calibration.image_to_court, template("nba"), {}, [],
                          1920, 1080, 30, fixed=True)
    return CourtMap(calibration, frames, 0, 1920, 1080), court_to_image


def _standing(frame, track_id, spot, court_to_image, lift_px=0., wrist=None):
    from app.court import apply
    x, y = apply(court_to_image, [spot])[0] / (1920, 1080)
    lift = lift_px / 1080
    landmarks = {"left_ankle": (x - .004, y - lift, .9), "right_ankle": (x + .004, y - lift, .9),
                 "left_hip": (x - .006, y - .08 - lift, .9), "right_hip": (x + .006, y - .08 - lift, .9),
                 "left_shoulder": (x - .008, y - .16 - lift, .9), "right_shoulder": (x + .008, y - .16 - lift, .9),
                 "left_wrist": (x - .012, y - .2 - lift, .9), "right_wrist": (x + .012, y - .2 - lift, .9)}
    if wrist:
        landmarks["right_wrist"] = (*wrist, .9)
    return PoseFrame(frame, frame / 30, landmarks, track_id=track_id)


def _court_shot(jump=True, defender_spot=(0., 22.)):
    from app.game import add_court_metrics
    court_map, court_to_image = _court_map()
    frames = []
    for frame in range(25):
        # The shooter plants at (-5, 29) and leaves the floor after frame 18.
        lift = 12. * max(0, frame - 18) if jump else 0.
        frames.append({"frame": frame, "time_s": frame / 30, "players": [
            _standing(frame, 1, (-5., 29.), court_to_image, lift),
            _standing(frame, 2, defender_spot, court_to_image, wrist=(.5, .3))]})
    game = {"release_frame": 24, "players": {"shooter_track_id": 1, "defender_track_id": 2},
            "metrics": {"separation_torso": 2.5, "contest_clearance_torso": 1.2, "visible_hand_clearance_torso": None},
            "evidence": [], "limitations": ["No court calibration: shot distance, defender depth, possession outcome "
                                            "and true 3D spacing are not measured."]}
    shot = ShotResult(1, 0., .8, 1.5, "unknown", 0., [], {}, game=game)
    summary = {"limitations": list(game["limitations"])}
    add_court_metrics([shot], frames, [Detection(24, .8, .52, .3, .9)], court_map, summary)
    return shot.game, summary


def test_court_metrics_measure_the_jump_shot_from_the_take_off_spot():
    game, summary = _court_shot()
    metrics = game["metrics"]
    assert metrics["shot_distance_ft"] == pytest.approx(24.3, abs=.15)
    assert metrics["shot_zone"] == "above_break_three"
    assert (metrics["shooter_court_x_ft"], metrics["shooter_court_y_ft"]) == pytest.approx((-5., 29.), abs=.15)
    assert metrics["separation_ft"] == pytest.approx(math.dist((-5., 29.), (0., 22.)), abs=.15)
    assert metrics["contest_clearance_ft"] is not None
    # Torso-length metrics stay as they were.
    assert metrics["separation_torso"] == 2.5 and metrics["contest_clearance_torso"] == 1.2
    assert "before take-off" in " ".join(game["evidence"])
    assert all(not item.startswith("No court calibration") for item in game["limitations"] + summary["limitations"])


def test_airborne_feet_are_not_mapped_as_the_shot_spot():
    # Mapping the release-frame feet (lifted ~70 px) would put the shooter feet farther out.
    game, _ = _court_shot()
    from app.court import apply
    court_map, court_to_image = _court_map()
    airborne = apply(court_to_image, [(-5., 29.)])[0] - (0, 72)
    assert math.dist(court_map.to_court(24, airborne), (-5., 29.)) > 2
    assert game["metrics"]["shooter_court_y_ft"] == pytest.approx(29., abs=.15)


def test_set_shot_uses_the_release_frame_feet():
    game, _ = _court_shot(jump=False)
    assert game["metrics"]["shot_distance_ft"] == pytest.approx(24.3, abs=.15)
    assert "no take-off was detected" in " ".join(game["evidence"])


def test_court_metrics_withhold_when_the_shooter_is_unknown():
    from app.game import add_court_metrics
    court_map, _ = _court_map()
    game = {"release_frame": None, "players": {"shooter_track_id": None, "defender_track_id": None},
            "metrics": {}, "evidence": [], "limitations": []}
    shot = ShotResult(1, 0., .8, 1.5, "unknown", 0., [], {}, game=game)
    add_court_metrics([shot], [], [], court_map)
    assert game["metrics"]["shot_distance_ft"] is None and "not identified" in game["evidence"][0]
