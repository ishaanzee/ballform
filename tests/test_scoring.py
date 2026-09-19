import pytest

from app.models import Detection, PoseFrame
from app.scoring import _form_metrics, analyze_shots, classify_view


def ball(frame, x, y):
    return Detection(frame, frame / 10, x, y, .9, .01)


def test_made_shot_crosses_rim_on_descent():
    track = [
        ball(0, .30, .70), ball(1, .35, .56), ball(2, .40, .40), ball(3, .46, .25),
        ball(4, .51, .18), ball(5, .53, .23), ball(6, .54, .32), ball(7, .55, .43),
        ball(8, .55, .55),
    ]
    shots = analyze_shots(track, [], 10, (.48, .36, .14, .08), {7: 3.0})
    assert len(shots) == 1
    assert shots[0].outcome == "made"
    assert shots[0].outcome_confidence > .7
    assert shots[0].outcome_frame == 7


def test_made_shot_uses_rim_track_during_camera_pan():
    track = [
        ball(0, .30, .70), ball(1, .35, .56), ball(2, .40, .40), ball(3, .46, .25),
        ball(4, .51, .18), ball(5, .56, .23), ball(6, .60, .32), ball(7, .64, .43),
        ball(8, .67, .55),
    ]
    rims = {frame: (.48 + .015 * frame, .36, .14, .08) for frame in range(9)}
    shots = analyze_shots(track, [], 10, rims, {7: 3.0})
    assert len(shots) == 1
    assert shots[0].outcome == "made"
    assert shots[0].outcome_frame == 7


def test_net_only_motion_cannot_turn_an_airball_into_a_make():
    # The ball is lost before it has a descending path through the rim. This
    # represents an airball that brushes the net from the side or below.
    track = [
        ball(0, .24, .70), ball(1, .31, .54), ball(2, .38, .36), ball(3, .42, .23),
        ball(4, .43, .18), ball(5, .39, .25), ball(6, .34, .34),
    ]
    shots = analyze_shots(track, [], 10, (.48, .36, .14, .08), {7: {"strength": 8.0}})
    assert len(shots) == 1
    assert shots[0].outcome == "unknown"


def test_outside_rim_crossing_stays_missed_despite_net_motion():
    track = [
        ball(0, .25, .70), ball(1, .31, .56), ball(2, .37, .39), ball(3, .42, .22),
        ball(4, .44, .18), ball(5, .38, .25), ball(6, .35, .34), ball(7, .33, .43),
    ]
    shots = analyze_shots(track, [], 10, (.48, .36, .14, .08), {7: {"strength": 9.0}})
    assert len(shots) == 1
    assert shots[0].outcome == "missed"


def test_occluded_path_needs_geometry_and_delayed_net_event_for_likely_make():
    track = [
        ball(0, .28, .70), ball(1, .34, .54), ball(2, .41, .36), ball(3, .47, .23),
        ball(4, .50, .18), ball(5, .53, .24), ball(6, .545, .32),
    ]
    shots = analyze_shots(track, [], 10, (.48, .36, .14, .08), {7: {"strength": 3.0}})
    assert len(shots) == 1
    assert shots[0].outcome == "likely made"
    assert shots[0].outcome_frame == 7


def test_unknown_without_rim():
    track = [ball(0, .3, .7), ball(1, .4, .5), ball(2, .5, .2), ball(3, .6, .5), ball(4, .7, .7)]
    shots = analyze_shots(track, [], 5, None)
    assert shots[0].outcome == "unknown"


def test_form_angles_use_image_aspect_and_selected_arm():
    pose = PoseFrame(10, 1.0, {
        "left_shoulder": (.2, .5, .9),
        "left_elbow": (.3, .3, .9),
        "left_wrist": (.4, .3, .9),
    })
    metrics, _ = _form_metrics([pose], [], 10, 10, aspect_ratio=2, handedness="left")
    assert metrics["elbow_angle_at_release_deg"] == pytest.approx(135)
    assert metrics["upper_arm_elevation_deg"] == pytest.approx(45)
    right, _ = _form_metrics([pose], [], 10, 10, aspect_ratio=2)
    assert right["elbow_angle_at_release_deg"] is None


def test_stale_pose_cannot_populate_release_measurements():
    pose = PoseFrame(7, .7, {
        "right_shoulder": (.2, .5, .9),
        "right_elbow": (.3, .3, .9),
        "right_wrist": (.4, .3, .9),
    })
    metrics, cues = _form_metrics([pose], [], 10, 10)
    assert metrics["elbow_angle_at_release_deg"] is None
    assert "unavailable" in cues[0]
    assert not any("No large" in cue for cue in cues)


def test_sparse_detections_cannot_bridge_seconds_to_invent_arc():
    track = [ball(0, .3, .7), ball(1, .4, .5), ball(50, .5, .2), ball(100, .6, .5), ball(101, .7, .7)]
    assert analyze_shots(track, [], 10, None) == []


def test_segment_window_uses_source_frames():
    track = [Detection(f, f / 30, .3 + f / 300, .2 + .001 * (f - 30) ** 2, .9) for f in range(0, 100, 3)]
    shots = analyze_shots(track, [], 30, None)
    assert len(shots) == 1
    assert shots[0].end_s <= 2.5


def test_visible_below_shoulder_bounce_is_not_a_shot():
    track = [ball(0, .3, .9), ball(1, .4, .8), ball(2, .5, .65), ball(3, .6, .8), ball(4, .7, .9)]
    poses = [PoseFrame(2, .2, {"left_shoulder": (.4, .3, .9), "right_shoulder": (.5, .3, .9)})]
    assert analyze_shots(track, poses, 10, None) == []


def test_view_classification_corrects_rectangular_image():
    pose = PoseFrame(0, 0, {
        "left_shoulder": (.4, .3, .9), "right_shoulder": (.6, .3, .9),
        "left_hip": (.4, .8, .9), "right_hip": (.6, .8, .9),
    })
    assert classify_view([pose])[0] == "side"
    assert classify_view([pose], aspect_ratio=2)[0] == "front/rear"


@pytest.mark.parametrize("fps", [30, 60])
def test_smooth_realistic_apex_detected_at_normal_frame_rates(fps):
    track = [Detection(f, f / fps, .2 + .2 * f / fps, .2 + .5 * (f / fps - 1) ** 2, .9)
             for f in range(2 * fps + 1)]
    shots = analyze_shots(track, [], fps, None)
    assert len(shots) == 1
    assert any("Release contact unavailable" in item for item in shots[0].evidence)


def test_game_shot_with_broad_apex_and_visible_release():
    fps = 60
    track = [Detection(f, f / fps, .4 + .002 * (f - 30),
                       .03 + .15 * ((f - 60) / 60) ** 2, .9)
             for f in range(0, 121, 2)]
    release_ball = next(ball for ball in track if ball.frame == 30)
    shooter = PoseFrame(30, 30 / fps, {
        "left_shoulder": (.38, .18, .9), "right_shoulder": (.42, .18, .9),
        "left_hip": (.38, .22, .9), "right_hip": (.42, .22, .9),
        "right_wrist": (release_ball.x, release_ball.y, .9),
    })
    shots = analyze_shots(track, [shooter], fps, None, aspect_ratio=16 / 9, game_mode=True)
    assert len(shots) == 1
    assert shots[0].release_s == .5


def test_monotonic_ball_path_is_not_an_apex():
    track = [Detection(f, f / 30, .2 + f / 200, .8 - f / 100, .9) for f in range(60)]
    assert analyze_shots(track, [], 30, None) == []
