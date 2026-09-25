import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.court import (STANDARDS, apply, camera_from_homography, fit, floor_point, general_position,
                       parse_landmarks, template, template_json, three_point_margin, takeoff,
                       vertical_plane_distance, zone)

from tests_support_court import synthetic_camera

W, H = 1920, 1080


def clicks(court_to_image, ids, standard="nba", noise=None):
    court = template(standard)
    image = apply(court_to_image, [court.landmarks[key] for key in ids])
    if noise:
        for key, offset in noise.items():
            image[ids.index(key)] += offset
    return [(key, (x / W, y / H)) for key, (x, y) in zip(ids, image)]


def test_nba_template_matches_rule_book_distances():
    court = template("nba")
    marks = court.landmarks
    assert marks["lane_base_right"][0] - marks["lane_base_left"][0] == 16
    assert marks["ft_left"][1] - court.dims["backboard_y"] == 15
    assert court.rim == (0., 5.25)
    assert math.dist(marks["three_top"], court.rim) == pytest.approx(23.75)
    assert math.dist(marks["three_break_left"], court.rim) == pytest.approx(23.75)
    assert court.three_break_y == pytest.approx(14.2, abs=.01)


def test_standards_differ_where_the_rule_books_do():
    assert template("fiba").dims["three_radius"] == pytest.approx(22.146, abs=.001)
    assert template("ncaa").landmarks["lane_base_right"][0] == 6
    # The high-school arc is a semicircle meeting straight lines at the rim's depth; no restricted area.
    nfhs = template("nfhs")
    assert nfhs.three_break_y == pytest.approx(nfhs.rim[1])
    assert "restricted_top" not in nfhs.landmarks
    assert set(template_json()) == set(STANDARDS)


@pytest.mark.parametrize("point, expected", [
    ((0., 10.), "paint"), ((10., 15.), "midrange"), ((23., 4.), "corner_three"),
    ((0., 28.8), "midrange"), ((0., 29.2), "above_break_three"), ((-18., 25.), "above_break_three"),
    ((0., 48.), None), ((30., 10.), None),
])
def test_zones(point, expected):
    assert zone(point, template("nba")) == expected


def test_three_point_margin_is_signed():
    court = template("nba")
    assert three_point_margin((0., 30.), court) == pytest.approx(1.)
    assert three_point_margin((21., 3.), court) == pytest.approx(-1.)


def test_fit_recovers_a_known_view_and_reports_leave_one_out_error():
    _, _, court_to_image = synthetic_camera()
    ids = ["lane_base_left", "lane_base_right", "ft_left", "ft_right", "three_top", "half_right"]
    calibration = fit(clicks(court_to_image, ids), W, H)
    assert max(calibration.errors_px) < .01
    assert max(calibration.leave_one_out_px) < .01
    probe = (12., 30.)
    assert apply(calibration.image_to_court, apply(court_to_image, [probe]))[0] == pytest.approx(probe, abs=1e-4)


def test_four_clicks_fit_exactly_so_no_error_can_be_measured():
    _, _, court_to_image = synthetic_camera()
    ids = ["lane_base_left", "lane_base_right", "ft_left", "ft_right"]
    calibration = fit(clicks(court_to_image, ids, noise={"ft_left": (6., 0.)}), W, H)
    assert max(calibration.errors_px) < .01  # the misclick is invisible with only 4 points
    assert calibration.leave_one_out_px is None
    assert calibration.summary()["leave_one_out_error_px"] is None


def test_a_misclick_is_dropped_but_a_lone_far_landmark_is_kept():
    _, _, court_to_image = synthetic_camera()
    ids = ["lane_base_left", "lane_base_right", "ft_left", "ft_right", "baseline_right",
           "three_top", "half_right"]
    calibration = fit(clicks(court_to_image, ids, noise={"ft_right": (40., 25.)}), W, H)
    assert calibration.outliers == ["ft_right"]
    assert max(e for key, e in zip(calibration.ids, calibration.errors_px) if key != "ft_right") < .5


def test_real_broadcast_clicks_keep_the_half_court_landmark():
    # Hand-placed clicks on frame 0 of the 3074e broadcast. The lone half-court click has
    # a 113 px leave-one-out error only because nothing else constrains that end of the
    # court; RANSAC dropped it (and then ft_left) before this was fixed.
    points = {"lane_base_right": (358, 565), "lane_base_left": (173, 722), "ft_right": (939, 600),
              "ft_left": (834, 762), "baseline_right": (507, 431), "three_base_right": (486, 450),
              "half_right": (1885, 506)}
    calibration = fit([(key, (x / W, y / H)) for key, (x, y) in points.items()], W, H)
    assert calibration.outliers == []
    assert max(calibration.errors_px) < 6
    # The 28 ft coaching-box line meets the far sideline at (1302, 471); it was not clicked.
    assert np.linalg.norm(apply(calibration.court_to_image, [(25., 28.)])[0] - (1302, 471)) < 5
    assert calibration.extrapolation_ft((10., 30.)) == 0.
    assert calibration.extrapolation_ft((-15., 30.)) > 2


def test_camera_is_recovered_from_the_floor_homography():
    _, rotation, court_to_image = synthetic_camera(focal=3500.)
    camera = camera_from_homography(court_to_image, W, H)
    assert camera.focal_px == pytest.approx(3500., rel=1e-6)
    assert camera.center == pytest.approx((-100., 40., 35.), abs=1e-6)
    # A mirrored left/right labelling still yields a camera above the floor.
    mirrored = camera_from_homography(court_to_image @ np.diag([-1., 1., 1.]), W, H)
    assert mirrored.center[2] == pytest.approx(35.)


def test_vertical_plane_distance_measures_feet_at_the_players_depth():
    k, rotation, court_to_image = synthetic_camera()
    camera = camera_from_homography(court_to_image, W, H)
    spot = np.array([5., 25., 0.])
    toward = camera.center - spot
    toward[2] = 0
    sideways = np.cross((0., 0., 1.), toward / np.linalg.norm(toward))
    a, b = spot + (0, 0, 9.), spot + (0, 0, 9.) + 2 * sideways + (0, 0, 1.5)

    def project(point):
        p = k @ (rotation @ (point - camera.center))
        return p[:2] / p[2]
    assert vertical_plane_distance(camera, spot[:2], project(a), project(b)) == pytest.approx(2.5, abs=1e-6)


def pose(ankles=None, box=None):
    landmarks = {}
    for name, value in (ankles or {}).items():
        landmarks[name] = value
    return SimpleNamespace(landmarks=landmarks, box=box)


def test_floor_point_prefers_ankles_then_the_pose_box():
    both = pose({"left_ankle": (.4, .8, .9), "right_ankle": (.5, .9, .9)})
    assert floor_point(both, 100, 100)[0] == pytest.approx((45., 85.))
    assert floor_point(both, 100, 100)[1] == "ankles"
    one = pose({"left_ankle": (.4, .8, .9), "right_ankle": (.5, .9, .2)}, box=(.3, .2, .6, .95))
    assert floor_point(one, 100, 100) == ((40., 80.), "one ankle")
    hidden = pose({"left_ankle": (.4, .8, .1)}, box=(.3, .2, .6, .95))
    point, method = floor_point(hidden, 100, 100)
    assert point == pytest.approx((45., 95.)) and method == "pose box"
    assert floor_point(pose(), 100, 100) is None


def test_takeoff_finds_the_last_grounded_frames_before_the_jump():
    # Earlier stride (foot up), plant on the floor, then the jump through release.
    ys = [480, 500, 500, 501, 500, 490, 470, 450]
    jump = takeoff(ys, torso_px=100)
    assert jump.jumped and jump.grounded == [2, 3, 4]
    assert jump.rise_torso == pytest.approx(.51)


def test_takeoff_reports_no_jump_for_a_set_shot():
    jump = takeoff([500, 501, 499, 500], torso_px=100)
    assert not jump.jumped


def test_takeoff_skips_missing_feet_and_needs_two_samples():
    assert takeoff([None, 500, None, 470], torso_px=100).grounded == [1]
    assert takeoff([None, None, 470], torso_px=100) is None


@pytest.mark.parametrize("payload, message", [
    ({"time_s": 0, "points": []}, "at least 4"),
    ({"points": [{"id": "ft_left", "image": [.1, .1]}] * 4}, "time_s"),
    ({"time_s": 0, "points": [{"id": "nope", "image": [.1, .1]}]}, "distinct id"),
    ({"time_s": 0, "points": [{"id": "ft_left", "image": [.1, .1]}, {"id": "ft_left", "image": [.2, .2]}]}, "distinct id"),
    ({"time_s": 0, "points": [{"id": "ft_left", "image": [1.5, .1]}]}, "normalized"),
    ({"standard": "wnba", "time_s": 0, "points": []}, "standard"),
    ({"time_s": -1, "points": []}, "time_s"),
    ({"time_s": 0, "points": [{"id": key, "image": [.1 * i, .5]} for i, key in enumerate(
        ["baseline_left", "three_base_left", "lane_base_left", "lane_base_right"], 1)]}, "no 3 on one"),
])
def test_invalid_landmarks_are_rejected(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_landmarks(payload)


def test_landmarks_parse_from_json():
    parsed = parse_landmarks('{"standard": "fiba", "frame": 12, "points": ['
                             '{"id": "lane_base_left", "image": [0.1, 0.8]}, {"id": "lane_base_right", "image": [0.3, 0.6]},'
                             '{"id": "ft_left", "image": [0.5, 0.8]}, {"id": "ft_right", "image": [0.6, 0.6]}]}')
    assert parsed["standard"] == "fiba" and parsed["frame"] == 12 and len(parsed["points"]) == 4


def test_general_position():
    assert general_position([(0, 0), (4, 0), (0, 4), (4, 4)])
    assert not general_position([(0, 0), (10, 0), (20, 0), (30, 0), (0, 5)])


def test_web_diagram_template_matches_the_python_template():
    import json
    from pathlib import Path
    published = json.loads((Path(__file__).parent.parent / "web" / "court-template.json").read_text())
    assert published == json.loads(json.dumps(template_json())), (
        "Regenerate web/court-template.json from app.court.template_json().")
