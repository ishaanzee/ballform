from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.camera_motion import CourtMap, anchor_frame, follow_court, person_boxes, segment_bounds
from app.court import apply, fit, template

W, H = 960, 540
PX_PER_FT = 8


def camera(frame, focal=1500.):
    """A fixed broadcast camera panning along the sideline and slowly zooming in."""
    position = np.array([-100., 40., 35.])
    forward = np.array([10. + .6 * frame, 25., 0.]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0., 0., 1.))
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    k = np.array([[focal * (1 + .004 * frame), 0., W / 2], [0., focal * (1 + .004 * frame), H / 2], [0., 0., 1.]])
    return k @ np.c_[rotation[:, 0], rotation[:, 1], -rotation @ position]


def floor_texture(seed=0, sparse=False):
    """Random floor texture over the court and a 10 ft apron, in texture pixels.

    A sparse floor is mostly flat, like clean wood, so graphics dominate the corners.
    """
    rng = np.random.default_rng(seed)
    size = (int(70 * PX_PER_FT), int(114 * PX_PER_FT))
    texture = (np.full(size, 150, np.uint8) if sparse
               else cv2.GaussianBlur(rng.integers(60, 200, size, dtype=np.uint8), (0, 0), 1.2))
    for _ in range(250 if sparse else 400):
        x, y = rng.integers(0, size[1]), rng.integers(0, size[0])
        cv2.circle(texture, (int(x), int(y)), int(rng.integers(2, 6)), int(rng.integers(0, 255)), -1)
    # texture pixel -> court feet: x from -35, y from -10
    to_court = np.array([[1 / PX_PER_FT, 0., -35.], [0., 1 / PX_PER_FT, -10.], [0., 0., 1.]])
    return texture, to_court


def write_video(path, frames=40, blank_after=None, overlay=False, cut_at=None, sparse=False):
    texture, to_court = floor_texture(sparse=sparse)
    graphic = np.random.default_rng(5).integers(0, 255, (140, 520), dtype=np.uint8)
    other, _ = floor_texture(seed=1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for frame in range(frames):
        source = other if cut_at is not None and frame >= cut_at else texture
        image = cv2.warpPerspective(source, camera(frame) @ to_court, (W, H), borderValue=30)
        if blank_after is not None and frame > blank_after:
            image[:] = 128
        if overlay == "large":  # a busy static graphic covering much of the floor
            image[380:520, 420:940] = graphic
        elif overlay:  # a static score bug over the floor
            cv2.rectangle(image, (700, 440), (940, 520), 20, -1)
            for i in range(12):
                cv2.putText(image, str(i % 10), (710 + 19 * i, 495), cv2.FONT_HERSHEY_SIMPLEX, .9, 250, 2)
        writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    writer.release()


def worst_error_ft(mappings, frames):
    probes = [(-10., 20.), (5., 30.), (15., 15.), (0., 40.)]
    worst = 0.
    for frame in frames:
        truth = np.linalg.inv(camera(frame))
        for probe in probes:
            pixel = apply(camera(frame), [probe])
            if not (0 <= pixel[0, 0] < W and 0 <= pixel[0, 1] < H):
                continue
            worst = max(worst, float(np.linalg.norm(apply(mappings[frame].image_to_court, pixel)[0]
                                                    - apply(truth, pixel)[0])))
    return worst


@pytest.fixture(scope="module")
def pan_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("pan") / "pan.mp4"
    write_video(path, overlay=True)
    return path


def test_panning_zooming_camera_is_followed_both_ways_from_the_anchor(pan_video):
    frames = list(range(40))
    anchor = 15
    mappings = follow_court(pan_video, frames, anchor, np.linalg.inv(camera(anchor)), template("nba"),
                            {}, [], W, H, 40)
    assert all(mappings[f].reliable for f in frames)
    assert mappings[anchor].status == "anchor"
    # About 24 ft of pan and 16% zoom; a static score bug sits over the floor.
    assert worst_error_ft(mappings, frames) < .5


def test_static_graphics_cannot_pin_the_floor_to_the_screen(tmp_path):
    # On a flat floor the graphic has more corners than the floor; left in, it drags the
    # mapping to follow the screen instead of the court (12.6 ft error before masking).
    path = tmp_path / "graphic.mp4"
    write_video(path, overlay="large", sparse=True)
    mappings = follow_court(path, list(range(40)), 15, np.linalg.inv(camera(15)), template("nba"), {}, [], W, H, 40)
    assert worst_error_ft(mappings, range(40)) < .5


def test_mapping_stops_at_a_cut(tmp_path):
    path = tmp_path / "cut.mp4"
    write_video(path, frames=30, cut_at=20)
    mappings = follow_court(path, list(range(30)), 5, np.linalg.inv(camera(5)), template("nba"), {}, [20], W, H, 30)
    assert all(mappings[f].reliable for f in range(20))
    assert {mappings[f].status for f in range(20, 30)} == {"cut"}
    assert all(mappings[f].image_to_court is None for f in range(20, 30))


def test_tracking_loss_withholds_later_frames(tmp_path):
    path = tmp_path / "blank.mp4"
    write_video(path, frames=30, blank_after=20)
    mappings = follow_court(path, list(range(30)), 0, np.linalg.inv(camera(0)), template("nba"), {}, [], W, H, 30)
    assert all(mappings[f].reliable for f in range(21))
    assert all(mappings[f].status == "lost" and not mappings[f].reliable for f in range(21, 30))


def test_fixed_cameras_keep_the_marked_mapping(pan_video):
    anchor_h = np.linalg.inv(camera(0))
    mappings = follow_court(pan_video, [0, 10, 20], 0, anchor_h, template("nba"), {}, [], W, H, 40, fixed=True)
    assert [m.status for m in mappings.values()] == ["anchor", "fixed", "fixed"]
    assert all(np.array_equal(m.image_to_court, anchor_h) for m in mappings.values())


def test_court_map_steadies_points_against_camera_motion(pan_video):
    frames = list(range(0, 40, 2))
    ids = ["lane_base_left", "lane_base_right", "ft_left", "ft_right", "three_top"]
    court = template("nba")
    image = apply(camera(0), [court.landmarks[key] for key in ids])
    calibration = fit([(key, (x / W, y / H)) for key, (x, y) in zip(ids, image)], W, H)
    court_map = CourtMap(calibration, follow_court(pan_video, frames, 0, calibration.image_to_court, court,
                                                   {}, [], W, H, 40), 0, W, H)
    spot = (5., 30.)
    pixel = apply(camera(30), [spot])[0]
    assert court_map.to_court(30, pixel) == pytest.approx(spot, abs=.3)
    assert court_map.to_anchor_image(30, pixel) == pytest.approx(tuple(apply(camera(0), [spot])[0]), abs=2)
    assert court_map.camera(30).center == pytest.approx((-100., 40., 35.), abs=3)
    assert court_map.summary()["reliable_frames"] == len(frames)


def test_segment_bounds_and_anchor_frame():
    assert segment_bounds(15, [10, 18, 220], 300) == (10, 18)
    assert segment_bounds(5, [10], 300) == (0, 10)
    assert anchor_frame({"frame": None, "time_s": 1.0}, 29.97, 100) == 30
    assert anchor_frame({"frame": 500}, 30, 100) == 99


def test_person_boxes_cover_poses_and_detector_boxes():
    pose = SimpleNamespace(landmarks={"left_ankle": (.4, .8, .9), "nose": (.42, .4, .9)}, box=None)
    boxed = SimpleNamespace(landmarks={}, box=(.1, .2, .2, .6))
    boxes = person_boxes({"players": [pose, boxed], "unposed": [(.9, (.5, .5, .6, .9))],
                          "possession": [(.8, (.7, .1, .8, .4))]})
    assert boxes[1:] == [(.1, .2, .2, .6), (.5, .5, .6, .9), (.7, .1, .8, .4)]
    x1, y1, x2, y2 = boxes[0]
    assert x1 == .4 and x2 == .42 and y1 < .4 and y2 > .8
