import math

import numpy as np
import pytest

from app import trajectory as T
from app.camera_motion import CourtMap, FrameMapping, follow_court
from app.court import apply, fit, template
from app.models import PoseFrame
from tests_support_court import synthetic_camera

W, H, FPS = 1920, 1080, 30.


def court_map(frames=120):
    court_to_image = synthetic_camera()[2]
    ids = ["lane_base_left", "lane_base_right", "ft_left", "ft_right", "three_top", "half_right", "half_left"]
    image = apply(court_to_image, [template("nba").landmarks[key] for key in ids])
    calibration = fit([(key, (x / W, y / H)) for key, (x, y) in zip(ids, image)], W, H)
    mappings = follow_court(None, list(range(frames)), 0, calibration.image_to_court, template("nba"), {}, [],
                            W, H, frames, fixed=True)
    return CourtMap(calibration, mappings, 0, W, H), court_to_image


def pose(frame, spot, court_to_image, track_id=1, lift=0., noise=(0., 0.), shift=(0., 0.), size=220.):
    """A standing player whose feet are on `spot`; lift raises the whole body (px), shift moves only the ankles."""
    u, v = apply(court_to_image, [spot])[0]
    u, v = u + noise[0], v + noise[1] - lift

    def at(dx, dy, extra=(0., 0.)):
        return ((u + dx * size + extra[0]) / W, (v + dy * size + extra[1]) / H, .95)
    landmarks = {"left_ankle": at(-.04, 0., shift), "right_ankle": at(.04, 0., shift),
                 "left_knee": at(-.04, -.25), "right_knee": at(.04, -.25), "left_hip": at(-.05, -.5),
                 "right_hip": at(.05, -.5), "left_shoulder": at(-.07, -.78), "right_shoulder": at(.07, -.78),
                 "nose": at(0., -.92)}
    box = ((u - .15 * size) / W, (v - .97 * size) / H, (u + .15 * size) / W, (v + .03 * size) / H)
    return PoseFrame(frame, frame / FPS, landmarks, track_id=track_id, box=box)


def frames_for(path, court_to_image, noise_px=0., seed=0, **per_frame):
    """One track following `path` (court feet per frame); per_frame maps keyword -> {frame: value}."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, spot in enumerate(path):
        if spot is None:
            frames.append({"frame": i, "time_s": i / FPS, "players": []})
            continue
        extra = {key: values[i] for key, values in per_frame.items() if i in values}
        noise = tuple(rng.normal(0., noise_px, 2)) if noise_px else (0., 0.)
        frames.append({"frame": i, "time_s": i / FPS,
                       "players": [pose(i, spot, court_to_image, noise=noise, **extra)]})
    return frames


def steps(trajectory):
    return np.linalg.norm(np.diff(trajectory.position, axis=0), axis=1) / np.diff(trajectory.times)


def raw_steps(frames, cmap):
    points = [cmap.to_court(f["frame"], T.floor_point(f["players"][0], W, H)[0]) for f in frames if f["players"]]
    return np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1) * FPS


def test_jacobian_matches_finite_differences():
    cmap, court_to_image = court_map(2)
    h = cmap.frames[0].image_to_court
    pixel = apply(court_to_image, [(-5., 28.)])[0]
    numeric = np.column_stack([(apply(h, [pixel + d])[0] - apply(h, [pixel - d])[0]) / 2e-3
                               for d in (np.array([1e-3, 0.]), np.array([0., 1e-3]))])
    assert T.homography_jacobian(h, pixel) == pytest.approx(numeric, rel=1e-4)


def test_measurement_noise_in_feet_grows_away_from_the_camera():
    cmap, court_to_image = court_map(2)
    near = T.measure(pose(0, (0., 10.), court_to_image), 0, 0., cmap)
    far = T.measure(pose(0, (20., 10.), court_to_image), 0, 0., cmap)  # the camera is at x = -100
    assert near.method == far.method == "ankles"
    assert np.sqrt(np.linalg.eigvalsh(far.cov)[-1]) > np.sqrt(np.linalg.eigvalsh(near.cov)[-1])


def test_jitter_is_smoothed_into_a_steady_run():
    cmap, court_to_image = court_map(90)
    truth = [(-10. + 12. * i / FPS, 25.) for i in range(90)]
    frames = frames_for(truth, court_to_image, noise_px=2.5, seed=1)
    (trajectory,) = T.floor_trajectories(frames, cmap)
    assert raw_steps(frames, cmap).max() > 20  # a few pixels of jitter look like sprinting
    middle = slice(10, -10)
    assert steps(trajectory).max() < 18
    assert trajectory.speed[middle] == pytest.approx(12., abs=1.5)
    assert np.abs(trajectory.position - np.asarray(truth)).max() < .5


def test_isolated_outliers_are_down_weighted():
    cmap, court_to_image = court_map(60)
    spot = (-4., 22.)
    bad = {20: (60., -40.), 21: (-50., 30.), 40: (0., -70.)}  # ankles on someone else for single frames
    frames = frames_for([spot] * 60, court_to_image, shift=bad)
    (trajectory,) = T.floor_trajectories(frames, cmap)
    assert np.linalg.norm(trajectory.position - spot, axis=1).max() < .3
    assert trajectory.speed.max() < 3
    assert all(trajectory.source[i] in "gj" for i in bad)


def test_airborne_frames_are_bridged_not_mapped_beyond_the_player():
    cmap, court_to_image = court_map(60)
    spot = (-6., 28.)
    # A 0.5 s jump in place: the whole body rises up to 60 px.
    lift = {i: 60. * math.sin(math.pi * (i - 20) / 15) for i in range(20, 36)}
    frames = frames_for([spot] * 60, court_to_image, lift=lift)
    lifted = T.floor_point(frames[27]["players"][0], W, H)[0]
    assert math.dist(cmap.to_court(27, lifted), spot) > 3  # what an unsmoothed floor point would say
    (trajectory,) = T.floor_trajectories(frames, cmap)
    assert "j" in "".join(trajectory.source[22:34])
    assert np.linalg.norm(trajectory.position - spot, axis=1).max() < .5
    assert trajectory.speed.max() < 3


def test_short_gap_is_bridged_and_long_gap_breaks():
    cmap, court_to_image = court_map(90)
    truth = [(-10. + 8. * i / FPS, 25.) for i in range(90)]
    short = [None if 30 <= i < 39 else p for i, p in enumerate(truth)]  # 0.3 s
    (bridged,) = T.floor_trajectories(frames_for(short, court_to_image), cmap)
    assert bridged.frames == list(range(90)) and set(bridged.source[30:39]) == {"-"}
    assert np.abs(bridged.position - np.asarray(truth)).max() < .3
    long = [None if 30 <= i < 60 else p for i, p in enumerate(truth)]  # 1 s
    pieces = T.floor_trajectories(frames_for(long, court_to_image), cmap)
    assert [(t.frames[0], t.frames[-1]) for t in pieces] == [(0, 29), (60, 89)]


def test_trajectories_break_at_cuts_and_unreliable_mapping():
    cmap, court_to_image = court_map(90)
    frames = frames_for([(0., 20. + 5. * i / FPS) for i in range(90)], court_to_image)
    pieces = T.floor_trajectories(frames, cmap, cuts=[45])
    assert [(t.frames[0], t.frames[-1]) for t in pieces] == [(0, 44), (45, 89)]
    cmap.frames[60] = FrameMapping(60, None, "lost")
    pieces = T.floor_trajectories(frames, cmap, cuts=[45])
    assert [(t.frames[0], t.frames[-1]) for t in pieces] == [(0, 44), (45, 59), (61, 89)]


def test_a_track_that_switches_players_is_split_not_smeared():
    cmap, court_to_image = court_map(60)
    path = [(-5., 25.) if i < 30 else (5., 25.) for i in range(60)]  # the ID jumps to a player 10 ft away
    pieces = T.floor_trajectories(frames_for(path, court_to_image), cmap)
    assert len(pieces) == 2
    assert pieces[0].frames[-1] < 30 <= pieces[1].frames[0]
    assert all(steps(t).max() < 5 for t in pieces)


def test_a_sharp_stop_is_blurred_by_about_a_fifth_of_a_second():
    cmap, court_to_image = court_map(90)
    path = [(-10. + 20. * min(i, 45) / FPS, 25.) for i in range(90)]
    (trajectory,) = T.floor_trajectories(frames_for(path, court_to_image), cmap)
    fast = np.flatnonzero(trajectory.speed >= 18)
    slow = np.flatnonzero(trajectory.speed <= 2)
    assert fast[fast < 45].max() >= 45 - .15 * FPS  # still at 90% of the speed 0.15 s before the stop
    assert slow[slow > 45].min() <= 45 + .15 * FPS  # and under 10% within 0.15 s after it


def test_trajectories_json_is_compact_and_complete():
    cmap, court_to_image = court_map(30)
    frames = frames_for([(0., 20. + 5. * i / FPS) for i in range(30)], court_to_image)
    data = T.trajectories_json(T.floor_trajectories(frames, cmap))
    (track,) = data["tracks"]
    assert track["track_id"] == 1 and track["frames"] == list(range(30))
    assert len(track["x"]) == len(track["vy"]) == len(track["sd"]) == len(track["source"]) == 30
    assert set(track["source"]) <= set(data["source_codes"])
    assert track["vy"][15] == pytest.approx(5., abs=.3)
