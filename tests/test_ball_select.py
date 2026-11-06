from types import SimpleNamespace

import numpy as np
import pytest

import app.vision as vision_module
from app.ball_select import CandidateCourtVision, StaticBallFilter, candidates_from, select, wrists_of
from app.models import Detection, PoseFrame
from app.pose import TorchPose

FPS = 30.0
SIZE = (1000, 600)


def ball(frame, x, y, confidence=.8, radius=.01):
    return Detection(frame, frame / FPS, x, y, confidence, radius)


def player(wrist):
    """A pose with a 100 px torso and both wrists at ``wrist`` (normalized)."""
    marks = {"left_shoulder": (.5, .3, .9), "right_shoulder": (.52, .3, .9),
             "left_hip": (.5, .3 + 100 / SIZE[1], .9), "right_hip": (.52, .3 + 100 / SIZE[1], .9),
             "left_wrist": (*wrist, .9), "right_wrist": (*wrist, .9)}
    return PoseFrame(0, 0.0, marks)


def still_filter(frames=12, at=(.3, .05), wrists=()):
    """A filter that has seen a candidate at ``at`` on every one of ``frames`` frames."""
    static = StaticBallFilter()
    static.size = SIZE
    for frame in range(frames):
        static.observe(frame / FPS, [ball(frame, *at)], list(wrists))
    return static


def test_still_hand_free_candidate_is_static_after_three_tenths_of_a_second():
    static = still_filter(frames=10)
    assert static.is_static(ball(10, .3, .05), 10 / FPS)
    assert not still_filter(frames=8).is_static(ball(8, .3, .05), 8 / FPS)
    # A candidate a ball radius away is a different object.
    assert not static.is_static(ball(10, .3 + .01, .05), 10 / FPS)


def test_short_detection_gaps_do_not_break_a_static_run():
    static = StaticBallFilter()
    static.size = SIZE
    for frame in range(12):
        static.observe(frame / FPS, [] if frame in (4, 5) else [ball(frame, .3, .05)], [])
    assert static.is_static(ball(12, .3, .05), 12 / FPS)
    static.reset()
    assert not static.is_static(ball(12, .3, .05), 12 / FPS)


def test_ball_held_near_a_wrist_or_at_the_rim_is_never_static():
    hand = wrists_of([player((.3, .05))], *SIZE)
    assert hand and hand[0][2] == pytest.approx(100)
    held = still_filter(wrists=hand)
    assert not held.is_static(ball(12, .3, .05), 12 / FPS, hand)
    rim = (.28, .04, .32, .06)
    assert not still_filter().is_static(ball(12, .3, .05), 12 / FPS, [], [rim])
    assert not still_filter().is_static(ball(12, .3, .05), 12 / FPS, [], [], category=2)
    far_hand = wrists_of([player((.6, .6))], *SIZE)
    assert still_filter(wrists=far_hand).is_static(ball(12, .3, .05), 12 / FPS, far_hand)


def test_static_candidate_loses_to_a_moving_one_and_does_not_gate_it():
    static = still_filter()
    lock = ball(11, .3, .05, confidence=.85)
    real = ball(12, .7, .4, confidence=.6)
    found = [(ball(12, .3, .05, confidence=.85), 0, 1), (real, 1, 1)]
    # The frame loop's gate keeps the lock: the real ball jumped too far for its confidence.
    assert select([f[0] for f in found], lock, None, 12, 12 / FPS).x == pytest.approx(.3)
    assert static.choose(found, lock, None, 12, 12 / FPS, [], []) is real
    # Nothing but the still object: no ball rather than the lock.
    assert static.choose(found[:1], lock, None, 12, 12 / FPS, [], []) is None


def test_select_matches_the_frame_loop_gate():
    previous, prior = ball(10, .5, .5, .7), ball(9, .48, .5, .7)
    near, far = ball(11, .53, .5, .5), ball(11, .9, .1, .8)
    assert select([far, near], previous, prior, 11, 11 / FPS) is near
    assert select([ball(11, .9, .1, .9), near], previous, prior, 11, 11 / FPS).x == pytest.approx(.53)
    assert select([far], None, None, 11, 11 / FPS) is far
    assert select([], previous, prior, 11, 11 / FPS) is None


def test_candidates_map_side_crops_like_the_frame_loop():
    found = candidates_from([(0, 0, 1000, 600, [(1, .9, (.45, .45, .55, .55)), (4, .9, (.1, .1, .2, .2))]),
                             (400, 0, 1000, 600, [(1, .5, (.0, .5, .1, .6))])], 1000, 600, 3, .1)
    assert [(index, category) for _, index, category in found] == [(0, 1), (1, 1)]
    assert found[0][0].x == pytest.approx(.5) and found[0][0].radius == pytest.approx(.05)
    assert found[1][0].x == pytest.approx(.43) and found[1][0].y == pytest.approx(.55)


class _Pose:
    def predict(self, source, **kwargs):
        return [SimpleNamespace(boxes=None, keypoints=None)]


def _vision(monkeypatch, detector, static_filter=None):
    monkeypatch.setattr(vision_module, "BasketballDetector", type(detector))
    return CandidateCourtVision(TorchPose(_Pose(), "mps"), detector, "mps", "courtside",
                                static_filter=static_filter)


def test_candidates_are_kept_and_the_choice_is_unchanged_without_the_filter(monkeypatch):
    class Detector:
        def detect(self, frame):
            return [(1, .5, (.45, .45, .55, .55))]

    objects = [(4, .9, (.2, .2, .3, .6)), (1, .6, (.1, .1, .12, .12))]
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    previous = ball(0, .3, .5, .7)
    vision = _vision(monkeypatch, Detector())
    plain = vision_module.CourtVision(TorchPose(_Pose(), "mps"), Detector(), "mps", "courtside")
    try:
        _, _, chosen = vision.detect(frame, 1, 1 / FPS, previous, prefetched_objects=objects)
        _, _, expected = plain.detect(frame, 1, 1 / FPS, previous, prefetched_objects=objects)
    finally:
        vision.close()
        plain.close()
    # Full frame, then both side crops (no full-frame ball reached .65).
    assert [c[4] for c in vision.ball_candidates] == [0, 1, 2]
    assert chosen == expected and chosen.x == pytest.approx(.3)
    assert vision.forced_side_crop_frames == 0


def test_a_static_full_frame_ball_runs_the_side_crops(monkeypatch):
    class Detector:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            return [(1, .6, (.45, .45, .55, .55))] if frame.shape[1] == 600 else []

    detector = Detector()
    static = StaticBallFilter()
    vision = _vision(monkeypatch, detector, static)
    lock = (1, .9, (.295, .04, .305, .06))
    frame = np.zeros((600, 1000, 3), dtype=np.uint8)
    try:
        for frame_no in range(12):
            vision.detect(frame, frame_no, frame_no / FPS, prefetched_objects=[(4, .9, (.6, .2, .7, .6)), lock])
            if frame_no == 8:
                # Not still for 0.3 s yet: the confident full-frame ball skips the side crops.
                assert vision.forced_side_crop_frames == 0 and detector.calls == 0
        _, _, chosen = vision.detect(frame, 12, 12 / FPS, ball(11, .3, .05, .9), ball(10, .3, .05, .9),
                                     prefetched_objects=[(4, .9, (.6, .2, .7, .6)), lock])
    finally:
        vision.close()
    assert vision.forced_side_crop_frames >= 1 and detector.calls >= 2
    assert chosen is not None and chosen.confidence == pytest.approx(.6)
    assert vision.static_choice_frames >= 1


def test_saved_candidates_round_trip_and_old_runs_without_them_still_score(tmp_path):
    from app.analyzer import load_scoring_inputs, save_scoring_inputs, score_clip

    old = {"fps": 30.0, "width": 1000, "height": 600, "total": 30, "frame_no": 29, "analyzed_fps": 30.0,
           "stride": 1, "mode": "form", "handedness": "right", "profile": "courtside", "rim": None,
           "rim_marked": False, "auto_rim": False, "tracked_rims": {}, "cut_frames": [], "balls": [],
           "poses": [], "player_frames": [], "flows": {}, "flow_rims": {}, "calibration": None,
           "court_landmarks": None, "game_mode": False}
    assert score_clip(dict(old), None, tmp_path / "input.mp4")["shots"] == []
    path = tmp_path / "scoring_inputs.pkl.gz"
    save_scoring_inputs({**old, "ball_candidates": {3: [(.5, .5, .9, .01, 0), (.43, .55, .5, .01, 1)]}}, path)
    saved = load_scoring_inputs(path)
    assert saved["ball_candidates"][3][1] == (.43, .55, .5, .01, 1)
    assert score_clip(saved, None, tmp_path / "input.mp4")["shots"] == []
