import numpy as np
import pytest
from types import SimpleNamespace
from threading import Event

import app.vision as vision_module

from app.vision import CourtVision, CutDetector, Person, camera_profile, map_keypoints, merge_people, on_court, scene_cut, validate_court, is_player
from app.basketball import decode, preprocess
from app.game import _body, _appearance_groups
from app.models import PoseFrame
from app.tracking import PoseTracker
from app.scoring import _release_proximity
from app.models import Detection


def test_crop_keypoints_map_back_to_original_frame():
    points = np.zeros((17, 3))
    points[9] = (100, 200, .82)
    mapped = map_keypoints(points, (800, 0, 1920, 1080), 1920, 1080)
    assert mapped[15] == pytest.approx((900/1920, 200/1080, .82))


def test_overlapping_crop_detections_merge_without_losing_nearby_player():
    a = Person((.2, .3, .3, .6), .95, {})
    duplicate = Person((.201, .3, .301, .6), .9, {})
    neighbor = Person((.27, .31, .38, .61), .9, {})
    assert merge_people([duplicate, neighbor, a]) == [a, neighbor]


def test_side_crops_keep_original_image_size_and_coordinates():
    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    empty = SimpleNamespace(boxes=None, keypoints=None)
    points = np.zeros((1, 17, 3))
    points[0, :, 2] = .9
    points[0, 5] = (220, 200, .9)
    points[0, 6] = (260, 200, .9)
    points[0, 11] = (220, 260, .9)
    points[0, 12] = (260, 260, .9)
    left_player = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=Tensor([[200, 180, 280, 300]]), conf=Tensor([.9])),
        keypoints=SimpleNamespace(data=Tensor(points)))

    class PoseModel:
        def __init__(self):
            self.calls = []

        def predict(self, source, **kwargs):
            self.calls.append((source, kwargs))
            return [left_player] if len(self.calls) == 2 else [empty]

    class BallModel:
        def predict(self, source, **kwargs):
            return [empty]

    pose_model = PoseModel()
    vision = CourtVision(pose_model, BallModel(), "mps", "moving")
    poses, _, ball = vision.detect(np.zeros((600, 1000, 3), dtype=np.uint8), 0, 0.0)

    assert len(pose_model.calls) == 3
    assert pose_model.calls[0][0].shape == (600, 1000, 3)
    assert pose_model.calls[0][1]["imgsz"] == 1280
    assert [source.shape for source, _ in pose_model.calls[1:]] == [(600, 600, 3)] * 2
    assert [kwargs["imgsz"] for _, kwargs in pose_model.calls[1:]] == [960, 960]
    assert len(poses) == 1 and ball is None
    assert poses[0].landmarks["left_shoulder"][:2] == pytest.approx((.22, 1/3))
    assert vision.pose_predict_calls == 3
    assert vision.pose_images == 3
    assert vision.ball_crop_overlap_frames == 0


def test_cpu_ball_crop_detection_overlaps_pose_without_skipping_crops(monkeypatch):
    pose_started = Event()

    class BallDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            if self.calls == 1:
                return [(4, .9, (.2, .2, .3, .6))]
            assert pose_started.wait(1), "Side ball detection ran before pose inference"
            return [(1, .8, (.45, .45, .55, .55))]

    class PoseModel:
        def predict(self, source, **kwargs):
            pose_started.set()
            return [SimpleNamespace(boxes=None, keypoints=None)]

    monkeypatch.setattr(vision_module, "BasketballDetector", BallDetector)
    detector = BallDetector()
    vision = CourtVision(PoseModel(), detector, "mps", "moving")
    try:
        _, _, ball = vision.detect(np.zeros((600, 1000, 3), dtype=np.uint8), 0, 0.0)
    finally:
        vision.close()
    assert detector.calls == 3
    assert vision.ball_crop_overlap_frames == 1
    assert vision.pose_images == 3
    assert ball is not None and ball.x == pytest.approx(.3)


def test_prefetched_full_frame_ball_objects_are_used_without_redetection(monkeypatch):
    class BallDetector:
        def detect(self, frame):
            raise AssertionError("Full-frame basketball detection should be prefetched")

    class PoseModel:
        def predict(self, source, **kwargs):
            return [SimpleNamespace(boxes=None, keypoints=None)]

    monkeypatch.setattr(vision_module, "BasketballDetector", BallDetector)
    vision = CourtVision(PoseModel(), BallDetector(), "mps", "moving")
    objects = [(4, .9, (.2, .2, .3, .6)), (1, .9, (.45, .45, .55, .55))]
    _, _, ball = vision.detect(np.zeros((600, 1000, 3), dtype=np.uint8), 0, 0.0,
                               prefetched_objects=objects)
    assert ball is not None and ball.confidence == pytest.approx(.9)
    assert vision.ball_crop_overlap_frames == 0


def test_court_filters_feet_not_head():
    person = Person((.4, .1, .6, .8), .9, {27: (.45, .8, .9), 28: (.55, .8, .9)})
    assert on_court(person, [[.1, .5], [.9, .5], [.9, .9], [.1, .9]])
    assert not on_court(person, [[0, 0], [1, 0], [1, .5], [0, .5]])


@pytest.mark.parametrize('polygon', [[[0,0],[1,1]], [[0,0],[1,0],[float('nan'),1]],
                                   [[0,0],[1,1],[0,1],[1,0]], [[0,0],[2,0],[1,1]]])
def test_invalid_court(polygon):
    with pytest.raises(ValueError):
        validate_court(polygon)


def small_pose(frame):
    return PoseFrame(frame, frame/30, {
        'left_shoulder': (.49, .4, .9), 'right_shoulder': (.51, .4, .9),
        'left_hip': (.49, .42, .9), 'right_hip': (.51, .42, .9)})


def test_distant_player_is_measurable_and_track_ids_never_cross_cut():
    a, b = small_pose(0), small_pose(1)
    assert _body(a, 16/9, min_torso=10/1080) is not None
    tracker = PoseTracker(16/9)
    tracker.update([a])
    tracker.reset()
    tracker.update([b])
    assert a.track_id is not None and b.track_id != a.track_id


def test_third_uniform_group_is_not_forced_onto_opposing_team():
    players = [PoseFrame(0, 0, {}, appearance=color) for color in
               [(0.1, .65, .35)]*5 + [(.8, .5, .5)]*5 + [(.45, .5, .5)]*3]
    groups = _appearance_groups(players, 0)
    assert groups is not None
    assignment, _ = groups
    assert all(assignment[i] != assignment[0] for i in range(5, 10))
    assert all(i not in assignment for i in range(10, 13))


def test_bright_white_jersey_stays_with_white_team():
    players = [PoseFrame(0, 0, {}, appearance=color) for color in
               [(0.15,.63,.37)]*5 + [(.64,.58,.51)]*4 + [(.78,.56,.50)]]
    assignment, _ = _appearance_groups(players, 0)
    assert assignment[9] == assignment[5] != assignment[0]
    bright_shooter, _ = _appearance_groups(players, 9)
    assert bright_shooter[9] == bright_shooter[5] != bright_shooter[0]


def test_cut_detection_and_profiles():
    old = np.zeros((100,100), dtype=np.uint8)
    assert not scene_cut(None, old)
    assert not scene_cut(old, old+5)
    assert scene_cut(old, old+200)
    assert camera_profile('auto', 'one_on_one') == 'broadcast'
    assert camera_profile('auto', 'form') == 'courtside'
    assert camera_profile('moving', 'one_on_one') == 'moving'


def test_single_frame_flash_is_not_a_cut_but_a_persistent_edit_is():
    old = np.full((100, 100), 100, dtype=np.uint8)
    flash = np.full((100, 100), 160, dtype=np.uint8)
    new = np.full((100, 100), 200, dtype=np.uint8)
    detector = CutDetector()
    assert [detector.observe(i, frame) for i, frame in
            enumerate((old, flash, old, new, new))] == [None, None, None, None, 3]


def test_basketball_detector_rgb_normalization_and_box_decode():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # BGR red -> RGB first channel.
    tensor = preprocess(frame)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor[0, :, 0, 0] == pytest.approx([(1-.485)/.229, -.456/.224, -.406/.225])
    boxes = np.zeros((1, 300, 4))
    logits = np.full((1, 300, 11), -20.)
    boxes[0, 0] = [.5, .4, .1, .2]
    logits[0, 0, 1] = 3.
    result = decode(boxes, logits)
    assert len(result) == 1
    assert result[0][0] == 1
    assert result[0][2] == pytest.approx((.45, .3, .55, .5))


def test_broadcast_role_matching_excludes_referee_and_crowd():
    player = Person((.2, .2, .3, .6), .9, {})
    referee = Person((.5, .2, .6, .6), .9, {})
    spectator = Person((.8, .2, .9, .4), .9, {})
    objects = [(4, .9, player.box), (9, .9, referee.box)]
    assert is_player(player, objects)
    assert not is_player(referee, objects)
    assert not is_player(spectator, objects)


def test_game_release_rejects_dribble_and_checks_left_hand():
    pose = small_pose(0)
    pose.landmarks['left_wrist'] = (.50, .43, .95)
    assert _release_proximity(pose, Detection(0, 0, .50, .43, .9), 'left', 1., True) is None
    pose.landmarks['left_wrist'] = (.50, .38, .95)
    assert _release_proximity(pose, Detection(0, 0, .50, .38, .9), 'left', 1., True) == 0
