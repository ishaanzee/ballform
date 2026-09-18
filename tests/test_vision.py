import numpy as np
import pytest

from app.vision import Person, camera_profile, map_keypoints, merge_people, on_court, scene_cut, validate_court, is_player
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
