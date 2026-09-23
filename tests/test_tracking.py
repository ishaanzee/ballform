import numpy as np

from app.models import Detection, PoseFrame
from app.tracking import BallHandlerTracker, HandlerDecision, PoseTracker, jersey_descriptor


def pose(x, frame=0, appearance=None):
    return PoseFrame(frame, frame / 30, {
        "left_shoulder": (x - .03, .35, .95),
        "right_shoulder": (x + .03, .35, .95),
        "left_hip": (x - .025, .60, .95),
        "right_hip": (x + .025, .60, .95),
    }, appearance=appearance)


def test_tracker_preserves_ids_when_detector_order_changes():
    tracker = PoseTracker(1., max_gap_frames=10)
    first = [pose(.25, appearance=(.2, .3, .4)), pose(.75, appearance=(.8, .6, .2))]
    tracker.update(first)
    first_ids = [item.track_id for item in first]
    second = [pose(.74, 1, (.8, .6, .2)), pose(.26, 1, (.2, .3, .4))]
    tracker.update(second)
    assert [item.track_id for item in second] == [first_ids[1], first_ids[0]]


def test_jersey_descriptor_reads_inset_torso_colour():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[60:130, 70:130] = (20, 40, 220)
    target = pose(.5)
    descriptor = jersey_descriptor(frame, target)
    assert descriptor is not None
    assert all(0 <= value <= 1 for value in descriptor)


def handler_pose(track_id, x, frame):
    player = pose(x, frame)
    player.track_id = track_id
    player.landmarks.update({
        "left_wrist": (x - .04, .49, .95),
        "right_wrist": (x + .04, .49, .95),
    })
    return player


def handler_ball(x, frame):
    return Detection(frame, frame / 30, x, .49, .9)


def test_handler_survives_brief_ball_occlusion_and_reacquires():
    tracker = BallHandlerTracker(1.)
    player = handler_pose(1, .3, 0)
    assert tracker.update([player], handler_ball(.34, 0), 0) == HandlerDecision(1, "observed")
    for frame in range(1, 9):
        assert tracker.update([handler_pose(1, .3, frame)], None, frame / 30) == HandlerDecision(1, "held")
    assert tracker.update([handler_pose(1, .3, 9)], handler_ball(.34, 9), .3) == HandlerDecision(1, "observed")


def test_handler_switch_needs_repeated_strong_evidence():
    tracker = BallHandlerTracker(1.)
    players = [handler_pose(1, .3, 0), handler_pose(2, .7, 0)]
    assert tracker.update(players, handler_ball(.34, 0), 0) == HandlerDecision(1, "observed")
    players = [handler_pose(1, .3, 1), handler_pose(2, .7, 1)]
    assert tracker.update(players, handler_ball(.74, 1), 1 / 30) == HandlerDecision(1, "held")
    # A single defender-adjacent ball sighting cannot steal the label.
    assert tracker.update(players, None, 2 / 30) == HandlerDecision(1, "held")
    assert tracker.update(players, handler_ball(.74, 3), 3 / 30) == HandlerDecision(1, "held")
    assert tracker.update(players, handler_ball(.74, 4), 4 / 30) == HandlerDecision(2, "observed")


def test_handler_hold_expires_and_camera_cut_clears_it():
    tracker = BallHandlerTracker(1.)
    player = handler_pose(1, .3, 0)
    tracker.update([player], handler_ball(.34, 0), 0)
    assert tracker.update([player], None, .7) == HandlerDecision(1, "held")
    assert tracker.update([player], None, .81) == HandlerDecision(None, "none")
    tracker.update([player], handler_ball(.34, 0), 0)
    tracker.reset()
    assert tracker.update([player], None, .1) == HandlerDecision(None, "none")


def test_visible_loose_ball_ends_handler_hold_sooner():
    tracker = BallHandlerTracker(1.)
    player = handler_pose(1, .3, 0)
    tracker.update([player], handler_ball(.34, 0), 0)
    assert tracker.update([player], handler_ball(.9, 1), .2) == HandlerDecision(1, "held")
    assert tracker.update([player], handler_ball(.9, 2), .36) == HandlerDecision(None, "none")


def test_handler_is_not_drawn_when_player_pose_is_missing():
    tracker = BallHandlerTracker(1.)
    player = handler_pose(1, .3, 0)
    tracker.update([player], handler_ball(.34, 0), 0)
    assert tracker.update([], None, 1 / 30) == HandlerDecision(None, "none")
    assert tracker.update([handler_pose(1, .3, 2)], None, 2 / 30) == HandlerDecision(1, "held")
