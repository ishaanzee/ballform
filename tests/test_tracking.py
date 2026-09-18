import numpy as np

from app.models import PoseFrame
from app.tracking import PoseTracker, jersey_descriptor


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
