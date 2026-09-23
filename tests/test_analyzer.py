from concurrent.futures import ThreadPoolExecutor

from app.analyzer import _review_highlight, _sampled_frames, _shooter_intervals
from app.models import ShotResult
from app.tracking import HandlerDecision


def test_shooter_overlay_runs_from_release_through_rim_event():
    shot = ShotResult(
        number=1, start_s=1.5, release_s=2.0, end_s=3.5, outcome="made",
        outcome_confidence=.8, evidence=[], metrics={}, outcome_frame=82,
        game={"release_frame": 60, "players": {"shooter_track_id": 7}},
    )

    assert _shooter_intervals([shot], 30.0) == [(60, 82, 7)]


def test_shooter_overlay_falls_back_to_detected_arc_end():
    shot = ShotResult(
        number=1, start_s=1.5, release_s=2.0, end_s=3.5, outcome="unknown",
        outcome_confidence=.2, evidence=[], metrics={},
        game={"release_frame": 60, "players": {"shooter_track_id": 7}},
    )

    assert _shooter_intervals([shot], 30.0) == [(60, 105, 7)]


def test_shooter_highlight_takes_priority_over_inferred_handler():
    frame = {"handler": HandlerDecision(3, "held")}
    intervals = [(60, 82, 7)]
    assert _review_highlight(59, frame, intervals) == (3, "P3 BALL?")
    assert _review_highlight(60, frame, intervals) == (7, "P7 SHOOTER")
    assert _review_highlight(82, frame, intervals) == (7, "P7 SHOOTER")
    assert _review_highlight(83, frame, intervals) == (3, "P3 BALL?")


def test_frame_prefetch_preserves_source_order_and_stride():
    class Capture:
        def __init__(self):
            self.index = 0

        def read(self):
            if self.index == 6:
                return False, None
            frame = self.index
            self.index += 1
            return True, frame

    class Detector:
        def detect(self, frame):
            return [frame * 10]

    capture = Capture()
    with ThreadPoolExecutor(max_workers=1) as executor:
        frames = _sampled_frames(capture, 2, Detector(), executor)
        first = next(frames)
        assert first[:3] == (0, 0, [0])
        assert capture.index == 3  # Only one analyzed frame is buffered ahead.
        rest = list(frames)
    assert [(frame_no, frame, objects) for frame_no, frame, objects, _ in rest] == [
        (2, 2, [20]), (4, 4, [40])]
    assert all(seconds >= 0 for _, _, _, seconds in [first, *rest])


def test_serial_frame_reader_does_not_run_detector():
    class Capture:
        def __init__(self):
            self.frames = iter([0, 1, 2, 3])

        def read(self):
            frame = next(self.frames, None)
            return (frame is not None, frame)

    assert list(_sampled_frames(Capture(), 2)) == [(0, 0, None, 0.0), (2, 2, None, 0.0)]
