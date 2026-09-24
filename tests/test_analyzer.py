from concurrent.futures import ThreadPoolExecutor

from app.analyzer import _review_highlight, _sampled_frames, _shooter_intervals, _wait_summary
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
    waits = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        frames = _sampled_frames(capture, 2, Detector(), executor, waits)
        first = next(frames)
        assert first[:3] == (0, 0, [0])
        assert capture.index == 3  # Only one analyzed frame is buffered ahead.
        rest = list(frames)
    assert [(frame_no, frame, objects) for frame_no, frame, objects, _ in rest] == [
        (2, 2, [20]), (4, 4, [40])]
    assert all(seconds >= 0 for _, _, _, seconds in [first, *rest])
    assert set(waits) == {0, 2, 4}
    assert all(seconds >= 0 for seconds in waits.values())


def test_pipeline_wait_summary_reports_blocked_time_and_tail():
    assert _wait_summary([])["frames"] == 0
    summary = _wait_summary([0, .002, .004, .010])
    assert summary["frames"] == 4
    assert summary["blocked_frames_over_1ms"] == 3
    assert summary["total_seconds"] == .016
    assert summary["median_ms"] == 3.0
    assert summary["max_ms"] == 10.0


def test_serial_frame_reader_does_not_run_detector():
    class Capture:
        def __init__(self):
            self.frames = iter([0, 1, 2, 3])

        def read(self):
            frame = next(self.frames, None)
            return (frame is not None, frame)

    assert list(_sampled_frames(Capture(), 2)) == [(0, 0, None, 0.0), (2, 2, None, 0.0)]


def test_shared_model_loads_once_and_does_not_cache_failures(monkeypatch):
    import pytest
    import app.analyzer as analyzer
    monkeypatch.setattr(analyzer, "_MODELS", {})
    calls = []
    loaded = []
    assert analyzer._shared_model(("pose", "a"), lambda: calls.append(1) or "model", loaded) == "model"
    assert analyzer._shared_model(("pose", "a"), lambda: calls.append(1) or "other", loaded) == "model"
    assert calls == [1] and loaded == ["pose"]

    def broken():
        raise RuntimeError("load failed")
    with pytest.raises(RuntimeError):
        analyzer._shared_model(("ball", "b"), broken)
    assert analyzer._shared_model(("ball", "b"), lambda: "retry") == "retry"


def test_model_checksum_is_skipped_until_file_changes(monkeypatch, tmp_path):
    import hashlib
    import os
    import app.analyzer as analyzer
    monkeypatch.setattr(analyzer, "_VERIFIED_MODELS", {})
    model = tmp_path / "model.onnx"
    model.write_bytes(b"a" * 1_000_001)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    hashes = []
    real = hashlib.file_digest
    monkeypatch.setattr(analyzer.hashlib, "file_digest", lambda *a: hashes.append(1) or real(*a))
    analyzer._ensure_model(model, "unused", digest)
    analyzer._ensure_model(model, "unused", digest)
    assert len(hashes) == 1
    stat = model.stat()
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    analyzer._ensure_model(model, "unused", digest)
    assert len(hashes) == 2
