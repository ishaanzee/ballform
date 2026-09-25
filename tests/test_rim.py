import cv2
import numpy as np
import pytest

from app.rim import detect_fixed_rim, rim_candidates, track_detected_rims


def rim(x, y=.3, w=.03, h=.01, conf=.85):
    return conf, (x, y, x + w, y + h)


def test_dominant_hoop_is_kept_and_a_false_positive_blip_dropped():
    detections = {f: [rim(.30 + .002 * f)] for f in range(0, 40)}
    detections[20].append(rim(.60, y=.35, conf=.6))  # a sleeve near the baseline, once
    rims, stats = track_detected_rims(detections, [])
    assert sorted(rims) == list(range(40))
    assert rims[20][0] == pytest.approx(.34, abs=.002)
    assert stats["other_hoop_tracks"] == 1


def test_panning_hoop_is_followed_and_jitter_is_smoothed():
    rng = np.random.default_rng(0)
    detections = {f: [rim(.2 + .004 * f + rng.normal(0, .0015))] for f in range(0, 60)}
    rims, _ = track_detected_rims(detections, [])
    residual = [abs(rims[f][0] - (.2 + .004 * f)) for f in range(2, 58)]
    raw = [abs(detections[f][0][1][0] - (.2 + .004 * f)) for f in range(2, 58)]
    assert np.mean(residual) < .7 * np.mean(raw)


def test_each_camera_segment_picks_its_own_hoop():
    detections = {f: [rim(.2)] for f in range(0, 30)}
    detections.update({f: [rim(.7)] for f in range(30, 60)})
    rims, stats = track_detected_rims(detections, [30])
    assert rims[10][0] == pytest.approx(.2) and rims[45][0] == pytest.approx(.7)
    assert stats["segments_with_rim"] == 2 and stats["other_hoop_tracks"] == 0


def test_segments_without_a_steady_hoop_get_no_rim():
    detections = {f: [] for f in range(0, 20)}
    detections[5] = [rim(.4)]
    detections[6] = [rim(.4)]
    rims, stats = track_detected_rims(detections, [])
    assert rims == {} and stats["segments_with_rim"] == 0


def test_rim_candidates_keep_confident_rims_only():
    objects = [(10, .9, (.1, .1, .2, .15)), (10, .3, (.5, .5, .6, .55)), (1, .9, (.1, .1, .12, .12))]
    assert rim_candidates(objects) == [(.9, (.1, .1, .2, .15))]


class FixedDetector:
    def __init__(self, seen_every=1):
        self.calls, self.seen_every = 0, seen_every

    def detect(self, frame):
        self.calls += 1
        return [(10, .9, (.4, .2, .45, .22))] if self.calls % self.seen_every == 0 else []


def _clip(path, frames=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    return path


def test_fixed_rim_is_the_median_of_sampled_detections(tmp_path):
    box = detect_fixed_rim(_clip(tmp_path / "clip.mp4"), FixedDetector(), samples=10)
    assert box == pytest.approx((.4, .2, .05, .02))


def test_fixed_rim_needs_the_hoop_on_most_samples(tmp_path):
    assert detect_fixed_rim(_clip(tmp_path / "clip.mp4"), FixedDetector(seen_every=4), samples=12) is None


def test_scoring_rim_prefers_a_mark_then_detection():
    from app.analyzer import _scoring_rim
    marked, tracked, detected = (.1, .1, .05, .02), {0: (.2, .2, .05, .02)}, {0: (.3, .3, .05, .02)}
    assert _scoring_rim("moving", True, marked, tracked, detected) == (tracked, "marked")
    assert _scoring_rim("moving", True, marked, {}, detected) == (marked, "marked")
    assert _scoring_rim("elevated", True, marked, {}, detected) == (marked, "marked")
    assert _scoring_rim("moving", False, None, {}, detected) == (detected, "detected")
    assert _scoring_rim("courtside", False, marked, {}, {}) == (marked, "detected")  # form-mode sample
    assert _scoring_rim("elevated", False, None, {}, {}) == (None, None)
