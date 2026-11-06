import cv2
import numpy as np
import pytest

from app.rim import _tiles, detect_fixed_rim, fixed_hoop_candidates, rim_candidates, track_detected_rims


def rim(x, y=.3, w=.03, h=.01, conf=.85):
    return conf, (x, y, x + w, y + h)


def test_dominant_hoop_is_kept_and_a_false_positive_blip_dropped():
    detections = {f: [rim(.30 + .002 * f)] for f in range(0, 40)}
    detections[20].append(rim(.60, y=.35, conf=.6))  # a sleeve near the baseline, once
    rims, stats = track_detected_rims(detections, [])
    assert sorted(rims) == list(range(40))
    assert rims[20][0] == pytest.approx(.34, abs=.002)
    assert stats["other_hoop_tracks"] == 1


def test_ball_at_the_rim_adding_an_overlapping_box_does_not_split_the_hoop():
    # 3e78, f6f0: as the ball arrives the detector also boxes ball and rim together.
    detections = {f: [rim(.30 - .002 * f)] for f in range(0, 40)}
    detections[20] = [rim(.254, w=.04, h=.016, conf=.64), rim(.26, conf=.55)]
    rims, stats = track_detected_rims(detections, [])
    assert sorted(rims) == list(range(40))
    assert stats["other_hoop_tracks"] == 0


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


def _three_hoops(frames=60):
    """A gym camera seeing its own hoop (x .50) between two neighbouring courts' hoops,
    which the detector is more sure of."""
    return {f: [rim(.20, w=.03, h=.017, conf=.9), rim(.485, w=.03, h=.017, conf=.6),
                rim(.77, w=.03, h=.017, conf=.9)] for f in range(frames)}


def test_ball_dropping_onto_a_hoop_picks_it_over_more_confident_neighbours():
    # Rim centre (.50, .3085); a shot comes down onto it from above, frames 20-29.
    balls = {f: (.50, .20 + .01 * (f - 20), .8) for f in range(20, 30)}
    rims, _ = track_detected_rims(_three_hoops(), [], balls=balls)
    assert rims[10][0] == pytest.approx(.485)
    without_balls, _ = track_detected_rims(_three_hoops(), [])
    assert without_balls[10][0] != pytest.approx(.485)


def test_ball_lifted_to_a_hoop_counts_for_a_layup():
    # Carried up from below the ring to just above it.
    balls = {f: (.505, .33 - .006 * (f - 20), .8) for f in range(20, 30)}
    rims, _ = track_detected_rims(_three_hoops(), [], balls=balls)
    assert rims[10][0] == pytest.approx(.485)


def test_still_faint_or_low_balls_near_a_hoop_do_not_pick_it():
    # A faint ball sitting by the left hoop, and a confident one passing in front of the
    # right hoop below its ring: neither is a shot at those hoops, so confidence decides.
    balls = {f: (.215, .30, .3) for f in range(0, 30)}
    balls.update({f: (.77 + .002 * (f - 30), .33 + .002 * (f - 30), .9) for f in range(30, 50)})
    rims, _ = track_detected_rims(_three_hoops(), [], balls=balls)
    without_balls, _ = track_detected_rims(_three_hoops(), [])
    assert rims[10] == without_balls[10]


class TiledDetector:
    """Sees only a player on a full 1280x720 frame; on the zoomed tiles (called in
    _tiles order) it sees the hoop at frame (.9, .3) and a pole-shaped box at (.1, .1)."""

    def __init__(self):
        self.tiles = list(_tiles(1280, 720))
        self.tile_calls = 0

    def detect(self, image):
        if image.shape[1] == 1280:
            return [(4, .9, (.4, .5, .42, .8))]  # a player, 0.3 of the frame tall
        x1, y1, x2, y2 = self.tiles[self.tile_calls % len(self.tiles)]
        self.tile_calls += 1
        found = []
        for x, y, w, h, conf in ((.9, .3, .015, .015, .4), (.1, .1, .008, .02, .5)):
            box = ((x - w) * 1280, (y - h) * 720, (x + w) * 1280, (y + h) * 720)
            if x1 <= box[0] and box[2] <= x2 and y1 <= box[1] and box[3] <= y2:
                found.append((10, conf, ((box[0] - x1) / (x2 - x1), (box[1] - y1) / (y2 - y1),
                                         (box[2] - x1) / (x2 - x1), (box[3] - y1) / (y2 - y1))))
        return found


def _wide_clip(path, frames=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for _ in range(frames):
        writer.write(np.zeros((720, 1280, 3), np.uint8))
    writer.release()
    return path


def test_tiled_search_finds_a_small_hoop_and_rejects_a_pole(tmp_path):
    hoops, stats = fixed_hoop_candidates(_wide_clip(tmp_path / "clip.mp4"), TiledDetector())
    assert stats["searched_tiles"] and len(hoops) == 1
    conf, (x1, y1, x2, y2) = hoops[0]
    assert conf == pytest.approx(.4) and (x1 + x2) / 2 == pytest.approx(.9, abs=.01)


def test_tiled_search_is_skipped_when_full_frames_show_the_rim(tmp_path):
    hoops, stats = fixed_hoop_candidates(_clip(tmp_path / "clip.mp4"), FixedDetector())
    assert hoops == [] and not stats["searched_tiles"]


# Static-hoop fill for fixed cameras (static_hoop_frames).

def _drop_onto(x, frames):
    """A confident ball coming down onto a hoop centred at x (ring at y .3085)."""
    return {f: (x, .20 + .01 * (f - frames[0]), .8) for f in frames}


def test_fixed_camera_hoop_fills_the_frames_before_it_was_first_detected():
    # 16d7: the hoop of play is only detected from frame 30, after the first shot arrived.
    detections = {f: [rim(.485, w=.03, h=.017)] if f >= 30 else [] for f in range(60)}
    rims, stats = track_detected_rims(detections, [], static_hoop_frames=25)
    assert sorted(rims) == list(range(60)) and list(rims) == sorted(rims)
    assert rims[5] == pytest.approx((.485, .3, .03, .017))
    assert stats["static_hoop_frames"] == 30
    unfilled, _ = track_detected_rims(detections, [])
    assert sorted(unfilled) == list(range(30, 60))


def test_split_hoop_of_play_beside_neighbouring_hoops_is_filled():
    # d063: the centre hoop goes unseen for 40 frames, so its track splits; the ball
    # drops onto it, and the two neighbouring courts' hoops get no ball.
    detections = _three_hoops(90)
    for f in range(20, 60):
        detections[f] = [d for d in detections[f] if d[1][0] != .485]
    balls = _drop_onto(.50, range(65, 75))
    rims, stats = track_detected_rims(detections, [], balls=balls, static_hoop_frames=25)
    assert sorted(rims) == list(range(90))
    assert rims[40] == pytest.approx((.485, .3, .03, .017))
    unfilled, _ = track_detected_rims(detections, [], balls=balls)
    assert 40 not in unfilled and unfilled[70][0] == pytest.approx(.485)


def test_moving_hoop_is_not_frozen():
    # A box drifting by over two rim widths is not a static hoop.
    detections = {f: [rim(.30 + .002 * f)] if f < 40 else [] for f in range(60)}
    rims, stats = track_detected_rims(detections, [], static_hoop_frames=25)
    assert sorted(rims) == list(range(40)) and stats["static_hoop_frames"] == 0


def test_hoop_seen_too_briefly_is_not_spread_over_the_clip():
    detections = {f: [rim(.4)] if f < 10 else [] for f in range(60)}
    rims, _ = track_detected_rims(detections, [], static_hoop_frames=25)
    assert sorted(rims) == list(range(10))


def test_no_fill_when_the_ball_goes_to_two_hoops():
    # Play moves from hoop A (detected until frame 40) to hoop B: A must not stand in for B.
    detections = {f: ([rim(.20, w=.03, h=.017, conf=.9)] if f < 40 else []) + [rim(.70, w=.03, h=.017, conf=.5)]
                  for f in range(60)}
    balls = {**_drop_onto(.215, range(10, 20)), **_drop_onto(.715, range(45, 53))}
    rims, stats = track_detected_rims(detections, [], balls=balls, static_hoop_frames=25)
    assert rims[10][0] == pytest.approx(.20) and 50 not in rims
    assert stats["static_hoop_frames"] == 0


def test_no_fill_among_several_hoops_without_ball_evidence():
    detections = {f: ([rim(.20, conf=.95)] if f < 40 else []) + [rim(.50, conf=.6), rim(.80, conf=.6)]
                  for f in range(60)}
    rims, _ = track_detected_rims(detections, [], static_hoop_frames=25)
    assert rims[10][0] == pytest.approx(.20) and 50 not in rims


def test_static_fill_stays_inside_its_camera_segment():
    detections = {f: [rim(.4)] if f < 30 else [] for f in range(60)}
    rims, _ = track_detected_rims(detections, [45], static_hoop_frames=25)
    assert sorted(rims) == list(range(45))


def test_only_fixed_camera_profiles_fill_a_static_hoop(monkeypatch, tmp_path):
    import app.analyzer as analyzer

    class Stop(Exception):
        pass

    seen = {}

    def capture(*_args, **kwargs):
        seen[profile] = kwargs.get("static_hoop_frames")
        raise Stop

    monkeypatch.setattr(analyzer, "track_detected_rims", capture)
    for profile in ("moving", "elevated", "courtside"):
        inputs = {"fps": 25.0, "width": 1920, "height": 1080, "total": 50, "frame_no": 49,
                  "analyzed_fps": 25.0, "stride": 1, "mode": "game", "handedness": "right",
                  "profile": profile, "game_mode": False, "rim": None, "rim_marked": False,
                  "auto_rim": True, "tracked_rims": {}, "cut_frames": [], "balls": [], "poses": [],
                  "player_frames": [{"frame": 0, "rims": []}], "flows": {}, "flow_rims": {},
                  "calibration": None, "court_landmarks": None}
        with pytest.raises(Stop):
            analyzer.score_clip(inputs, None, tmp_path / "input.mp4")
    assert seen == {"moving": None, "elevated": 25, "courtside": 25}
