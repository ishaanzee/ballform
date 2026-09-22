from app.analyzer import _shooter_intervals
from app.models import ShotResult


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
