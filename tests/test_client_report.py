import json
from datetime import datetime

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.client_report import build_client_report, form_consistency, render_client_report


def form_shot(number, elbow, outcome="made"):
    return {"number": number, "release_s": number * 1.5, "outcome": outcome, "outcome_confidence": .8,
            "evidence": ["Ball arc detected"], "metrics": {"elbow_angle_at_release_deg": elbow,
                                                           "release_height_body_ratio": 1.2},
            "cues": ["Compare with your own shots."], "game": None}


def game_shot(number, **metrics):
    return {"number": number, "release_s": 2.0, "outcome": "missed", "outcome_confidence": .7,
            "evidence": ["Detector jump-shot class fired"], "metrics": {}, "cues": [], "shot_type": "jump shot",
            "game": {"score": 61.2, "confidence": .7, "score_range": None,
                     "metrics": {"separation_torso": 1.4, "defender_selection_margin_torso": .3, **metrics}}}


def render(result, thumbs=None, **kwargs):
    thumbs = thumbs or [None] * len(result["shots"])
    return render_client_report(result, thumbs, analyzed_on=datetime(2026, 9, 25), **kwargs)


def test_form_report_shows_consistency_and_plain_labels():
    result = {"mode": "form", "handedness": "left", "video": {"duration_s": 12.0},
              "diagnostics": {"rim_source": "detected"}, "limitations": ["2D estimates only."],
              "shots": [form_shot(1, 150.0), form_shot(2, 160.0, "missed"), form_shot(3, 170.0)]}
    page = render(result, prepared_for="J.M.")
    assert "Shooting form review" in page
    assert "Elbow angle at release" in page and "elbow_angle_at_release_deg" not in page
    assert "Consistency across shots" in page
    assert "2 / 3" in page  # made
    assert "J.M." in page and "September 25, 2026" in page
    assert "2D estimates only." in page
    # Raw analyzer evidence stays on the review page.
    assert "Ball arc detected" not in page


def test_uncalled_shots_are_left_out_of_made():
    shots = [form_shot(1, 150.0), form_shot(2, 160.0, "unknown")]
    assert "1 / 1" in render({"mode": "form", "diagnostics": {}, "limitations": [], "shots": shots})
    page = render({"mode": "form", "diagnostics": {}, "limitations": [], "shots": [shots[1]]})
    assert "0 / 1" not in page and "Not measured" in page


def test_consistency_needs_two_measured_shots():
    rows = form_consistency([form_shot(1, 150.0), form_shot(2, 170.0), {"number": 3, "metrics": {}}])
    elbow = next(r for r in rows if r[0] == "Elbow angle at release")
    assert elbow[1:] == ("160°", "150°", "170°")
    assert form_consistency([form_shot(1, 150.0)]) == []


def test_game_report_prefers_feet_and_hides_model_internals():
    result = {"mode": "one_on_one", "diagnostics": {"rim_source": "detected"}, "limitations": [],
              "court_calibration": {"standard": "nba"}, "game_summary": {"mean_score": 61.2},
              "shots": [game_shot(1, separation_ft=6.5, shot_distance_ft=22.4, shot_zone="corner_three")]}
    page = render(result)
    assert "Game shot report" in page and "Every shot" in page
    assert "6.5 ft" in page and "1.4 torso lengths" not in page
    assert "Corner three" in page and "22.4 ft" in page
    assert "61.2 / 100" in page
    assert "selection margin" not in page.lower()


def test_game_report_falls_back_to_torso_lengths():
    page = render({"mode": "one_on_one", "diagnostics": {}, "limitations": [], "shots": [game_shot(1)]})
    assert "1.4 torso lengths" in page


def test_missing_rim_is_not_reported_as_misses():
    result = {"mode": "one_on_one", "diagnostics": {"rim_source": None}, "limitations": [],
              "shots": [game_shot(1)]}
    page = render(result)
    assert "Missed" not in page
    assert "Not measured" in page
    assert "basket was not visible" in page


def test_text_is_escaped():
    shot = form_shot(1, 150.0)
    shot["cues"] = ["<script>alert(1)</script>"]
    page = render({"mode": "form", "diagnostics": {}, "limitations": [], "shots": [shot]},
                  prepared_for="<b>x</b>")
    assert "<script>alert(1)</script>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page


def test_empty_clip_explains_what_to_film():
    page = render({"mode": "form", "diagnostics": {}, "limitations": [], "shots": []})
    assert "No complete shot was found" in page


def write_job(directory, result, frames=20):
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps(result))
    writer = cv2.VideoWriter(str(directory / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i * 10, np.uint8))
    writer.release()


def test_build_embeds_release_snapshots(tmp_path):
    job = tmp_path / "abc123"
    write_job(job, {"mode": "form", "diagnostics": {}, "limitations": [], "annotated_video": "annotated.mp4",
                    "shots": [form_shot(1, 150.0)]})
    page = build_client_report(job)
    assert page.count("data:image/jpeg;base64,") == 1
    assert "ballform-abc123.mp4" in page


def test_build_without_video_still_renders(tmp_path):
    job = tmp_path / "abc123"
    job.mkdir()
    (job / "result.json").write_text(json.dumps({"mode": "form", "diagnostics": {}, "limitations": [],
                                                  "shots": [form_shot(1, 150.0)]}))
    page = build_client_report(job)
    assert "data:image" not in page and "Shot 1" in page


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ACCESS_TOKEN", "")
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    return TestClient(main.app)


def test_api_serves_client_report(client, tmp_path):
    write_job(tmp_path / "abc123", {"mode": "form", "diagnostics": {}, "limitations": [],
                                    "shots": [form_shot(1, 150.0)]})
    response = client.get("/api/jobs/abc123/client-report", params={"prepared_for": "Coach K"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Coach K" in response.text


def test_api_client_report_404s(client):
    assert client.get("/api/jobs/missing1/client-report").status_code == 404
    assert client.get("/api/jobs/..%2Fetc/client-report").status_code == 404


def test_api_client_report_requires_token(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ACCESS_TOKEN", "secret")
    write_job(tmp_path / "abc123", {"mode": "form", "diagnostics": {}, "limitations": [], "shots": []})
    assert client.get("/api/jobs/abc123/client-report").status_code == 401
    assert client.get("/api/jobs/abc123/client-report", params={"token": "secret"}).status_code == 200
