import json

import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ACCESS_TOKEN", "")
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    return TestClient(main.app)


def test_upload_forwards_game_options(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args: calls.append(args))
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video", "video/mp4")},
                           data={"mode": "one_on_one", "handedness": "left"})
    assert response.status_code == 202
    assert calls[0][3:] == ("one_on_one", "left")
    assert calls[0][1].read_bytes() == b"video"


def test_existing_upload_defaults_to_form(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args: calls.append(args))
    response = client.post("/api/jobs", files={"video": ("shot.mp4", b"video")})
    assert response.status_code == 202
    assert calls[0][3:] == ("form", "right")


@pytest.mark.parametrize("data", [
    {"mode": "invalid"}, {"handedness": "either"},
    {"rim": json.dumps([float("nan"), .2, .1, .1])},
    {"rim": json.dumps([.95, .2, .2, .1])},
])
def test_invalid_options_rejected_before_job_creation(client, data):
    response = client.post("/api/jobs", files={"video": ("shot.mp4", b"video")}, data=data)
    assert response.status_code == 422
    assert not list(main.JOBS_DIR.iterdir())
