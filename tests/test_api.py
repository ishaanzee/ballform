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
    assert calls[0][3:] == ("one_on_one", "left", "auto", None)
    assert calls[0][1].read_bytes() == b"video"


def test_existing_upload_defaults_to_form(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args: calls.append(args))
    response = client.post("/api/jobs", files={"video": ("shot.mp4", b"video")})
    assert response.status_code == 202
    assert calls[0][3:] == ("form", "right", "auto", None)


def test_upload_forwards_broadcast_court(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args: calls.append(args))
    court = [[0, 0], [1, 0], [1, 1], [0, 1]]
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": "broadcast", "court": json.dumps(court)})
    assert response.status_code == 202
    assert calls[0][5:] == ("broadcast", court)


def test_upload_accepts_moving_camera_profile(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args: calls.append(args))
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": "moving",
                                 "rim": json.dumps([.7, .2, .1, .05])})
    assert response.status_code == 202
    assert calls[0][5] == "moving"


def test_moving_camera_requires_first_frame_rim(client):
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": "moving"})
    assert response.status_code == 422


@pytest.mark.parametrize("data", [
    {"mode": "invalid"}, {"handedness": "either"},
    {"camera": "unknown"}, {"court": "oops"}, {"court": "[1,2,3]"},
    {"court": "[[0,0],[1,1],[0,1],[1,0]]"},
    {"rim": json.dumps([float("nan"), .2, .1, .1])},
    {"rim": json.dumps([.95, .2, .2, .1])},
])
def test_invalid_options_rejected_before_job_creation(client, data):
    response = client.post("/api/jobs", files={"video": ("shot.mp4", b"video")}, data=data)
    assert response.status_code == 422
    assert not list(main.JOBS_DIR.iterdir())
