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


def test_upload_accepts_rim_frame(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args, **kwargs: calls.append((args, kwargs)))
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": "moving",
                                 "rim": json.dumps([.7, .2, .1, .05]), "rim_frame": "120"})
    assert response.status_code == 202
    assert calls[0][1]["rim_frame"] == 120


def test_upload_forwards_selected_pose_model(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args, **kwargs: calls.append((args, kwargs)))
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": "moving",
                                 "rim": json.dumps([.7, .2, .1, .05]),
                                 "pose_model": "yolo26s-pose"})
    assert response.status_code == 202
    assert response.json()["pose_model_requested"] == "yolo26s-pose"
    assert calls[0][1]["pose_model"] == "yolo26s-pose"


def test_run_reports_the_pose_checkpoint_loaded_by_analyzer(monkeypatch, tmp_path):
    job_id = "modelroutecheck"
    main.STATE.pop(job_id, None)

    def fake_analyze(_input, _output, _rim, progress, **kwargs):
        assert kwargs["pose_model"] == "yolo26s-pose"
        progress(.04, "Pose model loaded: yolo26s-pose.pt")
        return {"vision": {"pose_model": "yolo26s-pose.pt"}}

    monkeypatch.setattr(main, "analyze_video", fake_analyze)
    main._run(job_id, tmp_path / "clip.mp4", None, mode="one_on_one", pose_model="yolo26s-pose")
    assert main.STATE[job_id]["pose_model_loaded"] == "yolo26s-pose.pt"
    assert main.STATE[job_id]["result"]["vision"]["pose_model"] == "yolo26s-pose.pt"
    main.STATE.pop(job_id, None)


def test_app_page_and_assets_disable_stale_caching(client):
    assert client.get("/").headers["cache-control"] == "no-store"
    assert client.get("/assets/app.js?v=pose-model-confirmation-1").headers["cache-control"] == "no-store"


@pytest.mark.parametrize("data", [
    {"mode": "invalid"}, {"handedness": "either"},
    {"camera": "unknown"}, {"pose_model": "unknown"},
    {"court": "oops"}, {"court": "[1,2,3]"},
    {"court": "[[0,0],[1,1],[0,1],[1,0]]"},
    {"rim": json.dumps([float("nan"), .2, .1, .1])},
    {"rim": json.dumps([.95, .2, .2, .1])},
])
def test_invalid_options_rejected_before_job_creation(client, data):
    response = client.post("/api/jobs", files={"video": ("shot.mp4", b"video")}, data=data)
    assert response.status_code == 422
    assert not list(main.JOBS_DIR.iterdir())
