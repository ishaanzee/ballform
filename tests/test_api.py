import json

import cv2
import numpy as np
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
    monkeypatch.setattr(main, "_run", lambda *args, **kwargs: calls.append((args, kwargs)))
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video", "video/mp4")},
                           data={"mode": "one_on_one", "handedness": "left"})
    assert response.status_code == 202
    assert calls[0][0][3:] == ("one_on_one", "left", "courtside", None)
    assert calls[0][0][1].read_bytes() == b"video"
    assert calls[0][1]["pose_model"] == "yolo26s-pose"


def test_existing_upload_defaults_to_form(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args, **kwargs: calls.append((args, kwargs)))
    response = client.post("/api/jobs", files={"video": ("shot.mp4", b"video")})
    assert response.status_code == 202
    assert calls[0][0][3:] == ("form", "right", "courtside", None)


def test_upload_forwards_elevated_court(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args, **kwargs: calls.append((args, kwargs)))
    court = [[0, 0], [1, 0], [1, 1], [0, 1]]
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": "elevated", "court": json.dumps(court)})
    assert response.status_code == 202
    assert calls[0][0][5:] == ("elevated", court)


@pytest.mark.parametrize("camera", ["auto", "broadcast"])
def test_removed_camera_profiles_are_rejected_with_the_replacement(client, camera):
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": camera})
    assert response.status_code == 422
    assert "use moving" in response.json()["detail"]


def test_upload_accepts_moving_camera_profile(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args, **kwargs: calls.append((args, kwargs)))
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": "moving",
                                 "rim": json.dumps([.7, .2, .1, .05])})
    assert response.status_code == 202
    assert calls[0][0][5] == "moving"


def test_moving_camera_no_longer_requires_a_marked_rim(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args, **kwargs: calls.append((args, kwargs)))
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": "moving"})
    assert response.status_code == 202
    assert calls[0][0][2] is None  # the rim is detected during analysis


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
                                 "pose_model": "yolo26m-pose"})
    assert response.status_code == 202
    assert response.json()["pose_model_requested"] == "yolo26m-pose"
    assert calls[0][1]["pose_model"] == "yolo26m-pose"


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
    assert client.get("/assets/app.js?v=game-models-shooter-overlay-2").headers["cache-control"] == "no-store"


@pytest.mark.parametrize("data", [
    {"mode": "invalid"}, {"handedness": "either"},
    {"camera": "unknown"}, {"pose_model": "unknown"}, {"pose_model": "yolo11m-pose"},
    {"court": "oops"}, {"court": "[1,2,3]"},
    {"court": "[[0,0],[1,1],[0,1],[1,0]]"},
    {"rim": json.dumps([float("nan"), .2, .1, .1])},
    {"rim": json.dumps([.95, .2, .2, .1])},
])
def test_invalid_options_rejected_before_job_creation(client, data):
    response = client.post("/api/jobs", files={"video": ("shot.mp4", b"video")}, data=data)
    assert response.status_code == 422
    assert not list(main.JOBS_DIR.iterdir())


LANDMARKS = {"standard": "nba", "time_s": 1.5, "points": [
    {"id": "lane_base_left", "image": [.1, .7]}, {"id": "lane_base_right", "image": [.2, .5]},
    {"id": "ft_left", "image": [.45, .72]}, {"id": "ft_right", "image": [.5, .55]}]}


def test_upload_forwards_court_landmarks(client, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_run", lambda *args, **kwargs: calls.append((args, kwargs)))
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", "camera": "moving", "court_landmarks": json.dumps(LANDMARKS)})
    assert response.status_code == 202
    assert calls[0][1]["court_landmarks"] == LANDMARKS


@pytest.mark.parametrize("data, message", [
    ({"court_landmarks": "{"}, "JSON"),
    ({"court_landmarks": json.dumps({**LANDMARKS, "points": LANDMARKS["points"][:3]})}, "at least 4"),
    ({"court_landmarks": json.dumps({**LANDMARKS, "standard": "wnba"})}, "standard"),
    ({"court_landmarks": json.dumps(LANDMARKS), "mode": "form"}, "game analysis"),
])
def test_invalid_court_landmarks_are_rejected_with_a_reason(client, data, message):
    response = client.post("/api/jobs", files={"video": ("game.mp4", b"video")},
                           data={"mode": "one_on_one", **data})
    assert response.status_code == 422 and message in response.json()["detail"]
    assert not list(main.JOBS_DIR.iterdir())


def test_court_detect_proposes_landmarks_for_a_frame(client):
    from app.court import parse_landmarks
    from tests_support_court import render_court, synthetic_camera
    court_to_image = synthetic_camera(focal=2600.)[2] @ np.diag([-1., 1., 1.])  # diagram handedness
    _, png = cv2.imencode(".png", render_court(court_to_image, seed=3))
    response = client.post("/api/court/detect", files={"frame": ("frame.png", png.tobytes(), "image/png")},
                           data={"standard": "nba"})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] and body["standard"] == "nba" and len(body["points"]) >= 5
    # What the page gets back is what it sends with the analysis, and nothing was stored.
    parsed = parse_landmarks({"standard": "nba", "time_s": 1.5, "source": "auto", "points": body["points"]})
    assert parsed["source"] == "auto"
    assert not list(main.JOBS_DIR.iterdir())


def test_court_detect_rejects_bad_input(client):
    assert client.post("/api/court/detect", files={"frame": ("f.png", b"not an image")}).status_code == 422
    assert client.post("/api/court/detect", files={"frame": ("f.png", b"x")},
                       data={"standard": "wnba"}).status_code == 422


def test_court_detect_explains_a_frame_without_a_court(client):
    _, png = cv2.imencode(".png", np.full((360, 640, 3), 90, np.uint8))
    body = client.post("/api/court/detect", files={"frame": ("f.png", png.tobytes())}).json()
    assert body["ok"] is False and body["reason"] and body["points"] == []
