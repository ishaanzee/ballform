import json
import os

import pytest
from fastapi.testclient import TestClient

from app import court_map, main


def _job(tmp_path, with_court=True):
    job = tmp_path / "abc123"
    job.mkdir(parents=True)
    # A camera on the -x sideline: going down the picture comes toward -x on the floor.
    image_to_court = [[0.0, -0.05, 20.0], [0.02, 0.0, 0.0], [0.0, 0.0, 1.0]]
    court = {"summary": {"standard": "nba"},
             "frames": [{"frame": f, "reliable": f != 2, "image_to_court": image_to_court} for f in range(4)],
             "trajectories": {"tracks": [{"track_id": 7, "frames": [0, 1, 3], "x": [1.0, 1.5, 2.5], "y": [20.0, 20.5, 21.0]},
                                         {"track_id": 9, "frames": [], "x": [], "y": []}]}}
    players = [{"track_id": 7, "appearance": [0.5, 0.5, 0.5]}, {"track_id": 8, "appearance": None}]
    observations = {"fps": 30.0, "width": 1920, "height": 1080,
                    "players": [{"frame": f, "players": players} for f in range(4)]}
    result = {"video": {"fps": 30.0}, "shots": [
        {"number": 1, "start_s": 0.0, "release_s": 0.05, "end_s": 0.1, "outcome": "made",
         "game": {"players": {"shooter_track_id": 7, "defender_track_id": None}}},
        {"number": 2, "release_s": 1.0, "outcome": "unknown"}]}
    for name, content in (("court.json", court), ("observations.json", observations), ("result.json", result)):
        if with_court or name != "court.json":
            (job / name).write_text(json.dumps(content))
    return job


def test_court_map_lists_each_tracked_player_in_court_feet(tmp_path):
    payload = court_map.load(_job(tmp_path))
    assert [t["id"] for t in payload["tracks"]] == [7]   # a track with no floor points is left out
    track = payload["tracks"][0]
    assert track["frames"] == [0, 1, 3] and track["x"] == [1.0, 1.5, 2.5] and track["y"] == [20.0, 20.5, 21.0]
    assert track["colour"] == "#777777"   # mid grey in Lab
    assert payload["fps"] == 30.0 and payload["frames"] == 4 and payload["mapped_frames"] == 3
    assert payload["shots"][0] == {"number": 1, "start_s": 0.0, "release_s": 0.05, "end_s": 0.1, "outcome": "made",
                                   "shooter": 7, "defender": None}
    assert payload["shots"][1]["shooter"] is None
    assert payload["toward_camera"] == pytest.approx([-1.0, 0.0], abs=1e-6)


def test_court_map_is_kept_and_rebuilt_when_the_analysis_changes(tmp_path):
    job = _job(tmp_path)
    court_map.load(job)
    cache = job / court_map.CACHE
    cache.write_text(json.dumps({"kept": True}))
    assert court_map.load(job) == {"kept": True}
    result = job / "result.json"
    result.write_text(result.read_text())
    stamp = cache.stat().st_mtime + 5
    os.utime(result, (stamp, stamp))
    assert "tracks" in court_map.load(job)


def test_no_court_map_without_calibration(tmp_path, monkeypatch):
    job = _job(tmp_path, with_court=False)
    assert court_map.load(job) is None
    assert court_map.toward_camera({"frames": []}, 1920, 1080) is None
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    client = TestClient(main.app)
    assert client.get("/api/jobs/abc123/court-map").status_code == 404
    assert client.get("/api/jobs/..%2Fabc/court-map").status_code == 404
    _job(tmp_path / "with")
    monkeypatch.setattr(main, "JOBS_DIR", tmp_path / "with")
    assert client.get("/api/jobs/abc123/court-map").json()["tracks"][0]["id"] == 7
