import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
CLICKS = [["lane_base_right", 358, 565], ["lane_base_left", 173, 722], ["ft_right", 939, 600],
          ["ft_left", 834, 762], ["baseline_right", 507, 431], ["three_base_right", 486, 450],
          ["half_right", 1885, 506]]


PROPOSAL = {"ok": True, "reason": None, "confidence": .64, "standard": "nba",
            "points": [{"id": key, "image": [x / 1920, y / 1080]} for key, x, y in CLICKS]}


def harness(clicks, proposal=None):
    command = ["node", str(ROOT / "tests" / "court_js_harness.js"), str(ROOT), json.dumps(clicks)]
    if proposal is not None:
        command += ["1920", "1080", json.dumps(proposal)]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    return json.loads(output)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_fit_matches_the_server_fit_and_sends_the_landmarks():
    from app.court import fit, parse_landmarks
    result = harness(CLICKS)
    assert "3.5 px RMS" in result["status"]
    parsed = parse_landmarks(result["field"])  # the analyzer accepts what the page sends
    assert [key for key, _ in parsed["points"]] == [c[0] for c in CLICKS]
    assert fit(parsed["points"], 1920, 1080).summary()["clicked_error_px"]["rms"] == pytest.approx(3.5, abs=.05)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_browser_warns_that_four_clicks_cannot_be_checked():
    result = harness(CLICKS[:4])
    assert "cannot be checked" in result["status"] and "extrapolated" in result["status"]
    assert result["field"] is not None
    assert harness(CLICKS[:3])["field"] is None


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_an_accepted_proposal_is_sent_as_auto():
    from app.court import parse_landmarks
    result = harness([], PROPOSAL)
    assert result["source"] == "auto"
    assert "proposed 7 landmarks" in result["status"] and "no click error" in result["status"]
    parsed = parse_landmarks(result["field"])
    assert parsed["source"] == "auto" and [key for key, _ in parsed["points"]] == [c[0] for c in CLICKS]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_dragging_or_removing_a_proposed_point_marks_it_adjusted():
    dragged = harness([["drag", 358, 565, 362, 569]], PROPOSAL)
    assert dragged["source"] == "auto, adjusted"
    point = next(p for p in dragged["field"]["points"] if p["id"] == "lane_base_right")
    assert point["image"] == pytest.approx([362 / 1920, 569 / 1080])
    assert "Error on the marked points" in dragged["status"]
    removed = harness([["remove", "half_right"]], PROPOSAL)
    assert removed["source"] == "auto, adjusted"
    assert "half_right" not in [p["id"] for p in removed["field"]["points"]]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_a_failed_proposal_leaves_manual_marks_alone():
    result = harness(CLICKS[:5], {"ok": False, "reason": "No lines.", "confidence": 0, "points": []})
    # The failed proposal is applied first, then the manual clicks go ahead as before.
    assert result["source"] == "manual" and len(result["field"]["points"]) == 5
    assert "No lines." in result["status"]
