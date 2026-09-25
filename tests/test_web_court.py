import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
CLICKS = [["lane_base_right", 358, 565], ["lane_base_left", 173, 722], ["ft_right", 939, 600],
          ["ft_left", 834, 762], ["baseline_right", 507, 431], ["three_base_right", 486, 450],
          ["half_right", 1885, 506]]


def harness(clicks):
    output = subprocess.run(["node", str(ROOT / "tests" / "court_js_harness.js"), str(ROOT), json.dumps(clicks)],
                            capture_output=True, text=True, check=True).stdout
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
