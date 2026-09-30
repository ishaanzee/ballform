import json

import numpy as np
import pytest

from app.court import apply, fit, parse_landmarks, template
from app.court_detect import Line, _cameras, _plausible, _template_lines, _two_one, propose

from tests_support_court import render_court, synthetic_camera

W, H = 1920, 1080
# The calibration diagram is a map seen from half court facing the basket, which mirrors a
# right-handed (x, y, up) world; user clicks on real footage follow the diagram (their fits
# have a positive determinant), so the synthetic camera is mirrored to match.
MIRROR = np.diag([-1., 1., 1.])


def diagram_view(**camera):
    return synthetic_camera(**camera)[2] @ MIRROR


def landmark_errors(truth, found, standard="nba"):
    court = template(standard)
    points = np.asarray(list(court.landmarks.values()))
    a, b = apply(truth, points), apply(found, points)
    inside = (a[:, 0] >= 0) & (a[:, 0] < W) & (a[:, 1] >= 0) & (a[:, 1] < H)
    return np.linalg.norm(a[inside] - b[inside], axis=1)


@pytest.mark.parametrize("camera", [
    {"position": (-100., 40., 35.), "focal": 2600.},        # broadcast, basket on the left
    {"position": (100., 45., 30.), "focal": 2600.},         # the other side of the court
    {"position": (-80., 20., 45.), "look_at": (0., 18., 0.), "focal": 2200.},
])
def test_proposes_the_rendered_court(camera):
    truth = diagram_view(**camera)
    assert np.linalg.det(truth) > 0
    proposal = propose(render_court(truth, seed=1))
    assert proposal.ok, proposal.reason
    assert proposal.confidence >= .6
    errors = landmark_errors(truth, proposal.court_to_image)
    assert np.median(errors) < 5 and errors.max() < 15
    # The points are named template landmarks inside the frame that the existing fit accepts.
    assert len(proposal.points) >= 5
    assert all(0 < x < 1 and 0 < y < 1 for _, (x, y) in proposal.points)
    calibration = fit(proposal.points, W, H)
    assert landmark_errors(truth, calibration.court_to_image).max() < 15


def test_other_standards_use_their_own_lines():
    truth = diagram_view(position=(-90., 35., 35.), focal=2400.)
    proposal = propose(render_court(truth, standard="fiba", seed=2), "fiba")
    assert proposal.ok, proposal.reason
    assert proposal.standard == "fiba"
    assert np.median(landmark_errors(truth, proposal.court_to_image, "fiba")) < 5


@pytest.mark.parametrize("frame", [np.full((H, W, 3), 128, np.uint8),
                                   np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)])
def test_says_why_when_there_is_no_court(frame):
    proposal = propose(frame)
    assert not proposal.ok and proposal.reason
    assert proposal.points == []
    assert json.loads(json.dumps(proposal.to_dict()))["ok"] is False


def test_three_lines_and_the_focal_length_recover_the_view():
    """Two lane lines and the baseline plus a centred square-pixel camera leave only the focal length free."""
    k, _, _ = synthetic_camera(focal=2600.)
    truth = diagram_view(focal=2600.)
    inverse_t = np.linalg.inv(truth).T

    def image_line(a, b, c):
        eq = inverse_t @ np.array([a, b, c])
        return Line(eq / np.linalg.norm(eq[:2]), 100., np.zeros((2, 2)))

    court = template("nba")
    along, across = _template_lines(court)
    pair = (image_line(1., 0., 8.), image_line(1., 0., -8.))   # x = -8 and x = 8
    single = image_line(0., 1., 0.)                             # y = 0
    hs, _ = _two_one(pair, single, along, across, [k[0, 0]], np.array([W / 2, H / 2]))
    points = np.asarray([(-8., 0.), (8., 0.), (8., 19.), (0., 29.), (-22., 14.)])
    target = apply(truth, points)
    errors = [np.nanmax(np.linalg.norm(apply(h, points) - target, axis=1)) for h in hs]
    assert min(errors) < 1e-6


def test_plausibility_rejects_views_no_camera_could_take():
    truth = diagram_view()
    focal, error, centre = _cameras(truth[None], W, H)
    assert focal[0] == pytest.approx(3000., rel=1e-6) and error[0] < 1e-9
    assert np.abs(centre[0]) == pytest.approx([100., 40., 35.], abs=1e-6)
    squashed = truth @ np.diag([1., .05, 1.])  # a court flattened onto a line
    assert _plausible(np.stack([truth, squashed]), W, H).tolist() == [True, False]


def test_landmark_source_is_validated_and_reported():
    base = {"standard": "nba", "time_s": 0, "points": [
        {"id": "lane_base_left", "image": [.1, .7]}, {"id": "lane_base_right", "image": [.2, .5]},
        {"id": "ft_left", "image": [.45, .7]}, {"id": "ft_right", "image": [.5, .55]}]}
    assert parse_landmarks(base)["source"] == "manual"
    assert parse_landmarks({**base, "source": "auto, adjusted"})["source"] == "auto, adjusted"
    with pytest.raises(ValueError, match="source"):
        parse_landmarks({**base, "source": "robot"})
    calibration = fit(parse_landmarks(base)["points"], W, H)
    calibration.source = "auto"
    assert calibration.summary()["landmark_source"] == "auto"
