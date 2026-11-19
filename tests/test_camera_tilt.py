import math

import cv2
import numpy as np
import pytest

from app.camera_tilt import tilt_from_edges, upright_edges


def scene(tilt_deg, convergence=0.0, columns=(-0.7, -0.4, -0.1, 0.2, 0.5, 0.8), size=(1280, 720)):
    """A wall of posts filmed by a camera rolled ``tilt_deg``; ``convergence`` is the extra
    lean per frame height from the centre that a camera pointing up or down adds."""
    width, height = size
    frame = np.full((height, width, 3), 40, np.uint8)
    for u in columns:
        lean = math.radians(tilt_deg + convergence * u)
        x = width / 2 + u * height
        top = (round(x + 0.3 * height * math.tan(lean)), round(0.1 * height))
        bottom = (round(x - 0.3 * height * math.tan(lean)), round(0.7 * height))
        cv2.line(frame, top, bottom, (230, 230, 230), 9, cv2.LINE_AA)
    return frame


@pytest.mark.parametrize("tilt", [-5.0, -1.5, 0.0, 3.0])
def test_tilt_is_read_from_upright_edges(tilt):
    assert tilt_from_edges(upright_edges(scene(tilt))) == pytest.approx(tilt, abs=0.4)


def test_a_camera_pointing_up_does_not_change_the_tilt():
    assert tilt_from_edges(upright_edges(scene(2.0, convergence=-4.0))) == pytest.approx(2.0, abs=0.5)


def test_no_tilt_without_edges_that_settle_it():
    assert tilt_from_edges(upright_edges(np.full((720, 1280, 3), 40, np.uint8))) is None
    # Posts on one side only: with a camera pointing up or down their lean is not the tilt.
    assert tilt_from_edges(upright_edges(scene(2.0, columns=(0.6, 0.7, 0.8)))) is None
    # One post is one object, and it may itself lean.
    assert tilt_from_edges(upright_edges(scene(2.0, columns=(0.0,)))) is None
    # Edges that disagree with each other.
    rng = np.random.default_rng(0)
    scattered = np.column_stack([rng.uniform(-0.8, 0.8, 40), rng.uniform(-15, 15, 40), np.full(40, 0.2)])
    assert tilt_from_edges(scattered) is None


def test_the_shooter_and_the_floor_are_left_out():
    frame = scene(0.0)
    cv2.line(frame, (600, 200), (660, 500), (255, 255, 255), 9, cv2.LINE_AA)    # a leaning limb
    cv2.line(frame, (300, 560), (340, 700), (255, 255, 255), 9, cv2.LINE_AA)    # a floor line
    leans = lambda edges: sorted({round(abs(lean)) for _, lean, _ in edges})  # noqa: E731
    assert max(leans(upright_edges(frame))) > 5
    assert leans(upright_edges(frame, [(0.42, 0.2, 0.56, 0.75)], below=0.75)) == [0]
