import pytest

from app.geometry import elevation_angle, interpolate_crossing, joint_angle


def test_joint_angle():
    assert joint_angle((0, 0), (0, 1), (1, 1)) == pytest.approx(90)
    assert joint_angle((0, 0), (0, 1), (0, 2)) == pytest.approx(180)


def test_elevation_angle_image_coordinates():
    assert elevation_angle((0, 1), (1, 0)) == pytest.approx(45)


def test_interpolate_crossing():
    assert interpolate_crossing((0, 0), (1, 2), 1) == pytest.approx(.5)
    assert interpolate_crossing((0, 0), (1, .5), 1) is None
