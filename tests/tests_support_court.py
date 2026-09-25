"""Shared synthetic broadcast camera for court-calibration tests."""
import numpy as np


def synthetic_camera(position=(-100., 40., 35.), look_at=(0., 20., 0.), focal=3000., width=1920, height=1080):
    """Intrinsics, world (court x, y, up) -> camera rotation, and the court -> image homography."""
    position, forward = np.asarray(position), np.asarray(look_at) - np.asarray(position)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0., 0., 1.))
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    k = np.array([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]])
    return k, rotation, k @ np.c_[rotation[:, 0], rotation[:, 1], -rotation @ position]
