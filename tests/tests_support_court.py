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


def render_court(court_to_image, standard="nba", width=1920, height=1080, seed=0, players=6, line_px=5):
    """A synthetic broadcast frame: wood floor with the template's painted lines, a painted lane,
    a textured crowd beyond the far sideline and dark player-sized boxes on the floor."""
    import cv2

    from app.court import floor_polygon, project_lines, template

    rng = np.random.default_rng(seed)
    court = template(standard)
    # Crowd: dense random texture everywhere, then the floor painted over it.
    frame = rng.integers(0, 255, (height // 8, width // 8, 3), dtype=np.uint8)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
    floor = floor_polygon(court_to_image, court, apron=6.)
    wood = np.empty((height, width, 3), np.uint8)
    wood[:] = (120, 170, 215)  # BGR tan
    wood = cv2.add(wood, rng.normal(0, 4, wood.shape).astype(np.int16).clip(-12, 12).astype(np.uint8), dtype=cv2.CV_8U)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.clip(np.rint(floor), -4 * width, 4 * width).astype(np.int32)], 255)
    frame[mask > 0] = wood[mask > 0]
    d = court.dims
    lane = np.asarray([(-d["lane_width"] / 2, 0.), (d["lane_width"] / 2, 0.), (d["lane_width"] / 2, d["ft_y"]),
                       (-d["lane_width"] / 2, d["ft_y"])])
    from app.court import apply
    cv2.fillPoly(frame, [np.rint(apply(court_to_image, lane)).astype(np.int32)], (140, 60, 30))
    cv2.polylines(frame, project_lines(court_to_image, court, width, height, step=.25), False, (245, 245, 245),
                  line_px, cv2.LINE_AA)
    for _ in range(players):
        x, y = rng.uniform(.15, .85) * width, rng.uniform(.35, .8) * height
        w, h = .025 * width, .16 * height
        cv2.rectangle(frame, (int(x - w / 2), int(y - h)), (int(x + w / 2), int(y)), (40, 30, 30), -1)
    return frame
