"""Court template, image-to-floor homography and floor geometry in feet.

Court coordinates are feet on the calibrated half court: x runs across the
court from the lane's centre line (left/right as drawn on the calibration
diagram), y runs from the baseline (0) toward half court. A homography is only
valid for points on the floor, so the ball, the rim and airborne feet are never
mapped through it.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

METRE = 1 / .3048
# Dimensions in feet. rim_y is the floor point under the rim centre; ft_y is the
# free-throw line; the three-point corner lines run parallel to the sidelines
# at ±three_corner_x until they meet the arc of three_radius around the rim.
STANDARDS = {
    "nba": {"label": "NBA", "length": 94., "width": 50., "rim_y": 5.25, "backboard_y": 4.,
            "lane_width": 16., "ft_y": 19., "ft_radius": 6., "three_radius": 23.75,
            "three_corner_x": 22., "restricted_radius": 4., "center_radius": 6.},
    "fiba": {"label": "FIBA", "length": 28 * METRE, "width": 15 * METRE, "rim_y": 1.575 * METRE,
             "backboard_y": 1.2 * METRE, "lane_width": 4.9 * METRE, "ft_y": 5.8 * METRE,
             "ft_radius": 1.8 * METRE, "three_radius": 6.75 * METRE, "three_corner_x": 6.6 * METRE,
             "restricted_radius": 1.25 * METRE, "center_radius": 1.8 * METRE},
    "ncaa": {"label": "NCAA", "length": 94., "width": 50., "rim_y": 5.25, "backboard_y": 4.,
             "lane_width": 12., "ft_y": 19., "ft_radius": 6., "three_radius": 22 + 1.75 / 12,
             "three_corner_x": 21 + 7.875 / 12, "restricted_radius": 4., "center_radius": 6.},
    "nfhs": {"label": "High school (NFHS)", "length": 84., "width": 50., "rim_y": 5.25, "backboard_y": 4.,
             "lane_width": 12., "ft_y": 19., "ft_radius": 6., "three_radius": 19.75,
             "three_corner_x": 19.75, "restricted_radius": None, "center_radius": 6.},
}
LANDMARK_LABELS = {
    "baseline_left": "Baseline / left sideline corner",
    "baseline_right": "Baseline / right sideline corner",
    "three_base_left": "Left corner three / baseline",
    "three_base_right": "Right corner three / baseline",
    "lane_base_left": "Lane / baseline, left",
    "lane_base_right": "Lane / baseline, right",
    "ft_left": "Free-throw line, left end",
    "ft_right": "Free-throw line, right end",
    "ft_top": "Top of free-throw circle",
    "restricted_top": "Top of restricted-area arc",
    "three_break_left": "Left corner line meets the arc",
    "three_break_right": "Right corner line meets the arc",
    "three_top": "Top of three-point arc",
    "half_left": "Half-court line / left sideline",
    "half_right": "Half-court line / right sideline",
    "center_left": "Centre circle / half-court line, left",
    "center_right": "Centre circle / half-court line, right",
}
ZONES = ("paint", "midrange", "corner_three", "above_break_three")


@dataclass(frozen=True)
class Template:
    standard: str
    dims: dict
    landmarks: dict[str, tuple[float, float]]
    lines: list[list[tuple[float, float]]]

    @property
    def rim(self) -> tuple[float, float]:
        return 0., self.dims["rim_y"]

    @property
    def three_break_y(self) -> float:
        d = self.dims
        return d["rim_y"] + math.sqrt(max(0., d["three_radius"] ** 2 - d["three_corner_x"] ** 2))


def _arc(cx, cy, radius, start, stop, steps=24):
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a))
            for a in np.linspace(start, stop, steps)]


def template(standard: str = "nba") -> Template:
    if standard not in STANDARDS:
        raise ValueError(f"Court standard must be one of: {', '.join(STANDARDS)}")
    d = STANDARDS[standard]
    w, half, lane, rim_y = d["width"] / 2, d["length"] / 2, d["lane_width"] / 2, d["rim_y"]
    corner, radius = d["three_corner_x"], d["three_radius"]
    break_y = rim_y + math.sqrt(max(0., radius ** 2 - corner ** 2))
    theta = math.atan2(break_y - rim_y, corner)
    landmarks = {
        "baseline_left": (-w, 0.), "baseline_right": (w, 0.),
        "three_base_left": (-corner, 0.), "three_base_right": (corner, 0.),
        "lane_base_left": (-lane, 0.), "lane_base_right": (lane, 0.),
        "ft_left": (-lane, d["ft_y"]), "ft_right": (lane, d["ft_y"]),
        "ft_top": (0., d["ft_y"] + d["ft_radius"]),
        "three_break_left": (-corner, break_y), "three_break_right": (corner, break_y),
        "three_top": (0., rim_y + radius),
        "half_left": (-w, half), "half_right": (w, half),
        "center_left": (-d["center_radius"], half), "center_right": (d["center_radius"], half),
    }
    if d["restricted_radius"]:
        landmarks["restricted_top"] = (0., rim_y + d["restricted_radius"])
    near = [
        [(-w, 0.), (w, 0.)],
        [(-lane, 0.), (-lane, d["ft_y"]), (lane, d["ft_y"]), (lane, 0.)],
        _arc(0., d["ft_y"], d["ft_radius"], 0., 2 * math.pi, 40),
        [(-corner, 0.), (-corner, break_y)] + _arc(0., rim_y, radius, math.pi - theta, theta, 40)[1:-1]
        + [(corner, break_y), (corner, 0.)],
    ]
    if d["restricted_radius"]:
        r = d["restricted_radius"]
        near.append([(-r, d["backboard_y"]), (-r, rim_y)] + _arc(0., rim_y, r, math.pi, 0., 16)
                    + [(r, d["backboard_y"])])
    far = [[(x, d["length"] - y) for x, y in line] for line in near]
    lines = near + far + [
        [(-w, 0.), (-w, d["length"])], [(w, 0.), (w, d["length"])], [(-w, half), (w, half)],
        _arc(0., half, d["center_radius"], 0., 2 * math.pi, 40),
    ]
    return Template(standard, dict(d), landmarks, lines)


def template_json() -> dict:
    """Landmarks and line polylines for every standard, for the web diagram."""
    output = {}
    for name in STANDARDS:
        t = template(name)
        output[name] = {
            "label": t.dims["label"], "length": t.dims["length"], "width": t.dims["width"],
            "rim": [round(v, 4) for v in t.rim],
            "landmarks": {key: {"label": LANDMARK_LABELS[key], "court": [round(v, 4) for v in point]}
                          for key, point in t.landmarks.items()},
            "lines": [[[round(x, 4), round(y, 4)] for x, y in line] for line in t.lines],
        }
    return output


def parse_landmarks(payload) -> dict:
    """Validate the court_landmarks form field.

    {"standard": "nba", "time_s": 0.0 | "frame": 0, "points": [{"id": ..., "image": [x, y]}, ...]}
    with image points normalized to the frame. Raises ValueError with a user-facing message.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            raise ValueError("Court landmarks must be JSON.") from None
    if not isinstance(payload, dict):
        raise ValueError("Court landmarks must be an object with standard, time_s and points.")
    standard = payload.get("standard", "nba")
    if standard not in STANDARDS:
        raise ValueError(f"Court standard must be one of: {', '.join(STANDARDS)}.")
    frame, time_s = payload.get("frame"), payload.get("time_s")
    if frame is not None:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("Court landmark frame must be a non-negative frame number.")
    elif time_s is not None:
        if (isinstance(time_s, bool) or not isinstance(time_s, (int, float))
                or not math.isfinite(time_s) or time_s < 0):
            raise ValueError("Court landmark time_s must be a non-negative video time.")
        time_s = float(time_s)
    else:
        raise ValueError("Court landmarks need the video time (time_s) or frame they were marked on.")
    court = template(standard)
    points, seen = [], set()
    for item in payload.get("points") or []:
        if not isinstance(item, dict) or item.get("id") not in court.landmarks or item["id"] in seen:
            raise ValueError("Each court landmark needs a distinct id from the court diagram.")
        image = item.get("image")
        if (not isinstance(image, (list, tuple)) or len(image) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) and 0 <= v <= 1 for v in image)):
            raise ValueError("Court landmark image points must be normalized [x, y] in the frame.")
        seen.add(item["id"])
        points.append((item["id"], (float(image[0]), float(image[1]))))
    if len(points) < 4:
        raise ValueError("Mark at least 4 court landmarks (5 or more lets Ballform check the fit).")
    if not general_position([court.landmarks[key] for key, _ in points]):
        raise ValueError("Court landmarks need 4 points with no 3 on one court line, "
                         "for example the lane corners plus a three-point landmark.")
    return {"standard": standard, "frame": frame, "time_s": time_s if frame is None else None,
            "points": points}


def general_position(points) -> bool:
    """True when some 4 points have no 3 collinear (a homography is then determined)."""
    def area(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
    return any(all(area(*triple) > 1. for triple in itertools.combinations(quad, 3))
               for quad in itertools.combinations(points, 4))


def apply(h: np.ndarray, points) -> np.ndarray:
    """Apply a homography to (N, 2) points; rows behind the camera come back as nan."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    mapped = np.c_[points, np.ones(len(points))] @ np.asarray(h, dtype=float).T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = mapped[:, :2] / mapped[:, 2:3]
    out[~(np.abs(mapped[:, 2]) > 1e-12)] = np.nan
    return out


@dataclass
class Calibration:
    standard: str
    image_to_court: np.ndarray
    court_to_image: np.ndarray
    ids: list[str]
    image_px: np.ndarray
    errors_px: list[float]
    leave_one_out_px: list[float] | None
    outliers: list[str] = field(default_factory=list)

    def extrapolation_ft(self, court_point) -> float:
        """Feet outside the area spanned by the used clicks (0 inside it).

        On a test broadcast, clicks bunched around the lane agreed within 2 px
        yet put half court 113 px (about 2.5 ft) away from where a fit including
        a half-court click put it, while spots inside the arc moved by under
        1 ft. The leave-one-out spread did not reveal that, so extrapolation is
        reported directly.
        """
        court = template(self.standard)
        used = np.asarray([court.landmarks[key] for key in self.ids if key not in self.outliers], np.float32)
        hull = cv2.convexHull(used)
        inside = cv2.pointPolygonTest(hull, (float(court_point[0]), float(court_point[1])), True)
        return max(0., -float(inside))

    def summary(self) -> dict:
        def stats(values):
            # A point whose removal leaves no 4 points in general position has no leave-one-out error.
            values = [v for v in values or [] if math.isfinite(v)]
            return None if not values else {"rms": round(float(np.sqrt(np.mean(np.square(values)))), 2),
                                            "max": round(float(np.max(values)), 2), "points": len(values)}
        return {
            "standard": self.standard, "points": len(self.ids),
            "clicked_error_px": stats(self.errors_px),
            "leave_one_out_error_px": stats(self.leave_one_out_px),
            "per_point_error_px": {key: round(e, 2) for key, e in zip(self.ids, self.errors_px)},
            "per_point_leave_one_out_px": ({key: round(e, 2) if math.isfinite(e) else None
                                            for key, e in zip(self.ids, self.leave_one_out_px)}
                                           if self.leave_one_out_px else None),
            "dropped_outliers": self.outliers,
        }


def _fit(court_pts: np.ndarray, image_px: np.ndarray) -> np.ndarray | None:
    h, _ = cv2.findHomography(court_pts.astype(np.float64), image_px.astype(np.float64), 0)
    return h if h is not None and np.all(np.isfinite(h)) else None


def _leave_one_out(court_pts: np.ndarray, image_px: np.ndarray) -> list[float]:
    errors = []
    for index in range(len(court_pts)):
        keep = np.arange(len(court_pts)) != index
        held = _fit(court_pts[keep], image_px[keep]) if general_position(court_pts[keep]) else None
        errors.append(float(np.linalg.norm(apply(held, court_pts[index:index + 1])[0] - image_px[index]))
                      if held is not None else float("nan"))
    return errors


def fit(points: list[tuple[str, tuple[float, float]]], width: int, height: int,
        standard: str = "nba") -> Calibration:
    """Floor homography from clicked landmarks (normalized image points), by least squares.

    Four points determine a homography exactly, so their error is always zero;
    with five or more, each point's leave-one-out error (refit without it and
    reproject it) shows whether the clicks agree. From six points, the click
    with the largest residual is dropped as a misclick while that residual
    exceeds 1.5% of the frame height and three times the median. RANSAC and
    leave-one-out rejection were tried first: 4-point hypotheses extrapolate so
    poorly across a broadcast view that both discarded a correct, lone
    half-court click on a test clip.
    """
    court = template(standard)
    ids = [key for key, _ in points]
    court_pts = np.asarray([court.landmarks[key] for key in ids], dtype=float)
    image_px = np.asarray([(x * width, y * height) for _, (x, y) in points], dtype=float)
    used = np.ones(len(ids), bool)
    outliers = []
    h = _fit(court_pts, image_px)
    while h is not None and used.sum() >= 6:
        residual = np.linalg.norm(apply(h, court_pts[used]) - image_px[used], axis=1)
        worst = int(np.argmax(residual))
        if residual[worst] <= max(.015 * height, 3 * float(np.median(residual))):
            break
        index = int(np.flatnonzero(used)[worst])
        trial = used.copy()
        trial[index] = False
        if not general_position(court_pts[trial]):
            break
        used = trial
        outliers.append(ids[index])
        h = _fit(court_pts[used], image_px[used])
    if h is None:
        raise ValueError("Could not compute a court mapping from these landmarks.")
    w = np.c_[court_pts, np.ones(len(court_pts))] @ h[2]
    if not (np.all(w > 0) or np.all(w < 0)):
        raise ValueError("The clicked landmarks do not form a consistent view of the floor; check left/right pairs.")
    errors = np.linalg.norm(apply(h, court_pts) - image_px, axis=1).tolist()
    loo = _leave_one_out(court_pts, image_px) if len(ids) >= 5 else None
    return Calibration(standard, np.linalg.inv(h), h, ids, image_px, errors, loo, outliers)


@dataclass(frozen=True)
class Camera:
    """Pinhole camera recovered from a floor homography (square pixels, centred principal point)."""
    k: np.ndarray
    r: np.ndarray       # world (court x, y, up) -> camera
    center: np.ndarray  # camera position in court feet, z = height above the floor

    @property
    def focal_px(self) -> float:
        return float(self.k[0, 0])

    def ray(self, image_point) -> np.ndarray:
        direction = self.r.T @ np.linalg.solve(self.k, np.array([image_point[0], image_point[1], 1.]))
        return direction / np.linalg.norm(direction)


def camera_from_homography(court_to_image: np.ndarray, width: int, height: int) -> Camera | None:
    """Recover focal length, pose and position from one floor homography.

    With the principal point at the image centre and square pixels, the two
    rotation columns K^-1 h1 and K^-1 h2 must be orthogonal and equally long,
    which fixes the focal length. Returns None when the view is too
    fronto-parallel for that to be well conditioned or the result is implausible.
    """
    shift = np.array([[1., 0., -width / 2], [0., 1., -height / 2], [0., 0., 1.]])
    h = shift @ np.asarray(court_to_image, dtype=float)
    h = h / np.linalg.norm(h[:, 0])
    (a, b, c), (d, e, f) = h[:, 0], h[:, 1]
    rows = np.array([[a * d + b * e, c * f], [a * a + b * b - d * d - e * e, c * c - f * f]])
    denom = float(rows[:, 0] @ rows[:, 0])
    if denom < 1e-18:
        return None
    inverse_f2 = -float(rows[:, 0] @ rows[:, 1]) / denom
    if not inverse_f2 > 0:
        return None
    focal = 1 / math.sqrt(inverse_f2)
    if not .3 * width <= focal <= 20 * width:
        return None
    k0 = np.diag([focal, focal, 1.])
    columns = np.linalg.solve(k0, h)
    scale = 2 / (np.linalg.norm(columns[:, 0]) + np.linalg.norm(columns[:, 1]))
    columns *= scale
    if columns[2, 2] < 0:  # the court must be in front of the camera
        columns = -columns
    r1, r2, t = columns[:, 0], columns[:, 1], columns[:, 2]
    u, _, vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
    rotation = u @ vt
    center = -rotation.T @ t
    if center[2] < 0:
        # A mirrored left/right labelling puts the camera under the floor; flip
        # the vertical axis so heights stay positive (distances are unaffected).
        rotation = rotation @ np.diag([1., 1., -1.])
        center = -rotation.T @ t
    k = k0.copy()
    k[0, 2], k[1, 2] = width / 2, height / 2
    return Camera(k, rotation, center)


def vertical_plane_distance(camera: Camera, floor_point, image_a, image_b) -> float | None:
    """Feet between two image points assumed to lie in the vertical plane through floor_point facing the camera."""
    anchor = np.array([floor_point[0], floor_point[1], 0.])
    normal = camera.center - anchor
    normal[2] = 0.
    if np.linalg.norm(normal) < 1e-6:
        return None
    normal /= np.linalg.norm(normal)
    hits = []
    for point in (image_a, image_b):
        ray = camera.ray(point)
        denominator = float(normal @ ray)
        if abs(denominator) < 1e-6:
            return None
        distance = float(normal @ (anchor - camera.center)) / denominator
        if distance <= 0:
            return None
        hits.append(camera.center + distance * ray)
    return float(np.linalg.norm(hits[0] - hits[1]))


def zone(point, court: Template) -> str | None:
    """Shot zone for a floor point on the calibrated half; None off the court or past half court."""
    x, y = point
    d = court.dims
    if abs(x) > d["width"] / 2 + 1 or y < -1 or y > d["length"] / 2:
        return None
    if abs(x) <= d["lane_width"] / 2 and y <= d["ft_y"]:
        return "paint"
    if y <= court.three_break_y:
        return "corner_three" if abs(x) >= d["three_corner_x"] else "midrange"
    return "above_break_three" if math.dist(point, court.rim) >= d["three_radius"] else "midrange"


def three_point_margin(point, court: Template) -> float:
    """Signed feet from the three-point line (positive = behind it)."""
    x, y = point
    d = court.dims
    if y <= court.three_break_y:
        return abs(x) - d["three_corner_x"]
    return math.dist(point, court.rim) - d["three_radius"]


def project_lines(court_to_image: np.ndarray, court: Template, width: int, height: int,
                  step: float = 1.) -> list[np.ndarray]:
    """Court lines as image polylines (pixel int32), split where they leave the view or pass behind the camera."""
    polylines = []
    margin = max(width, height)
    for line in court.lines:
        dense = []
        for a, b in zip(line, line[1:]):
            count = max(1, int(math.dist(a, b) / step))
            dense.extend((a[0] + (b[0] - a[0]) * i / count, a[1] + (b[1] - a[1]) * i / count)
                         for i in range(count))
        dense.append(line[-1])
        mapped = np.c_[np.asarray(dense), np.ones(len(dense))] @ court_to_image.T
        valid = mapped[:, 2] > 1e-9
        image = np.full((len(dense), 2), np.nan)
        image[valid] = mapped[valid, :2] / mapped[valid, 2:3]
        valid &= np.all(np.abs(image - (width / 2, height / 2)) < margin, axis=1)
        run = []
        for ok, point in zip(valid, image):
            if ok:
                run.append(point)
            elif len(run) > 1:
                polylines.append(np.rint(run).astype(np.int32))
                run = []
            else:
                run = []
        if len(run) > 1:
            polylines.append(np.rint(run).astype(np.int32))
    return polylines


def floor_polygon(court_to_image: np.ndarray, court: Template, apron: float = 3.) -> np.ndarray | None:
    """The whole floor (court plus an apron) in image pixels, clipped to the half-space in front of the camera."""
    w, length = court.dims["width"] / 2 + apron, court.dims["length"] + apron
    corners = [(-w, -apron), (w, -apron), (w, length), (-w, length)]
    homogeneous = [court_to_image @ np.array([x, y, 1.]) for x, y in corners]
    clipped = []
    eps = 1e-6
    for current, following in zip(homogeneous, homogeneous[1:] + homogeneous[:1]):
        if current[2] > eps:
            clipped.append(current)
        if (current[2] > eps) != (following[2] > eps):
            t = (eps - current[2]) / (following[2] - current[2])
            clipped.append(current + t * (following - current))
    if len(clipped) < 3:
        return None
    return np.asarray([p[:2] / p[2] for p in clipped], dtype=np.float32)


def floor_point(pose, width: int, height: int) -> tuple[tuple[float, float], str] | None:
    """A player's floor contact in pixels: the midpoint of visible ankles, else the pose box's bottom centre."""
    ankles = [pose.landmarks.get(name) for name in ("left_ankle", "right_ankle")]
    ankles = [p for p in ankles if p and len(p) >= 3 and p[2] >= .4 and all(math.isfinite(v) for v in p)]
    if ankles:
        return ((sum(p[0] for p in ankles) / len(ankles) * width, sum(p[1] for p in ankles) / len(ankles) * height),
                "ankles" if len(ankles) == 2 else "one ankle")
    box = getattr(pose, "box", None)
    if box is not None and all(math.isfinite(v) for v in box):
        return ((box[0] + box[2]) / 2 * width, box[3] * height), "pose box"
    return None


def lowest_foot_y(pose, height: int) -> float | None:
    """Image y (pixels) of the lower visible ankle, else the pose box bottom."""
    ankles = [pose.landmarks.get(name) for name in ("left_ankle", "right_ankle")]
    ys = [p[1] * height for p in ankles if p and len(p) >= 3 and p[2] >= .4 and math.isfinite(p[1])]
    if ys:
        return max(ys)
    box = getattr(pose, "box", None)
    return box[3] * height if box is not None and math.isfinite(box[3]) else None


@dataclass(frozen=True)
class Takeoff:
    grounded: list[int]   # indices of the samples used for the floor position (latest last)
    rise_torso: float     # lowest-foot rise from take-off to the last sample, in torso lengths
    jumped: bool


def takeoff(foot_y: list[float | None], torso_px: float, tolerance: float = .08,
            min_rise: float = .2, keep: int = 3) -> Takeoff | None:
    """Find the last grounded samples before a jump.

    foot_y holds the lower foot's image y (larger is lower), with camera motion
    removed, for consecutive samples ending at release. Walking back from
    release, the foot descends to the floor and then stays level while the
    player plants; an earlier sample where the foot is clearly higher is a
    previous stride, so the walk stops there. The latest samples within
    `tolerance` torso lengths of the floor level are the take-off. A rise under
    `min_rise` torso lengths means no jump was seen (the shooter may have been
    on the floor at release).
    """
    if not torso_px or torso_px <= 0:
        return None
    tol = tolerance * torso_px
    indices = [i for i in range(len(foot_y) - 1, -1, -1) if foot_y[i] is not None and math.isfinite(foot_y[i])]
    if len(indices) < 2:
        return None
    last = indices[0]
    floor = foot_y[last]
    walked = [last]
    for i in indices[1:]:
        if foot_y[i] < floor - tol:
            break
        floor = max(floor, foot_y[i])
        walked.append(i)
    grounded = [i for i in walked if foot_y[i] >= floor - tol][:keep]
    rise = (floor - foot_y[last]) / torso_px
    return Takeoff(sorted(grounded), rise, rise >= min_rise)
