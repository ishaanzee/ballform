"""Propose a court calibration from one video frame by fitting the painted lines.

The proposal is only a starting point for the calibration preview: it returns the
named template landmarks where the fitted court puts them, and the user checks
the drawn lines and moves or removes points before analyzing.

Steps, at 960 px working width:

1. Line pixels: thin lines of any colour (a pixel that differs from both sides
   while the two sides match) plus strong Lab colour edges (painted lanes and
   aprons end in an edge rather than a line). Neighbourhoods dense with edges,
   such as crowds and graphics, are dropped; the floor is mostly plain.
2. Straight lines from the line segment detector, merged where collinear.
3. The lines are split into the two floor directions by their vanishing points.
4. Hypotheses: two lines of each direction matched to two template lines of
   constant court y (baseline, free-throw line, half court) and two of constant
   x (sidelines, corner threes, lane sides). When only one line of a direction
   is visible, three lines plus a camera with square pixels and a centred
   principal point leave one free parameter, and the focal length is swept.
   Views no real camera could take, and matches whose detected segments run
   past the end of their template line, are discarded.
5. Scoring: the share of projected template lines that lands on line pixels
   running the same way, counted over distinct image cells so a court squashed
   onto one strong line cannot score every template line on the same pixels.
6. Refinement of the best few by point-to-line least squares, keeping the
   best-scoring stage.
7. A proposal is returned only when at least MIN_SUPPORT of the template lines
   in view land on line pixels and at least MIN_LINES separate lines are found.
   On the test frames correct fits scored 0.48-0.72 and wrong ones 0.31-0.43;
   two broadcast graphics frames scored over 0.5 but showed only 2 lines, which
   is what MIN_LINES is for. Both thresholds rest on few frames.

No learned model is involved; the only inputs are the frame and the court standard.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.court import Template, apply, general_position, template

WORK_WIDTH = 960
MIN_SUPPORT = .45        # share of the in-view template lines that must land on painted lines
MIN_LINES = 4            # distinct template lines that must be found
EDGE_MARGIN = .01        # proposed landmarks must lie this far inside the frame (share of width/height)


@dataclass
class Line:
    eq: np.ndarray        # a x + b y + c = 0 with unit (a, b)
    weight: float         # covered length in working pixels
    ends: np.ndarray      # (2, 2) ends of the longest covered run

    @property
    def mid(self) -> np.ndarray:
        return self.ends.mean(axis=0)

    @property
    def direction(self) -> np.ndarray:
        return np.array([self.eq[1], -self.eq[0]])


@dataclass
class Proposal:
    ok: bool
    reason: str | None
    confidence: float
    standard: str
    points: list[tuple[str, tuple[float, float]]] = field(default_factory=list)  # normalized image points
    court_to_image: np.ndarray | None = None                                      # full-resolution pixels
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "confidence": round(self.confidence, 3),
                "standard": self.standard,
                "points": [{"id": key, "image": [round(x, 5), round(y, 5)]} for key, (x, y) in self.points],
                "court_to_image": None if self.court_to_image is None
                else [[float(v) for v in row] for row in self.court_to_image],
                "diagnostics": self.diagnostics}


# --- line pixels -------------------------------------------------------------------------------

def _shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """a[y - dy, x - dx], with the uncovered border left as it was."""
    out = a.copy()
    h, w = a.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = a[max(-dy, 0):h + min(-dy, 0),
                                                                  max(-dx, 0):w + min(-dx, 0)]
    return out


def line_mask(frame: np.ndarray, boxes=(), ridge_px: int = 3, ridge_min: float = 14.,
              clutter_max: float | None = .3) -> np.ndarray:
    """Painted-line pixels (255) of a working-resolution BGR frame, of any colour.

    boxes are normalized (x1, y1, x2, y2) regions to ignore, such as players.
    """
    lab = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32), (3, 3), 0)
    ridge = np.zeros(lab.shape[:2], np.float32)
    for dx, dy in ((ridge_px, 0), (0, ridge_px), (ridge_px, ridge_px), (ridge_px, -ridge_px)):
        a, b = _shift(lab, dx, dy), _shift(lab, -dx, -dy)
        response = (np.minimum(np.linalg.norm(lab - a, axis=2), np.linalg.norm(lab - b, axis=2))
                    - np.linalg.norm(a - b, axis=2))
        ridge = np.maximum(ridge, response)
    mask = (ridge > ridge_min).astype(np.uint8) * 255
    lab8 = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2LAB)
    edge = np.zeros_like(mask)
    for channel in cv2.split(lab8):
        edge |= cv2.Canny(channel, 40, 90)
    mask |= edge
    if clutter_max is not None:
        density = cv2.boxFilter((edge > 0).astype(np.float32), -1, (25, 25))
        mask[density > clutter_max] = 0
    h, w = mask.shape
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (int(x1 * w), int(y1 * h)), (int(math.ceil(x2 * w)), int(math.ceil(y2 * h))), 0, -1)
    return mask


@dataclass
class Evidence:
    """Line pixels of one working-resolution frame, with the local line direction."""
    mask: np.ndarray
    dist: np.ndarray            # px to the nearest line pixel
    nearest: np.ndarray         # (H, W) index into `pixels` of the nearest line pixel
    pixels: np.ndarray          # (N, 2) x, y of line pixels
    direction: np.ndarray       # (N, 2) unit line direction at each line pixel
    near_direction: np.ndarray  # (H, W, 2) direction of the nearest line pixel

    @classmethod
    def build(cls, small: np.ndarray, boxes=(), **options) -> "Evidence":
        mask = line_mask(small, boxes, **options)
        lab = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 1.2)
        gx, gy = cv2.Sobel(lab, cv2.CV_32F, 1, 0), cv2.Sobel(lab, cv2.CV_32F, 0, 1)
        jxx = cv2.GaussianBlur((gx * gx).sum(axis=2), (0, 0), 2.5)
        jyy = cv2.GaussianBlur((gy * gy).sum(axis=2), (0, 0), 2.5)
        jxy = cv2.GaussianBlur((gx * gy).sum(axis=2), (0, 0), 2.5)
        normal = .5 * np.arctan2(2 * jxy, jxx - jyy)   # dominant gradient orientation (structure tensor)
        rows, cols = np.nonzero(mask)
        pixels = np.column_stack([cols, rows]).astype(np.float32)
        theta = normal[rows, cols]
        direction = np.column_stack([-np.sin(theta), np.cos(theta)]).astype(np.float32)
        if len(rows):
            dist, labels = cv2.distanceTransformWithLabels(255 - mask, cv2.DIST_L2, 5,
                                                           labelType=cv2.DIST_LABEL_PIXEL)
            nearest = (labels - 1).astype(np.int32)  # labels follow the scan order of the line pixels
            near_direction = direction[nearest]
        else:
            dist = np.full(mask.shape, np.inf, np.float32)
            nearest = np.zeros(mask.shape, np.int32)
            pixels, direction = np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32)
            near_direction = np.zeros(mask.shape + (2,), np.float32)
        return cls(mask, dist, nearest, pixels, direction, near_direction)


# --- straight lines and their two floor directions ----------------------------------------------

def candidate_lines(small: np.ndarray, mask: np.ndarray, count: int = 40, min_length: float = 20.,
                    merge_px: float = 3.5, merge_deg: float = 1.5, max_gap: float = 40.) -> list[Line]:
    """Long straight lines, strongest first.

    Segments come from the line segment detector on the Lab channels and are kept where
    the line mask supports them. Collinear pieces, including the two edges of one painted
    line, are merged; a line's ends are those of its longest run with gaps under max_gap.
    """
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    lsd = cv2.createLineSegmentDetector()
    support = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    found = [lsd.detect(channel)[0] for channel in cv2.split(lab)]
    found = [f.reshape(-1, 4) for f in found if f is not None]
    if not found:
        return []
    segments = np.vstack(found)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    segments, lengths = segments[lengths >= min_length], lengths[lengths >= min_length]
    ts = np.linspace(0, 1, 11)
    h, w = mask.shape
    xs = np.clip((segments[:, None, 0] + ts * (segments[:, None, 2] - segments[:, None, 0])).astype(int), 0, w - 1)
    ys = np.clip((segments[:, None, 1] + ts * (segments[:, None, 3] - segments[:, None, 1])).astype(int), 0, h - 1)
    supported = support[ys, xs].mean(axis=1) >= .6
    segments, lengths = segments[supported], lengths[supported]
    groups: list[list] = []  # [unit normal, rho, [segments]]
    cos_merge = math.cos(math.radians(merge_deg))
    for index in np.argsort(-lengths):
        x1, y1, x2, y2 = segments[index]
        direction = np.array([x2 - x1, y2 - y1]) / lengths[index]
        normal = np.array([-direction[1], direction[0]])
        for group in groups:
            n, rho = group[0], group[1]
            if (abs(float(n @ normal)) >= cos_merge and abs(n @ (x1, y1) - rho) < merge_px
                    and abs(n @ (x2, y2) - rho) < merge_px):
                group[2].append(segments[index])
                break
        else:
            groups.append([normal, float(normal @ (x1, y1)), [segments[index]]])
    out = []
    for _, _, members in groups:
        pts = np.asarray(members, np.float32).reshape(-1, 2)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, .01, .01).ravel()
        along = np.array([vx, vy], float)
        origin = np.array([x0, y0], float)
        spans = np.sort((np.asarray(members, float).reshape(-1, 2, 2) - origin) @ along, axis=1)
        spans = spans[np.argsort(spans[:, 0])]
        runs, covered = [], 0.
        start, end, run_cover = spans[0, 0], spans[0, 1], spans[0, 1] - spans[0, 0]
        for a, b in spans[1:]:
            if a > end + max_gap:
                runs.append((run_cover, start, end))
                start, end, run_cover = a, b, b - a
            elif b > end:
                run_cover += b - max(a, end)
                end = b
        runs.append((run_cover, start, end))
        covered = sum(r[0] for r in runs)
        _, start, end = max(runs)
        normal = np.array([-along[1], along[0]])
        out.append(Line(np.array([normal[0], normal[1], -normal @ origin]), float(covered),
                        np.array([origin + start * along, origin + end * along])))
    out.sort(key=lambda line: -line.weight)
    return out[:count]


def _vanishing_point(lines: list[Line], width: int, height: int) -> np.ndarray:
    """Weighted least-squares meeting point of lines (homogeneous, unit length)."""
    scale = max(width, height) / 2
    to_pixels = np.array([[scale, 0, width / 2], [0, scale, height / 2], [0, 0, 1.]])  # lines map as l' = T^T l
    eqs = np.asarray([line.eq for line in lines]) @ to_pixels
    eqs /= np.linalg.norm(eqs[:, :2], axis=1, keepdims=True)
    weights = np.sqrt([line.weight for line in lines])
    v = np.linalg.svd(weights[:, None] * eqs)[2][-1]
    v = to_pixels @ v
    return v / np.linalg.norm(v)


def _families(lines: list[Line], width: int, height: int, per_family: int = 8, tolerance_deg: float = 2.,
              options: int = 3) -> list[tuple[list[Line], list[Line]]]:
    """Candidate splits of the lines into the two floor directions, best first.

    Painted court lines run in two perpendicular floor directions, and the lines of each
    meet at one vanishing point, outside the picture. Every pair of lines proposes a
    point and collects the lines pointing at it; points backed by mostly the same lines
    as a stronger one are dropped, and two sets are paired when every line of one crosses
    every line of the other at a clear angle.
    """
    if len(lines) < 3:
        return []
    margin = .1 * max(width, height)
    tol = math.sin(math.radians(tolerance_deg))
    eqs = np.asarray([line.eq for line in lines])
    mids = np.asarray([line.mid for line in lines])
    weights = np.asarray([line.weight for line in lines])
    directions = np.asarray([line.direction for line in lines])

    candidates: dict[tuple, float] = {}
    for a, b in itertools.combinations(range(len(lines)), 2):
        vp = np.cross(eqs[a], eqs[b])
        vp = vp / np.linalg.norm(vp)
        if abs(vp[2]) > 1e-12:
            x, y = vp[:2] / vp[2]
            if -margin < x < width + margin and -margin < y < height + margin:
                continue
        to_vp = vp[:2] - vp[2] * mids
        norm = np.linalg.norm(to_vp, axis=1)
        cross = np.abs(directions[:, 0] * to_vp[:, 1] - directions[:, 1] * to_vp[:, 0])
        members = (norm < 1e-12) | (cross <= tol * np.maximum(norm, 1e-12))
        if members.sum() >= 2:
            key = tuple(int(i) for i in np.flatnonzero(members))
            candidates.setdefault(key, float(weights[members].sum()))
    ranked: list[tuple[tuple, float]] = []
    for key, support in sorted(candidates.items(), key=lambda item: -item[1]):
        if all(len(set(key) & set(k)) <= .5 * len(key) for k, _ in ranked):
            ranked.append((key, support))
        if len(ranked) >= 12:
            break

    def top(key):
        return tuple(sorted(key, key=lambda i: -weights[i])[:per_family])

    def crossing(ka, kb, min_deg=8.):
        cosines = np.abs(directions[list(ka)] @ directions[list(kb)].T)
        return float(cosines.max()) <= math.cos(math.radians(min_deg))

    pairs = sorted(((sa + sb, ka, kb) for (ka, sa), (kb, sb) in itertools.combinations(ranked, 2)
                    if not set(ka) & set(kb) and crossing(top(ka), top(kb))), key=lambda item: -item[0])
    out, seen = [], set()
    for _, ka, kb in pairs:
        fa, fb = top(ka), top(kb)
        if frozenset([fa, fb]) in seen:
            continue
        seen.add(frozenset([fa, fb]))
        out.append(([lines[i] for i in fa], [lines[i] for i in fb]))
        if len(out) >= options:
            break
    return out


# --- hypotheses ------------------------------------------------------------------------------------

def _template_lines(court: Template):
    """Straight template lines as (axis, value, low, high): axis 0 is constant x, 1 constant y.

    low..high is the extent along the line, in feet.
    """
    d = court.dims
    w, length, lane, corner = d["width"] / 2, d["length"], d["lane_width"] / 2, d["three_corner_x"]
    along = [(0, s * w, 0., length) for s in (-1, 1)]
    along += [(0, s * corner, 0., court.three_break_y) for s in (-1, 1)]
    along += [(0, s * lane, 0., d["ft_y"]) for s in (-1, 1)]
    across = [(1, 0., -w, w), (1, d["ft_y"], -lane, lane), (1, length / 2, -w, w)]
    return sorted(along, key=lambda item: item[1]), across


class _Batch:
    """Collects hypotheses chunk by chunk, keeping only those `_keep` and `_plausible` accept.

    Each chunk comes with, per hypothesis, the detected line ends that must fall on their
    template line. Filtering as they arrive keeps memory to one chunk of candidates.
    """

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.kept: list[np.ndarray] = []
        self.total = 0

    def add(self, h, ends, axis, ranges):
        self.total += len(h)
        h = h[_keep(h, ends, axis, ranges, self.width, self.height)]
        self.kept.append(h[_plausible(h, self.width, self.height)])

    def arrays(self) -> np.ndarray:
        return np.concatenate(self.kept) if self.kept else np.empty((0, 3, 3))


def _keep(h, ends, axis, ranges, width, height, tolerance: float = 2.5) -> np.ndarray:
    """Views in front of the camera, not mirrored, and with every matched segment inside its template line.

    ends (N, S, 2, 2) image points; axis (N, S) 0 or 1; ranges (N, S, 2) feet.
    """
    n = len(h)
    finite = np.all(np.isfinite(h.reshape(n, -1)), axis=1)
    finite[finite] = np.abs(np.linalg.det(h[finite])) > 1e-18
    keep = np.zeros(n, bool)
    if not finite.any():
        return keep
    index = np.flatnonzero(finite)
    hf = h[index]
    inv = np.linalg.inv(hf)
    pts = ends[index].reshape(len(index), -1, 2)
    homog = np.concatenate([pts, np.ones(pts.shape[:2] + (1,))], axis=2)
    mapped = np.einsum("nij,npj->npi", inv, homog)
    with np.errstate(divide="ignore", invalid="ignore"):
        court = mapped[..., :2] / mapped[..., 2:3]
    # The line ends must be in front of the camera: w of their court points under h > 0.
    w = np.einsum("nj,npj->np", hf[:, 2], np.concatenate([court, np.ones(court.shape[:2] + (1,))], axis=2))
    court = court.reshape(len(index), -1, 2, 2)
    coordinate = np.where(axis[index][..., None] == 0, court[..., 1], court[..., 0])  # along the line
    low, high = ranges[index][..., :1], ranges[index][..., 1:]
    slack = tolerance + .05 * (high - low)
    inside = np.all(np.isfinite(coordinate) & (coordinate >= low - slack) & (coordinate <= high + slack), axis=(1, 2))
    front = np.all(w > 0, axis=1) | np.all(w < 0, axis=1)
    flip = np.where(np.all(w < 0, axis=1), -1., 1.)
    keep[index] = inside & front & (np.linalg.det(hf * flip[:, None, None]) > 0)
    h[index] *= flip[:, None, None]
    return keep


def _hypotheses_two_two(families, court: Template, width: int, height: int, batch: _Batch) -> None:
    """Two lines of each floor direction matched to two template lines of each."""
    along, across = _template_lines(court)
    scale_src, scale_dst = 50., float(max(width, height))
    for fam_y, fam_x, (i, j) in ((a, b, pair) for a, b in (families, families[::-1])
                                 for pair in itertools.combinations(a, 2)):
        src, dst, ends, axis, ranges = [], [], [], [], []
        for k, l in itertools.combinations(fam_x, 2):
            corners = [np.cross(a.eq, b.eq) for a in (i, j) for b in (k, l)]
            if any(abs(c[2]) < 1e-12 for c in corners):
                continue
            corners = [c[:2] / c[2] for c in corners]
            if any(not np.all(np.abs(c - (width / 2, height / 2)) < 4 * max(width, height)) for c in corners):
                continue
            seg = np.asarray([i.ends, j.ends, k.ends, l.ends])
            for ta, tb in itertools.permutations(across, 2):
                for tc, td in itertools.permutations(along, 2):
                    # corners are ordered (i,k), (i,l), (j,k), (j,l)
                    src.append([(tc[1], ta[1]), (td[1], ta[1]), (tc[1], tb[1]), (td[1], tb[1])])
                    dst.append(corners)
                    ends.append(seg)
                    axis.append([1, 1, 0, 0])
                    ranges.append([ta[2:], tb[2:], tc[2:], td[2:]])
        if not src:
            continue
        src, dst = np.asarray(src, float), np.asarray(dst, float)
        h = _batch_homography(src / scale_src, dst / scale_dst)
        h = np.diag([scale_dst, scale_dst, 1.]) @ h @ np.diag([1 / scale_src, 1 / scale_src, 1.])
        batch.add(h, np.asarray(ends), np.asarray(axis), np.asarray(ranges, float))


def _hypotheses_two_one(families, court: Template, width: int, height: int, batch: _Batch,
                        pairs_per_family: int = 6, singles: int = 4, focal_steps: int = 24) -> None:
    """Two parallel lines and one perpendicular line, for views that show a single line one way.

    Three lines fix six of the eight homography parameters and a camera with square pixels
    and a centred principal point fixes a seventh; the focal length is swept for the last.
    """
    along, across = _template_lines(court)
    centre = np.array([width / 2, height / 2])
    focals = np.geomspace(.6 * width, 6 * width, focal_steps)
    swap = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for fam_a, fam_b in (families, families[::-1]):
        for pair in itertools.combinations(fam_a[:pairs_per_family], 2):
            for single in fam_b[:singles]:
                seg = np.asarray([pair[0].ends, pair[1].ends, single.ends])
                for pair_lines, single_lines, swapped in ((along, across, False), (across, along, True)):
                    h, templates = _two_one(pair, single, pair_lines, single_lines, focals, centre)
                    if not len(h):
                        continue
                    if swapped:
                        h = h @ swap  # the pair was solved as constant-"x" lines in swapped court axes
                    n = len(h)
                    axis = np.tile([1, 1, 0] if swapped else [0, 0, 1], (n, 1))
                    ranges = np.asarray([[a[2:], b[2:], c[2:]] for a, b, c in templates], float)
                    batch.add(h, np.broadcast_to(seg, (n,) + seg.shape).copy(), axis, ranges)


def _two_one(pair, single, pair_lines, single_lines, focals, centre):
    """Homographies for one pair (as lines of constant first court coordinate) and one single line."""
    l1, l2, s = pair[0].eq, pair[1].eq, single.eq
    v_pair = np.cross(l1, l2)
    p1, p2 = np.cross(l1, s), np.cross(l2, s)
    d = v_pair[:2] - centre * v_pair[2]
    basis = np.column_stack([p1, p2])
    combos = [(a, b, c) for a, b in itertools.permutations(pair_lines, 2) for c in single_lines]
    x1 = np.array([a[1] for a, _, _ in combos])
    x2 = np.array([b[1] for _, b, _ in combos])
    y0 = np.array([c[1] for _, _, c in combos])
    hs, templates = [], []
    for f in focals:
        polar = np.array([d[0], d[1], -d @ centre + f * f * v_pair[2]])
        v_s = np.cross(s, polar)
        (a, b), *_ = np.linalg.lstsq(basis, v_s, rcond=None)
        k_inv = np.array([1 / f, 1 / f, 1.])
        shift = np.array([centre[0], centre[1], 0.])
        norm_pair = np.linalg.norm(k_inv * (v_pair - shift * v_pair[2]))
        if norm_pair < 1e-12:
            continue
        beta = np.linalg.norm(k_inv * (v_s - shift * v_s[2])) / norm_pair
        for sign in (1., -1.):
            col2 = sign * beta * v_pair
            origin = -x2[:, None] * a * p1 - x1[:, None] * b * p2 - y0[:, None] * col2
            h = np.stack([np.broadcast_to(v_s, origin.shape), np.broadcast_to(col2, origin.shape), origin], axis=2)
            hs.append(h)
            templates.extend(combos)
    if not hs:
        return np.empty((0, 3, 3)), []
    return np.concatenate(hs), templates


def _batch_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """(N, 4, 2) court points and image points -> (N, 3, 3) homographies (h33 = 1); nan where singular."""
    n = len(src)
    a = np.zeros((n, 8, 8))
    b = np.zeros((n, 8))
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    a[:, 0::2, 0], a[:, 0::2, 1], a[:, 0::2, 2] = x, y, 1
    a[:, 0::2, 6], a[:, 0::2, 7] = -u * x, -u * y
    a[:, 1::2, 3], a[:, 1::2, 4], a[:, 1::2, 5] = x, y, 1
    a[:, 1::2, 6], a[:, 1::2, 7] = -v * x, -v * y
    b[:, 0::2], b[:, 1::2] = u, v
    ok = np.abs(np.linalg.det(a)) > 1e-12
    h = np.full((n, 9), np.nan)
    h[ok, :8] = np.linalg.solve(a[ok], b[ok][..., None])[..., 0]
    h[ok, 8] = 1
    return h.reshape(n, 3, 3)


def _cameras(hs: np.ndarray, width: int, height: int):
    """Focal length (px), orthonormality error and camera centre (court ft) per homography.

    Assumes square pixels and the principal point at the image centre, as
    `court.camera_from_homography` does. The error measures how far the two rotation
    columns are from orthogonal and equally long; the manual calibrations had 0.005-0.10.
    """
    shift = np.array([[1., 0., -width / 2], [0., 1., -height / 2], [0., 0., 1.]])
    h = shift @ hs
    a, b = h[:, :, 0], h[:, :, 1]
    rows = np.stack([np.stack([a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1], a[:, 2] * b[:, 2]], -1),
                     np.stack([a[:, 0] ** 2 + a[:, 1] ** 2 - b[:, 0] ** 2 - b[:, 1] ** 2,
                               a[:, 2] ** 2 - b[:, 2] ** 2], -1)], 1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv_f2 = -np.sum(rows[:, :, 0] * rows[:, :, 1], axis=1) / np.sum(rows[:, :, 0] ** 2, axis=1)
        focal = 1 / np.sqrt(inv_f2)
        k_inv = np.stack([1 / focal, 1 / focal, np.ones_like(focal)], -1)
        r1, r2, t = a * k_inv, b * k_inv, h[:, :, 2] * k_inv
        n1, n2 = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
        error = np.maximum(np.abs(np.sum(r1 * r2, axis=1)) / (n1 * n2), np.abs(np.log(n1 / n2)))
        scale = 2 / (n1 + n2)
        r1, r2, t = r1 * scale[:, None], r2 * scale[:, None], t * scale[:, None]
        rotation = np.stack([r1, r2, np.cross(r1, r2)], -1)
        centre = -np.einsum("nji,nj->ni", rotation, t)
    error = np.where(np.isfinite(error) & (inv_f2 > 0), error, np.inf)
    return focal, error, centre


def _plausible(hs: np.ndarray, width: int, height: int, max_error: float = .15) -> np.ndarray:
    """Views a real camera could take: consistent intrinsics, 3-150 ft up, within 200 ft of the court."""
    if not len(hs):
        return np.zeros(0, bool)
    focal, error, centre = _cameras(hs, width, height)
    with np.errstate(invalid="ignore"):
        return ((error <= max_error) & (focal >= .3 * width) & (focal <= 20 * width)
                & (np.abs(centre[:, 2]) >= 3) & (np.abs(centre[:, 2]) <= 150)
                & (np.hypot(centre[:, 0], centre[:, 1] - 47) <= 200))


# --- scoring and refinement --------------------------------------------------------------------------

def _line_samples(court: Template, step: float):
    """Points along every template line and their unit tangents, in court feet."""
    points, tangents = [], []
    for line in court.lines:
        for a, b in zip(line, line[1:]):
            length = math.dist(a, b)
            if length < 1e-9:
                continue
            n = max(1, int(round(length / step)))
            t = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
            for i in range(n):
                f = (i + .5) / n
                points.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
                tangents.append(t)
    return np.asarray(points), np.asarray(tangents)


def _line_ids(court: Template, step: float) -> np.ndarray:
    """Which template polyline each sample of `_line_samples(court, step)` belongs to."""
    ids = []
    for index, line in enumerate(court.lines):
        for a, b in zip(line, line[1:]):
            length = math.dist(a, b)
            if length >= 1e-9:
                ids.extend([index] * max(1, int(round(length / step))))
    return np.asarray(ids)


def _project(hs: np.ndarray, points: np.ndarray, tangents: np.ndarray):
    """Image points, unit image tangents, validity and px per foot for (N, 3, 3) homographies."""
    homog = np.c_[points, np.ones(len(points))]
    ahead = np.c_[points + .1 * tangents, np.ones(len(points))]
    p = np.einsum("nij,pj->npi", hs, homog)
    q = np.einsum("nij,pj->npi", hs, ahead)
    w, wq = p[..., 2], q[..., 2]
    front = (w > 1e-9) & (wq > 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = p[..., :2] / w[..., None]
        offset = q[..., :2] / wq[..., None] - xy
        length = np.linalg.norm(offset, axis=-1)
        tangent = offset / length[..., None]
    return xy, tangent, front & np.isfinite(length) & (length > 0), length * 10


def _support(hs, points, tangents, ev: Evidence, near: float, angle: float):
    """Per sample: inside the frame, and on a line pixel running the same way."""
    height, width = ev.dist.shape
    xy, tangent, ok, px_per_ft = _project(hs, points, tangents)
    inside = ok & (xy[..., 0] >= 0) & (xy[..., 1] >= 0) & (xy[..., 0] <= width - 1) & (xy[..., 1] <= height - 1)
    xi = np.where(inside, xy[..., 0], 0).astype(np.int32)
    yi = np.where(inside, xy[..., 1], 0).astype(np.int32)
    direction = ev.near_direction[yi, xi]
    agree = np.abs(np.nan_to_num(tangent[..., 0]) * direction[..., 0]
                   + np.nan_to_num(tangent[..., 1]) * direction[..., 1]) >= math.cos(math.radians(angle))
    return inside, inside & (ev.dist[yi, xi] <= near) & agree, xi, yi, px_per_ft


def _coarse_score(hs, points, tangents, ev: Evidence, step: float, near: float = 5., angle: float = 15.,
                  penalty: float = .4, chunk: int = 4000) -> np.ndarray:
    """Fast first pass: supported minus penalized unsupported template length (px, capped per sample)."""
    scores = np.zeros(len(hs))
    for start in range(0, len(hs), chunk):
        inside, supported, _, _, px_per_ft = _support(hs[start:start + chunk], points, tangents, ev, near, angle)
        length = np.minimum(np.nan_to_num(px_per_ft * step), 8.)
        scores[start:start + chunk] = np.sum(np.where(inside, np.where(supported, length, -penalty * length), 0.),
                                             axis=1)
    return scores


def _score(hs, points, tangents, ev: Evidence, near: float = 2.5, angle: float = 12., penalty: float = .4,
           cell: int = 3, chunk: int = 1000) -> np.ndarray:
    """Distinct supported image cells minus a share of the unsupported ones, per homography."""
    cols = ev.dist.shape[1] // cell + 1
    scores = np.zeros(len(hs))
    for start in range(0, len(hs), chunk):
        inside, supported, xi, yi, _ = _support(hs[start:start + chunk], points, tangents, ev, near, angle)
        cell_id = (yi // cell) * cols + xi // cell
        for flag, weight in ((supported, 1.), (inside & ~supported, -penalty)):
            ids = np.where(flag, cell_id, -1)
            ids.sort(axis=1)
            count = ((np.diff(ids, axis=1) != 0) & (ids[:, 1:] >= 0)).sum(axis=1) + (ids[:, 0] >= 0)
            scores[start:start + chunk] += weight * count
    return scores


def _refine(h: np.ndarray, points: np.ndarray, tangents: np.ndarray, ev: Evidence,
            schedule=(20., 12., 8., 5., 3., 2.), iterations: int = 6, angle: float = 15.) -> np.ndarray:
    """Point-to-line refinement: pull projected template lines onto line pixels.

    Each template sample within `tau` px of a line pixel running the same way contributes
    its offset along that pixel's normal. The eight homography parameters are updated by
    damped Gauss-Newton with Huber weights, in normalized units, as tau shrinks.
    """
    height, width = ev.dist.shape
    s_img, s_court = float(max(width, height)), 50.
    to_img, from_court = np.diag([1 / s_img, 1 / s_img, 1.]), np.diag([s_court, s_court, 1.])
    hn = to_img @ h @ from_court
    hn = hn / hn[2, 2]
    pn = np.c_[points / s_court, np.ones(len(points))]
    lam = 1e-3

    def fine(hn):
        full = np.linalg.inv(to_img) @ hn @ np.linalg.inv(from_court)
        return float(_score(full[None], points, tangents, ev, near=2.)[0])

    # The least-squares cost is not the fine score (a wrong nearest pixel can pull a line
    # off), so the best-scoring homography seen after each stage is kept.
    best_score, best = fine(hn), hn
    for tau in schedule:
        for _ in range(iterations):
            full = np.linalg.inv(to_img) @ hn @ np.linalg.inv(from_court)
            _, keep, xi, yi, _ = _support(full[None], points, tangents, ev, tau, angle)
            keep, xi, yi = keep[0], xi[0], yi[0]
            if keep.sum() < 20:
                break
            p = pn[keep]
            index = ev.nearest[yi[keep], xi[keep]]
            target = ev.pixels[index] / s_img
            normal = np.c_[-ev.direction[index, 1], ev.direction[index, 0]]

            def residuals(hv):
                q = p @ np.append(hv, 1.).reshape(3, 3).T
                uv = q[:, :2] / q[:, 2:3]
                return np.sum(normal * (uv - target), axis=1), q, uv

            hv = hn.ravel()[:8]
            r, q, uv = residuals(hv)
            w = q[:, 2:3]
            jac = np.hstack([normal[:, :1] * p / w, normal[:, 1:] * p / w,
                             -np.sum(normal * uv, axis=1, keepdims=True) * p / w])[:, :8]
            limit = 1. / s_img  # Huber threshold: one working pixel
            weights = np.where(np.abs(r) <= limit, 1., limit / np.maximum(np.abs(r), 1e-12))
            a, g = jac.T @ (jac * weights[:, None]), (jac * weights[:, None]).T @ r
            cost = float(np.sum(weights * r * r))
            for _attempt in range(6):
                try:
                    delta = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-12), -g)
                except np.linalg.LinAlgError:
                    break
                new_r, _, _ = residuals(hv + delta)
                new_cost = float(np.sum(weights * new_r * new_r))
                if np.isfinite(new_cost) and new_cost < cost:
                    hn = np.append(hv + delta, 1.).reshape(3, 3)
                    lam = max(lam / 3, 1e-7)
                    break
                lam *= 10
            else:
                break
        score = fine(hn)
        if score > best_score:
            best_score, best = score, hn
    out = np.linalg.inv(to_img) @ best @ np.linalg.inv(from_court)
    return out * np.sign(out[2, 2] if abs(out[2, 2]) > 1e-12 else 1.)


def _search(batch: _Batch, court: Template, ev: Evidence, width: int, height: int,
            keep_coarse: int = 300, keep_fine: int = 30) -> tuple[np.ndarray | None, dict]:
    hs = batch.arrays()
    info = {"hypotheses": batch.total, "plausible": int(len(hs))}
    if not len(hs):
        return None, info
    points, tangents = _line_samples(court, 1.)
    order = np.argsort(-_coarse_score(hs, points, tangents, ev, 1.))[:keep_coarse]
    mid_points, mid_tangents = _line_samples(court, .5)
    top = order[np.argsort(-_score(hs[order], mid_points, mid_tangents, ev, near=4.))[:keep_fine]]
    fine_points, fine_tangents = _line_samples(court, .25)
    best = None
    for index in top:
        h = _refine(hs[index], fine_points, fine_tangents, ev)
        if not _plausible(h[None], width, height)[0]:
            continue
        score = float(_score(h[None], fine_points, fine_tangents, ev, near=2.)[0])
        if best is None or score > best[0]:
            best = (score, h)
    if best is None:
        return None, info
    info["score"] = round(best[0], 1)
    return best[1], info


def _coverage(h: np.ndarray, court: Template, ev: Evidence, step: float = .25) -> tuple[float, int, int]:
    """Share of in-view template samples on line pixels, and how many template lines are found / in view."""
    points, tangents = _line_samples(court, step)
    ids = _line_ids(court, step)
    inside, supported, _, _, _ = _support(h[None], points, tangents, ev, 2., 12.)
    inside, supported = inside[0], supported[0]
    if not inside.any():
        return 0., 0, 0
    found = in_view = 0
    for line in np.unique(ids[inside]):
        members = inside & (ids == line)
        if members.sum() < 8:
            continue
        in_view += 1
        found += supported[members].mean() >= .3
    return float(supported.sum() / inside.sum()), int(found), int(in_view)


def propose(frame: np.ndarray, standard: str = "nba", boxes=()) -> Proposal:
    """Propose court landmarks for one BGR frame (any size).

    Returns the template landmarks that fall inside the frame, as normalized image points
    under the best-fitting court, or ok=False with a reason.
    """
    started = time.perf_counter()
    court = template(standard)
    height, width = frame.shape[:2]
    scale = min(1., WORK_WIDTH / width)
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else frame
    sh, sw = small.shape[:2]
    ev = Evidence.build(small, boxes)
    lines = candidate_lines(small, ev.mask)
    families = _families(lines, sw, sh)
    diagnostics: dict = {"lines": len(lines), "direction_sets": len(families)}

    def fail(reason: str, confidence: float = 0., fitted: np.ndarray | None = None) -> Proposal:
        # A rejected fit is kept for evaluation scripts; the page does not draw it.
        diagnostics["seconds"] = round(time.perf_counter() - started, 2)
        return Proposal(False, reason, confidence, standard, court_to_image=fitted, diagnostics=diagnostics)

    if not families:
        return fail("Not enough straight painted lines were found in two floor directions. "
                    "Pick a frame where the lane and a sideline or baseline are clear.")
    results = []
    for label, maker in (("two_by_two", _hypotheses_two_two), ("two_plus_one", _hypotheses_two_one)):
        batch = _Batch(sw, sh)
        for family in families:
            maker(family, court, sw, sh, batch)
        h, info = _search(batch, court, ev, sw, sh)
        diagnostics[label] = info
        if h is not None:
            results.append((info["score"], label, h))
        # Two lines each way is the common broadcast case; skip the slower sweep when it fits well.
        if h is not None and label == "two_by_two" and _coverage(h, court, ev)[0] >= MIN_SUPPORT:
            break
    if not results:
        return fail("No camera view of the court template fits the painted lines found in this frame.")
    _, label, h = max(results, key=lambda item: item[0])
    support, found, in_view = _coverage(h, court, ev)
    diagnostics.update(method=label, support=round(support, 3), lines_found=found, lines_in_view=in_view)
    court_to_image = np.diag([1 / scale, 1 / scale, 1.]) @ h
    court_to_image /= np.linalg.norm(court_to_image)
    if support < MIN_SUPPORT or found < MIN_LINES:
        return fail(f"The best court fit only lands on {support:.0%} of the lines it expects in view "
                    f"({found} of {in_view} lines found), too little to propose. Mark the landmarks by hand, "
                    "or try a frame with more of the floor lines visible.", support, court_to_image)
    image = apply(court_to_image, list(court.landmarks.values()))
    points = []
    for key, (x, y) in zip(court.landmarks, image):
        nx, ny = x / width, y / height
        if np.isfinite(nx) and np.isfinite(ny) and EDGE_MARGIN <= nx <= 1 - EDGE_MARGIN and EDGE_MARGIN <= ny <= 1 - EDGE_MARGIN:
            points.append((key, (float(nx), float(ny))))
    if len(points) < 4 or not general_position([court.landmarks[key] for key, _ in points]):
        return fail("The court fit leaves fewer than 4 usable landmarks inside the frame. "
                    "Mark the landmarks by hand.", support, court_to_image)
    diagnostics["seconds"] = round(time.perf_counter() - started, 2)
    return Proposal(True, None, support, standard, points, court_to_image, diagnostics)
