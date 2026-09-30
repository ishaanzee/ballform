"""Per-player floor trajectories in court feet, smoothed for speeds.

A single frame's floor point (app/court.py) is good for a spot, but frame to
frame it jitters: ankle keypoints wobble by a few pixels, a lifted foot maps to
a point beyond the player, and a pixel is several feet far from the camera.
Here every tracked player's floor points are filtered with a constant-velocity
Kalman filter and a Rauch-Tung-Striebel (forward-backward) smoother:

- Measurement noise is set in pixels, from how the point was found (both
  ankles < one ankle < pose box, as a fraction of the player's height in the
  image), and mapped to feet through the local Jacobian of the image-to-court
  homography, so a pixel of jitter counts for more feet far from the camera.
- Frames with a raised foot are skipped. With camera motion removed, the
  lower foot's image height is closed morphologically over a short window; a
  foot well above that floor level is off the ground, or its keypoint is
  misplaced. A jump's horizontal motion is ballistic, so the constant-velocity
  model carries the player through it. One camera cannot tell a lifted foot
  from a quick step away from it and back, so some such steps are skipped too.
- Outliers are down-weighted, not trusted: a measurement outside a
  Mahalanobis gate moves the forward filter by at most a gate-sized step, and
  the smoother is then rerun with weights from the smoothed residuals (full
  weight inside the gate, gate / d^2 beyond it, so a far outlier counts for
  almost nothing), so a sharp but real change of direction keeps its weight.
  When several outliers in a row agree on a place well away from where the
  player was heading, the trajectory is split there instead: the track most
  likely switched to someone else.
- Trajectories break at camera cuts, on any frame without a reliable court
  mapping, and after a gap longer than MAX_GAP_S; they are never
  interpolated across a break. Short gaps inside a trajectory (including
  stitched track fragments, app/tracking.py) are bridged by the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.court import floor_point, lowest_foot_y

# Pixel noise of one floor point, as a fraction of the player's image height
# (u across, v down), with a floor of MIN_PIXEL_SIGMA. The ankle values and
# ACCEL_PSD maximise the filter's likelihood on three calibrated broadcast
# clips (scripts/measure_floor_trajectories.py --tune); they come to about
# 1.5-2 px. One ankle stands in for the midpoint of both, so it is off by up to
# half a stance across; the pose box bottom (after BOX_FOOT_OFFSET) spreads by
# about 0.03 of the height. Too few such points exist to fit those two.
PIXEL_SIGMA = {"ankles": (.006, .01), "one ankle": (.04, .02), "pose box": (.03, .04)}
MIN_PIXEL_SIGMA = 1.5
# The pose box bottom sits below the ankles: on three broadcast clips, by a
# median 0.12-0.14 of the box height (5th-95th percentile 0.05-0.19).
BOX_FOOT_OFFSET = .13
EDGE_MARGIN = .015         # fraction of the frame; feet or a pose box this close to an edge are not used
ACCEL_PSD = 150.           # white-noise acceleration spectral density, ft^2/s^3
INITIAL_SPEED_SD = 10.     # ft/s, prior on the velocity at the start of a trajectory
GATE = 13.8                # chi-square, 2 dof, 99.9%: beyond it a measurement is down-weighted
IRLS_PASSES = 2            # smoother passes reweighted by the smoothed residuals
WEIGHT_USED = .5           # a measurement with a lower final weight is reported as an outlier
RESET_AFTER = 4            # consecutive outliers that agree on a new place (within RESET_SPREAD_FT)
RESET_SPREAD_FT = 3.       # at least RESET_JUMP_FT from where the player was heading start a new
RESET_JUMP_FT = 3.         # trajectory: the track most likely switched to someone else
MAX_GAP_S = .5             # longest gap without a measurement that a trajectory bridges
AIR_WINDOW_S = .5          # closing window for the floor level of the lower foot
AIR_RISE_TORSO = .25       # lower foot this far above the floor level is off the floor (or misplaced)
AIR_EDGE_TORSO = .1        # ... and so are neighbouring frames still this far above it
MIN_MEASURED = 3           # accepted measurements for a trajectory to be kept


@dataclass
class Measurement:
    frame: int
    time_s: float
    pixel: tuple[float, float]
    method: str                         # ankles | one ankle | pose box
    court: np.ndarray                   # feet
    cov: np.ndarray                     # 2x2, feet^2
    foot_y: float | None = None         # lower foot, camera motion removed (anchor-image pixels)
    torso_px: float | None = None       # torso length in the same pixels
    rise_torso: float | None = None     # lower foot above the local floor level, torso lengths
    airborne: bool = False


@dataclass
class Trajectory:
    """One unbroken, smoothed floor trajectory of one track."""
    track_id: int
    frames: list[int]
    times: np.ndarray                   # (N,)
    position: np.ndarray                # (N, 2) smoothed, feet
    velocity: np.ndarray                # (N, 2) smoothed, ft/s
    position_cov: np.ndarray            # (N, 2, 2) smoothed, feet^2
    source: list[str]                   # per frame, see SOURCE_CODES
    raw: list[np.ndarray | None] = field(default_factory=list)  # measured floor point, if any

    def index(self, frame: int) -> int | None:
        try:
            return self.frames.index(frame)
        except ValueError:
            return None

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocity, axis=1)

    @property
    def position_sd(self) -> np.ndarray:
        """Per frame, the standard deviation along the least certain direction, feet."""
        return np.sqrt(np.linalg.eigvalsh(self.position_cov)[:, -1])


# Per-frame source codes in trajectories and court.json.
SOURCE_CODES = {"a": "both ankles", "o": "one ankle", "b": "pose box",
                "j": "foot raised (airborne, or a misplaced ankle), skipped",
                "g": "outlier, down-weighted", "-": "no measurement, bridged"}
_METHOD_CODE = {"ankles": "a", "one ankle": "o", "pose box": "b"}


def homography_jacobian(h: np.ndarray, pixel) -> np.ndarray | None:
    """d(court feet)/d(pixel) of an image-to-court homography at one pixel."""
    u, v = pixel
    x, y, w = h @ np.array([u, v, 1.])
    if not abs(w) > 1e-12:
        return None
    x, y = x / w, y / w
    return np.array([[h[0, 0] - x * h[2, 0], h[0, 1] - x * h[2, 1]],
                     [h[1, 0] - y * h[2, 0], h[1, 1] - y * h[2, 1]]]) / w


def _player_height_px(pose, width: int, height: int) -> float | None:
    box = getattr(pose, "box", None)
    if box is not None and all(math.isfinite(v) for v in box) and box[3] > box[1]:
        return (box[3] - box[1]) * height
    ys = [p[1] for p in pose.landmarks.values() if p and len(p) >= 3 and p[2] >= .4 and math.isfinite(p[1])]
    return (max(ys) - min(ys)) * height * 1.15 if len(ys) >= 4 and max(ys) > min(ys) else None


def _torso_px(pose, width: int, height: int) -> float | None:
    points = [pose.landmarks.get(name) for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if any(p is None or len(p) < 3 or p[2] < .5 for p in points):
        return None
    shoulder = ((points[0][0] + points[1][0]) / 2 * width, (points[0][1] + points[1][1]) / 2 * height)
    hip = ((points[2][0] + points[3][0]) / 2 * width, (points[2][1] + points[3][1]) / 2 * height)
    return math.dist(shoulder, hip)


def measure(pose, frame: int, time_s: float, court_map, sigma: dict | None = None) -> Measurement | None:
    """One floor measurement of a pose on a reliably mapped frame, with its covariance in feet."""
    width, height = court_map.width, court_map.height
    mapping = court_map.reliable(frame)
    found = floor_point(pose, width, height) if mapping is not None else None
    if found is None:
        return None
    (u, v), method = found
    box = getattr(pose, "box", None)
    # Feet at the frame's bottom edge, or a pose box cut by any edge, are probably cut off.
    if v > (1 - EDGE_MARGIN) * height or (method == "pose box" and box is not None and (
            box[3] > 1 - EDGE_MARGIN or box[0] < EDGE_MARGIN or box[2] > 1 - EDGE_MARGIN)):
        return None
    size = _player_height_px(pose, width, height)
    if size is None:
        torso = _torso_px(pose, width, height)
        size = 3 * torso if torso else None
    if size is None:
        return None
    if method == "pose box":
        v -= BOX_FOOT_OFFSET * size
    court = court_map.to_court(frame, (u, v))
    jacobian = homography_jacobian(mapping.image_to_court, (u, v))
    if court is None or jacobian is None:
        return None
    su, sv = ((sigma or PIXEL_SIGMA)[method][0] * size, (sigma or PIXEL_SIGMA)[method][1] * size)
    pixel_cov = np.diag([max(su, MIN_PIXEL_SIGMA) ** 2, max(sv, MIN_PIXEL_SIGMA) ** 2])
    measurement = Measurement(frame, time_s, (u, v), method, np.asarray(court, float),
                              jacobian @ pixel_cov @ jacobian.T)
    low = lowest_foot_y(pose, height)
    torso = _torso_px(pose, width, height) or size / 3
    if low is not None:
        foot = court_map.to_anchor_image(frame, (u, low))
        above = court_map.to_anchor_image(frame, (u, low - torso))
        if foot is not None and above is not None:
            measurement.foot_y = foot[1]
            measurement.torso_px = math.dist(foot, above)
    return measurement


def mark_airborne(measurements: list[Measurement], window_s: float = AIR_WINDOW_S) -> None:
    """Flag samples whose lower foot is clearly above the floor level around them.

    The floor level is a morphological closing of the lower foot's image y
    (camera motion removed): a running maximum then a running minimum over
    window_s. It follows the foot while the player walks or runs toward or away
    from the camera (a monotone change passes through unchanged) and fills in
    the short rise of a jump. It also fills in a quick step away from the
    camera and back, which a single view cannot tell from a lift.
    """
    usable = [m for m in measurements if m.foot_y is not None and m.torso_px]
    if len(usable) < 3:
        return
    times = np.array([m.time_s for m in usable])
    ys = np.array([m.foot_y for m in usable])
    half = window_s / 2
    # Repeat the end values beyond both ends, so a player still moving at the
    # start or end of the trajectory does not read as a raised foot there.
    spacing = max(float(np.median(np.diff(times))), 1e-3)
    pad = np.arange(1, math.ceil(window_s / spacing) + 1) * spacing
    padded_t = np.r_[times[0] - pad[::-1], times, times[-1] + pad]
    padded_y = np.r_[np.full(len(pad), ys[0]), ys, np.full(len(pad), ys[-1])]
    dilated = np.array([padded_y[np.abs(padded_t - t) <= half].max() for t in padded_t])
    closed = np.array([dilated[np.abs(padded_t - t) <= half].min() for t in times])
    rise = (closed - ys) / np.array([m.torso_px for m in usable])
    airborne = rise > AIR_RISE_TORSO
    # Hysteresis: a jump starts and ends with the foot only slightly raised.
    for order in (range(1, len(usable)), range(len(usable) - 2, -1, -1)):
        for i in order:
            neighbour = i - 1 if order.step == 1 else i + 1
            if airborne[neighbour] and rise[i] > AIR_EDGE_TORSO and abs(times[i] - times[neighbour]) <= .1:
                airborne[i] = True
    for m, value, flag in zip(usable, rise, airborne):
        m.rise_torso, m.airborne = float(value), bool(flag)


def _transition(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    q2 = np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]]) * q
    big = np.zeros((4, 4))
    big[np.ix_([0, 2], [0, 2])] = q2
    big[np.ix_([1, 3], [1, 3])] = q2
    return f, big


_H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])


@dataclass
class _Pass:
    means: np.ndarray               # filtered
    covs: np.ndarray
    predicted_means: np.ndarray
    predicted_covs: np.ndarray
    split: int | None               # where a run of agreeing, distant outliers starts a new trajectory
    log_likelihood: float


def _huber(d2: float, gate: float) -> float:
    return d2 if d2 <= gate else 2 * math.sqrt(gate * d2) - gate


def _filter(times: np.ndarray, z: list, r: list, q: float, gate: float,
            weights: np.ndarray | None = None, split: bool = True) -> _Pass:
    """Forward Kalman pass from z[0], which must be a measurement.

    Without weights, a measurement outside the gate is down-weighted so that it
    moves the state by at most a gate-sized step (a Huber update). With
    weights, measurement i counts as if its covariance were r[i] / weights[i].
    With split, a run of RESET_AFTER outliers that agree on a place at least
    RESET_JUMP_FT from where the player was heading stops the pass there.
    """
    n = len(times)
    means, covs = np.zeros((n, 4)), np.zeros((n, 4, 4))
    predicted_means, predicted_covs = np.zeros((n, 4)), np.zeros((n, 4, 4))
    x = np.r_[z[0], 0., 0.]
    p = np.zeros((4, 4))
    # The first measurement is the starting position (with its weight, when reweighting).
    p[:2, :2] = r[0] / (weights[0] if weights is not None and weights[0] > 0 else 1.)
    p[2, 2] = p[3, 3] = INITIAL_SPEED_SD ** 2
    rejected: list[int] = []
    log_likelihood = 0.
    for i in range(n):
        if i > 0:
            f, qm = _transition(times[i] - times[i - 1], q)
            x, p = f @ x, f @ p @ f.T + qm
        predicted_means[i], predicted_covs[i] = x, p
        if i > 0 and z[i] is not None and (weights is None or weights[i] > 0):
            innovation = z[i] - x[:2]
            s = p[:2, :2] + r[i]
            d2 = float(innovation @ np.linalg.solve(s, innovation))
            log_likelihood -= .5 * (_huber(d2, gate) + math.log(np.linalg.det(s)) + 2 * math.log(2 * math.pi))
            if weights is not None:
                s = p[:2, :2] + r[i] / weights[i]
            elif d2 > gate:
                s = s * math.sqrt(d2 / gate)
            gain = p @ _H.T @ np.linalg.inv(s)
            x = x + gain @ innovation
            p = (np.eye(4) - gain @ _H) @ p
            p = (p + p.T) / 2
            if split and d2 > gate:
                rejected.append(i)
                if len(rejected) >= RESET_AFTER:
                    k = rejected[0]
                    before = means[k - 1]
                    points = np.asarray([z[j] for j in rejected])
                    heading = np.asarray([before[:2] + before[2:] * (times[j] - times[k - 1]) for j in rejected])
                    spread = np.linalg.norm(points - np.median(points, axis=0), axis=1).max()
                    jump = float(np.median(np.linalg.norm(points - heading, axis=1)))
                    if spread <= RESET_SPREAD_FT and jump >= RESET_JUMP_FT:
                        return _Pass(means[:k], covs[:k], predicted_means[:k], predicted_covs[:k], k,
                                     log_likelihood)
                    rejected = rejected[1:]
            else:
                rejected = []
        means[i], covs[i] = x, p
    return _Pass(means, covs, predicted_means, predicted_covs, None, log_likelihood)


def _rts(times: np.ndarray, forward: _Pass, q: float) -> tuple[np.ndarray, np.ndarray]:
    means, covs = forward.means.copy(), forward.covs.copy()
    for i in range(len(means) - 2, -1, -1):
        f, _ = _transition(times[i + 1] - times[i], q)
        gain = forward.covs[i] @ f.T @ np.linalg.inv(forward.predicted_covs[i + 1])
        means[i] = forward.means[i] + gain @ (means[i + 1] - forward.predicted_means[i + 1])
        covs[i] = forward.covs[i] + gain @ (covs[i + 1] - forward.predicted_covs[i + 1]) @ gain.T
    return means, covs


def _robust_smooth(times, z, r, q, gate):
    """Huber forward pass (which may split), then RTS smoothing reweighted by the smoothed residuals.

    Weighting from the smoothed rather than the predicted position keeps a
    sharp but real change of direction: the smoothed path passes close to
    those points, so they keep their full weight, while an isolated bad point
    stays far from it.
    """
    forward = _filter(times, z, r, q, gate)
    n = len(forward.means)
    means, covs = _rts(times[:n], forward, q)
    weights = np.zeros(n)
    for _ in range(IRLS_PASSES):
        for i in range(n):
            if z[i] is not None:
                e = z[i] - means[i, :2]
                d2 = float(e @ np.linalg.solve(r[i], e))
                weights[i] = min(1., gate / d2) if d2 > 0 else 1.
        reweighted = _filter(times[:n], z[:n], r[:n], q, gate, weights=weights, split=False)
        means, covs = _rts(times[:n], reweighted, q)
    return forward, means, covs, weights


def smooth_span(times, z, r, q: float = ACCEL_PSD, gate: float = GATE):
    """Smooth one unbroken span; returns [(start, stop, means, covs, weights, log_likelihood)] pieces.

    z[i] is None where there is no usable measurement; weights are the final
    measurement weights (1 = fully trusted). A new piece starts where a run of
    agreeing, distant outliers split the span; leading frames without a
    measurement are skipped.
    """
    times = np.asarray(times, float)
    pieces, start = [], 0
    while start < len(times):
        while start < len(times) and z[start] is None:
            start += 1
        if start >= len(times):
            break
        forward, means, covs, weights = _robust_smooth(times[start:], z[start:], r[start:], q, gate)
        stop = start + len(means)
        pieces.append((start, stop, means, covs, weights, forward.log_likelihood))
        if forward.split is None:
            break
        start += forward.split
    return pieces


def _spans(player_frames: list[dict], court_map, cuts: list[int], max_gap_s: float,
           sigma: dict | None) -> list[tuple[int, list[tuple[dict, Measurement | None]]]]:
    """Split each track into spans with no cut, no unreliable mapping and no long gap."""
    cut_set = set(cuts or [])
    open_spans: dict[int, list] = {}
    last_seen: dict[int, float] = {}
    done = []

    def close(track_id):
        span = open_spans.pop(track_id, None)
        if span:
            # Drop trailing frames without a measurement.
            while span and span[-1][1] is None:
                span.pop()
            if span:
                done.append((track_id, span))
        last_seen.pop(track_id, None)

    frames = sorted(player_frames, key=lambda f: f["frame"])
    previous_frame = None
    for frame in frames:
        number = frame["frame"]
        crossed_cut = previous_frame is not None and any(previous_frame < c <= number for c in cut_set)
        previous_frame = number
        if crossed_cut or court_map.reliable(number) is None:
            for track_id in list(open_spans):
                close(track_id)
            if court_map.reliable(number) is None:
                continue
        seen = {}
        for pose in frame["players"]:
            if pose.track_id is None or pose.track_id in seen:
                continue
            seen[pose.track_id] = measure(pose, number, frame["time_s"], court_map, sigma)
        for track_id in list(open_spans):
            if seen.get(track_id) is None and frame["time_s"] - last_seen[track_id] > max_gap_s:
                close(track_id)
        for track_id, found in seen.items():
            if found is not None:
                if track_id in open_spans and frame["time_s"] - last_seen[track_id] > max_gap_s:
                    close(track_id)
                open_spans.setdefault(track_id, [])
                last_seen[track_id] = frame["time_s"]
        for track_id, span in open_spans.items():
            span.append((frame, seen.get(track_id)))
    for track_id in list(open_spans):
        close(track_id)
    return done


def floor_trajectories(player_frames: list[dict], court_map, cuts: list[int] | None = None,
                       q: float = ACCEL_PSD, gate: float = GATE, max_gap_s: float = MAX_GAP_S,
                       sigma: dict | None = None, stats: dict | None = None,
                       exclude: set[tuple[int, int]] | None = None) -> list[Trajectory]:
    """Smoothed floor trajectories of every tracked player over reliably mapped frames.

    Track IDs are used as they are after stitching (app/tracking.py); a stitched
    join is a gap of at most 0.5 s and is bridged like any other short gap.
    `exclude` holds (track_id, frame) measurements to leave out, for validation.
    """
    trajectories = []
    counts = {"log_likelihood": 0., "measurements": 0, "accepted": 0, "gated": 0, "airborne": 0,
              "splits": 0, "spans": 0, "dropped_short": 0}
    for track_id, span in _spans(player_frames, court_map, cuts or [], max_gap_s, sigma):
        counts["spans"] += 1
        measurements = [m for _, m in span if m is not None]
        mark_airborne(measurements)
        counts["airborne"] += sum(m.airborne for m in measurements)
        times = np.array([frame["time_s"] for frame, _ in span])
        usable = [m is not None and not m.airborne and (track_id, m.frame) not in (exclude or ())
                  for _, m in span]
        z = [m.court if ok else None for (_, m), ok in zip(span, usable)]
        r = [m.cov if ok else None for (_, m), ok in zip(span, usable)]
        counts["measurements"] += sum(usable)
        pieces = smooth_span(times, z, r, q, gate)
        counts["splits"] += max(0, len(pieces) - 1)
        for start, stop, means, covs, weights, piece_ll in pieces:
            counts["log_likelihood"] += piece_ll
            accepted = weights >= WEIGHT_USED
            local = [i for i in range(stop - start) if accepted[i]]
            counts["accepted"] += len(local)
            counts["gated"] += sum(z[start + i] is not None for i in range(stop - start)) - len(local)
            if len(local) < MIN_MEASURED:
                counts["dropped_short"] += 1
                continue
            keep = list(range(local[0], local[-1] + 1))
            source, raw = [], []
            for i in keep:
                m = span[start + i][1]
                raw.append(None if m is None else m.court)
                if m is None or not usable[start + i] and not m.airborne:
                    source.append("-")
                elif m.airborne:
                    source.append("j")
                elif not accepted[i]:
                    source.append("g")
                else:
                    source.append(_METHOD_CODE[m.method])
            trajectories.append(Trajectory(
                track_id, [span[start + i][0]["frame"] for i in keep], times[start:stop][keep],
                means[keep, :2], means[keep, 2:], covs[keep][:, :2, :2], source, raw))
    if stats is not None:
        stats.update(counts)
    return trajectories


def trajectories_json(trajectories: list[Trajectory]) -> dict:
    """Compact form for court.json: per trajectory, frames and smoothed x, y (ft), vx, vy (ft/s)."""
    return {
        "method": ("Constant-velocity Kalman filter with a robust RTS smoother per track; measurement noise from "
                   "how the floor point was found, mapped to feet through the homography's Jacobian; frames with a "
                   "raised foot skipped and outliers down-weighted; broken at camera cuts, unreliable court mapping "
                   f"and gaps over {MAX_GAP_S:g} s. x, y in court feet; vx, vy in ft/s; sd is the position standard "
                   "deviation in feet along the least certain direction."),
        "source_codes": SOURCE_CODES,
        "tracks": [{"track_id": t.track_id, "frames": t.frames,
                    "x": [round(float(v), 1) for v in t.position[:, 0]],
                    "y": [round(float(v), 1) for v in t.position[:, 1]],
                    "vx": [round(float(v), 1) for v in t.velocity[:, 0]],
                    "vy": [round(float(v), 1) for v in t.velocity[:, 1]],
                    "sd": [round(float(v), 2) for v in t.position_sd],
                    "source": "".join(t.source)} for t in trajectories],
    }
