from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence

import numpy as np

from app.geometry import distance, elevation_angle, interpolate_crossing, joint_angle
from app.models import Detection, PoseFrame, ShotResult

RimBox = tuple[float, float, float, float]
RimInput = RimBox | Mapping[int, RimBox] | None


def _rim_at(rim: RimInput, frame: int) -> RimBox | None:
    """Return a rim box at a source frame, interpolating tracked samples."""
    if rim is None:
        return None
    if not isinstance(rim, Mapping):
        return rim
    if frame in rim:
        return rim[frame]
    keys = sorted(rim)
    index = bisect_left(keys, frame)
    if index == 0 or index == len(keys):
        return None
    left, right = keys[index - 1], keys[index]
    # Do not bridge a tracker failure or camera cut.
    if right - left > 12:
        return None
    amount = (frame - left) / (right - left)
    return tuple(rim[left][i] + amount * (rim[right][i] - rim[left][i]) for i in range(4))  # type: ignore[return-value]

def _point(pose: PoseFrame, name: str, aspect_ratio: float = 1.0) -> tuple[float, float] | None:
    value = pose.landmarks.get(name)
    if value is None or value[2] < 0.35:
        return None
    return value[0] * aspect_ratio, value[1]


def _nearest_pose(poses: Sequence[PoseFrame], frame: int, fps: float) -> PoseFrame | None:
    pose = min(poses, key=lambda p: abs(p.frame - frame), default=None)
    return pose if pose and abs(pose.frame - frame) / fps <= .15 else None


def classify_view(poses: Sequence[PoseFrame], aspect_ratio: float = 1.0) -> tuple[str, float]:
    ratios = []
    for pose in poses:
        ls, rs = _point(pose, "left_shoulder", aspect_ratio), _point(pose, "right_shoulder", aspect_ratio)
        lh, rh = _point(pose, "left_hip", aspect_ratio), _point(pose, "right_hip", aspect_ratio)
        if all((ls, rs, lh, rh)):
            shoulder_width = distance(ls, rs)  # type: ignore[arg-type]
            torso = distance(((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2), ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2))  # type: ignore[index]
            if torso > 1e-5:
                ratios.append(shoulder_width / torso)
    if not ratios:
        return "unknown", 0.0
    ratio = float(np.median(ratios))
    if ratio > 0.72:
        return "front/rear", min(1.0, (ratio - 0.55) / 0.45)
    if ratio < 0.50:
        return "side", min(1.0, (0.67 - ratio) / 0.35)
    return "oblique", 0.65


def _release_proximity(pose, detection, side, aspect_ratio, game_mode):
    wrist = _point(pose, f"{side}_wrist", aspect_ratio)
    if wrist is None or detection is None:
        return None
    proximity = distance(wrist, (detection.x * aspect_ratio, detection.y))
    if not game_mode:
        return proximity
    body = [_point(pose, name, aspect_ratio) for name in
            ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if not all(body):
        return None
    shoulder = tuple((body[0][i]+body[1][i])/2 for i in (0, 1))
    hip = tuple((body[2][i]+body[3][i])/2 for i in (0, 1))
    torso = distance(shoulder, hip)
    # Waist-level contact followed by a bounce is not a shot release. Scaling
    # to the player's body avoids a large fixed-radius contact test in wide views.
    if (torso <= 0 or proximity > .9 * torso or detection.confidence < .4
            or wrist[1] > shoulder[1] + .15*torso):
        return None
    return proximity / torso * .10


def find_release(ball: Sequence[Detection], poses: Sequence[PoseFrame], start: int, apex: int, aspect_ratio: float = 1.0, handedness: str = "right", game_mode: bool = False) -> int:
    candidates: list[tuple[float, int]] = []
    by_frame = {b.frame: b for b in ball}
    for pose in poses:
        if not start <= pose.frame <= apex:
            continue
        detection = by_frame.get(pose.frame)
        for side in (("left", "right") if game_mode else (handedness,)):
            proximity = _release_proximity(pose, detection, side, aspect_ratio, game_mode)
            if proximity is not None:
                candidates.append((proximity, pose.frame))
    if candidates and min(c[0] for c in candidates) <= .12:
        near_frame = min(candidates)[1]
        # The release follows the last close hand/ball contact before separation.
        threshold = min(.12, max(0.035, min(c[0] for c in candidates) * 1.8))
        close = [f for d, f in candidates if d <= threshold and f >= near_frame]
        return max(close, default=near_frame)
    return start


def _trajectory_release_angle(ball: Sequence[Detection], release_frame: int, fps: float, aspect_ratio: float = 1.0) -> float | None:
    pts = [b for b in ball if release_frame <= b.frame <= release_frame + max(3, round(fps * 0.30))]
    if len(pts) < 3:
        return None
    x = np.asarray([p.x * aspect_ratio for p in pts])
    y = np.asarray([p.y for p in pts])
    t = np.asarray([(p.frame - release_frame) / fps for p in pts])
    vx = float(np.polyfit(t, x, 1)[0])
    vy = float(np.polyfit(t, y, 1)[0])
    return math.degrees(math.atan2(-vy, abs(vx)))


def _form_metrics(poses: Sequence[PoseFrame], ball: Sequence[Detection], release: int, fps: float, aspect_ratio: float = 1.0, handedness: str = "right") -> tuple[dict, list[str]]:
    pose = _nearest_pose(poses, release, fps)
    metrics: dict[str, float | str | None] = {
        "elbow_angle_at_release_deg": None,
        "set_point_elbow_angle_deg": None,
        "upper_arm_elevation_deg": None,
        "wrist_over_elbow_pct_shoulder_width": None,
        "release_height_body_ratio": None,
        "follow_through_extension_deg": None,
        "release_angle_2d_deg": _trajectory_release_angle(ball, release, fps, aspect_ratio),
    }
    cues: list[str] = []
    if pose:
        shoulder, elbow, wrist = (_point(pose, f"{handedness}_{k}", aspect_ratio) for k in ("shoulder", "elbow", "wrist"))
        if shoulder and elbow and wrist:
            metrics["elbow_angle_at_release_deg"] = _round(joint_angle(shoulder, elbow, wrist))
            metrics["upper_arm_elevation_deg"] = _round(elevation_angle(shoulder, elbow))
            ls, rs = _point(pose, "left_shoulder", aspect_ratio), _point(pose, "right_shoulder", aspect_ratio)
            width = distance(ls, rs) if ls and rs else 0
            if width > 1e-5:
                metrics["wrist_over_elbow_pct_shoulder_width"] = round(100 * (wrist[0] - elbow[0]) / width, 1)
            nose = _point(pose, "nose")
            ankles = [_point(pose, "left_ankle"), _point(pose, "right_ankle")]
            visible_ankles = [p for p in ankles if p]
            if nose and visible_ankles:
                foot_y = max(p[1] for p in visible_ankles)
                height = foot_y - nose[1]
                if height > 1e-5:
                    metrics["release_height_body_ratio"] = round((foot_y - wrist[1]) / height, 2)

    pre_angles, post_angles = [], []
    for p in poses:
        s, e, w = (_point(p, f"{handedness}_{k}", aspect_ratio) for k in ("shoulder", "elbow", "wrist"))
        if s and e and w:
            angle = joint_angle(s, e, w)
            if angle is not None and release - fps <= p.frame <= release:
                pre_angles.append(angle)
            if angle is not None and release <= p.frame <= release + 0.5 * fps:
                post_angles.append(angle)
    metrics["set_point_elbow_angle_deg"] = _round(min(pre_angles, default=None))
    metrics["follow_through_extension_deg"] = _round(max(post_angles, default=None))

    if metrics["elbow_angle_at_release_deg"] is None:
        cues.append("Release arm mechanics unavailable: no sufficiently visible shooting arm within 0.15 seconds of the estimated release.")
    else:
        cues.append("Compare projected arm angles with your own shots from the same camera position; this view does not establish an ideal joint angle.")
    if metrics["release_angle_2d_deg"] is not None:
        cues.append("Launch angle is a camera-plane estimate and changes with camera orientation; it is not a calibrated 3D launch angle.")
    return metrics, cues


def _round(value: float | None) -> float | None:
    return round(value, 1) if value is not None and math.isfinite(value) else None


def arc_apexes(ordered: Sequence[Detection], poses: Sequence[PoseFrame], fps: float,
               game_mode: bool = False) -> list[int]:
    """Frames of ball arc apexes, at most one per second; ``ordered`` is sorted by frame."""
    # Candidate apexes: strong upward approach, downward departure, and enough prominence.
    candidates: list[int] = []
    # Wide game footage often has a broad apex near the top of the image.
    # Measure rise and fall over a longer window there; release contact and
    # shoulder-height checks still guard against calling a dribble a shot.
    gap = max(1, round(fps * (0.4 if game_mode else 0.25)))
    frames = [b.frame for b in ordered]
    max_gap = max(2, round(fps * .35))
    for i in range(1, len(ordered) - 1):
        current = ordered[i]
        if current.y > ordered[i - 1].y or current.y > ordered[i + 1].y:
            continue
        before_idx = max(0, bisect_right(frames, current.frame - gap) - 1)
        after_idx = min(len(ordered) - 1, bisect_left(frames, current.frame + gap))
        before, after = ordered[before_idx], ordered[after_idx]
        if any(b - a > max_gap for a, b in zip(frames[before_idx:after_idx], frames[before_idx + 1:after_idx + 1])):
            continue
        if before.y - current.y > 0.015 and after.y - current.y > 0.015:
            # Detector ordering is not identity in games; a random player's shoulder
            # height cannot determine whether another player's ball is a dribble.
            pose = None if game_mode else _nearest_pose(poses, current.frame, fps)
            shoulders = [_point(pose, name) for name in ("left_shoulder", "right_shoulder")] if pose else []
            # Only reject an obvious below-shoulder bounce, with both shoulders visible.
            if len(shoulders) == 2 and all(shoulders) and current.y > max(p[1] for p in shoulders) + .05:
                continue
            if not candidates or current.frame - candidates[-1] > fps * 1.0:
                candidates.append(current.frame)
            elif current.y < next(b.y for b in ordered if b.frame == candidates[-1]):
                candidates[-1] = current.frame
    return candidates


def rim_outcome(segment: Sequence[Detection], apex_frame: int, rim: RimInput,
                net_motion: dict[int, float | dict[str, float]] | None, fps: float
                ) -> tuple[str, float, list[str], int | None]:
    """Made/missed from the descent after ``apex_frame``; the caller checks a rim exists there."""
    outcome, confidence, evidence, outcome_frame = "unknown", 0.0, [], None
    descending = [b for b in segment if b.frame >= apex_frame]
    crossings = []
    for p1, p2 in zip(descending, descending[1:]):
        rim1, rim2 = _rim_at(rim, p1.frame), _rim_at(rim, p2.frame)
        if rim1 is None or rim2 is None:
            continue
        plane1, plane2 = rim1[1] + .45 * rim1[3], rim2[1] + .45 * rim2[3]
        relative1, relative2 = p1.y - plane1, p2.y - plane2
        if relative1 > 0 or relative2 <= relative1 or relative2 < 0:
            continue
        amount = -relative1 / max(relative2 - relative1, 1e-9)
        x = p1.x + amount * (p2.x - p1.x)
        crossing_rim = tuple(rim1[i] + amount * (rim2[i] - rim1[i]) for i in range(4))
        if p2.frame - p1.frame <= max(3, fps * 0.25):
            crossings.append((p2.frame, x, crossing_rim))
    inside = [(f, x) for f, x, box in crossings
              if box[0] - .08 * box[2] <= x <= box[0] + 1.08 * box[2]]
    flow = net_motion or {}
    def motion_event(start: int, end: int) -> tuple[float, int | None]:
        """Return net-specific motion, not camera/background motion.

        Older callers can still pass a numeric, baseline-normalized
        value. Newer callers pass ``strength`` after subtracting a
        surrounding reference region.
        """
        candidates = []
        for frame, value in flow.items():
            if not start <= frame <= end:
                continue
            score = float(value.get("strength", 0.0)) if isinstance(value, dict) else float(value)
            candidates.append((score, frame))
        return max(candidates, default=(0.0, None))

    if inside:
        crossing_frame = inside[0][0]
        motion_score, motion_frame = motion_event(crossing_frame, round(crossing_frame + .35 * fps))
        outcome = "made"
        outcome_frame = crossing_frame
        confidence = min(0.98, 0.68 + 0.18 * min(1.0, motion_score / 2.5))
        evidence += ["Ball center crossed the rim plane downward inside the rim",
                     f"Net-specific motion score: {motion_score:.1f}× baseline"]
    elif crossings:
        outcome = "missed"
        confidence = 0.74
        evidence.append("Descending ball crossed the rim plane outside the rim")
    else:
        # A moving net alone is never enough: an airball can brush the
        # net. When the ball is occluded at the hoop, require a descending
        # path that projects through the rim *and* a delayed net event.
        pre_rim = [b for b in descending if (box := _rim_at(rim, b.frame)) is not None
                   and b.y <= box[1] + .8 * box[3]]
        last = pre_rim[-1] if pre_rim else None
        projected_u = None
        if last:
            prior = [b for b in descending if b.frame < last.frame and b.y < last.y
                     and last.frame - b.frame <= max(3, round(.25 * fps))]
            last_rim = _rim_at(rim, last.frame)
            if prior and last_rim:
                previous = prior[-1]
                previous_rim = _rim_at(rim, previous.frame)
                # Unlike a measured crossing, this intentionally projects
                # the final visible descent beyond the last detection.
                if previous_rim:
                    previous_y = previous.y - (previous_rim[1] + .45 * previous_rim[3])
                    last_y = last.y - (last_rim[1] + .45 * last_rim[3])
                    if previous_y < last_y < 0:
                        alpha = -previous_y / (last_y - previous_y)
                        previous_u = (previous.x - previous_rim[0]) / previous_rim[2]
                        last_u = (last.x - last_rim[0]) / last_rim[2]
                        projected_u = previous_u + alpha * (last_u - previous_u)
        path_through_rim = projected_u is not None and -.08 <= projected_u <= 1.08
        if last and path_through_rim:
            motion_score, motion_frame = motion_event(last.frame, round(last.frame + .4 * fps))
            if motion_score >= 2.2:
                outcome = "likely made"
                outcome_frame = motion_frame or last.frame
                confidence = min(.72, .52 + .10 * min(2.0, motion_score / 2.2))
                evidence += ["Ball was occluded on a descending path projected through the rim",
                             f"Net-specific motion score: {motion_score:.1f}× baseline"]
            else:
                evidence.append("Ball was occluded near the rim, but no net-specific motion confirmed the event")
        else:
            evidence.append("No reliable rim-plane crossing was visible")
    return outcome, confidence, evidence, outcome_frame


def analyze_shots(
    ball: Sequence[Detection],
    poses: Sequence[PoseFrame],
    fps: float,
    rim: RimInput,
    net_motion: dict[int, float | dict[str, float]] | None = None,
    aspect_ratio: float = 1.0,
    handedness: str = "right",
    game_mode: bool = False,
) -> list[ShotResult]:
    """Segment arcs and calculate form/outcome. Coordinates are normalized 0..1."""
    if fps <= 0 or not math.isfinite(fps) or aspect_ratio <= 0 or not math.isfinite(aspect_ratio):
        raise ValueError("fps and aspect_ratio must be finite and positive")
    if handedness not in {"left", "right"}:
        raise ValueError("handedness must be left or right")
    if len(ball) < 4:
        return []
    ordered = sorted(ball, key=lambda b: b.frame)
    frames = [b.frame for b in ordered]
    max_gap = max(2, round(fps * .35))
    candidates = arc_apexes(ordered, poses, fps, game_mode)

    results: list[ShotResult] = []
    for apex_frame in candidates:
        apex_idx = min(range(len(ordered)), key=lambda i: abs(ordered[i].frame - apex_frame))
        lo = bisect_left(frames, apex_frame - fps * 1.25)
        hi = bisect_right(frames, apex_frame + fps * 1.5) - 1
        for index in range(apex_idx, lo, -1):
            if frames[index] - frames[index - 1] > max_gap:
                lo = index
                break
        for index in range(apex_idx, hi):
            if frames[index + 1] - frames[index] > max_gap:
                hi = index
                break
        segment = ordered[lo : hi + 1]
        release = find_release(segment, poses, segment[0].frame, apex_frame, aspect_ratio, handedness, game_mode)
        release_ball = next((b for b in segment if b.frame == release), None)
        release_contact = release_ball is not None and any(
            p.frame == release and (proximity := _release_proximity(p, release_ball, side, aspect_ratio, game_mode)) is not None
            and proximity <= .12
            for p in poses
            for side in (("left", "right") if game_mode else (handedness,))
        )
        if game_mode and not release_contact:
            continue
        if game_mode:
            apex_ball = ordered[apex_idx]
            flight_supported = False
            for pose in poses:
                if pose.frame != release or not any(
                        (contact := _release_proximity(pose, release_ball, side, aspect_ratio, True)) is not None
                        and contact <= .12 for side in ('left', 'right')):
                    continue
                body = [_point(pose, name, aspect_ratio) for name in
                        ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')]
                if not all(body):
                    continue
                shoulder = tuple((body[0][i]+body[1][i])/2 for i in (0,1))
                hip = tuple((body[2][i]+body[3][i])/2 for i in (0,1))
                torso = distance(shoulder, hip)
                if apex_ball.y < shoulder[1] - .75*torso and release_ball.y-apex_ball.y >= .75*torso:
                    flight_supported = True
                    break
            if not flight_supported:
                continue
        outcome, confidence, evidence, outcome_frame = "unknown", 0.0, ["Ball arc detected"], None
        evidence.append(
            "Release time estimated from visible shooting-hand/ball proximity"
            if release_contact else
            "Release contact unavailable: timestamp uses the start of the visible arc and may precede or follow actual release"
        )
        segment_rim = _rim_at(rim, apex_frame)
        if segment_rim:
            outcome, confidence, rim_evidence, outcome_frame = rim_outcome(
                segment, apex_frame, rim, net_motion, fps)
            evidence += rim_evidence
        else:
            evidence.append("Outcome unavailable because the rim was not marked")
        metrics, cues = ({}, []) if game_mode else _form_metrics(poses, segment, release, fps, aspect_ratio, handedness)
        results.append(ShotResult(
            number=len(results) + 1,
            start_s=round(segment[0].time_s, 2),
            release_s=round(release / fps, 2),
            end_s=round(segment[-1].time_s, 2),
            outcome=outcome,
            outcome_confidence=round(confidence, 2),
            evidence=evidence,
            metrics=metrics,
            cues=cues,
            outcome_frame=outcome_frame,
        ))
    return results
