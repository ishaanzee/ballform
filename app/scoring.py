from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from app.geometry import distance, elevation_angle, interpolate_crossing, joint_angle
from app.models import Detection, PoseFrame, ShotResult

RIGHT = {"shoulder": "right_shoulder", "elbow": "right_elbow", "wrist": "right_wrist"}


def _point(pose: PoseFrame, name: str) -> tuple[float, float] | None:
    value = pose.landmarks.get(name)
    if value is None or value[2] < 0.35:
        return None
    return value[0], value[1]


def _nearest_pose(poses: Sequence[PoseFrame], frame: int) -> PoseFrame | None:
    return min(poses, key=lambda p: abs(p.frame - frame), default=None)


def classify_view(poses: Sequence[PoseFrame]) -> tuple[str, float]:
    ratios = []
    for pose in poses:
        ls, rs = _point(pose, "left_shoulder"), _point(pose, "right_shoulder")
        lh, rh = _point(pose, "left_hip"), _point(pose, "right_hip")
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


def find_release(ball: Sequence[Detection], poses: Sequence[PoseFrame], start: int, apex: int) -> int:
    candidates: list[tuple[float, int]] = []
    by_frame = {b.frame: b for b in ball}
    for pose in poses:
        if not start <= pose.frame <= apex:
            continue
        wrist = _point(pose, RIGHT["wrist"])
        detection = by_frame.get(pose.frame)
        if wrist and detection:
            proximity = distance(wrist, (detection.x, detection.y))
            candidates.append((proximity, pose.frame))
    if candidates:
        near_frame = min(candidates)[1]
        # The release follows the last close hand/ball contact before separation.
        threshold = max(0.035, min(c[0] for c in candidates) * 1.8)
        close = [f for d, f in candidates if d <= threshold and f >= near_frame]
        return max(close, default=near_frame)
    return start


def _trajectory_release_angle(ball: Sequence[Detection], release_frame: int, fps: float) -> float | None:
    pts = [b for b in ball if release_frame <= b.frame <= release_frame + max(3, round(fps * 0.30))]
    if len(pts) < 3:
        return None
    x = np.asarray([p.x for p in pts])
    y = np.asarray([p.y for p in pts])
    t = np.asarray([(p.frame - release_frame) / fps for p in pts])
    vx = float(np.polyfit(t, x, 1)[0])
    vy = float(np.polyfit(t, y, 1)[0])
    return math.degrees(math.atan2(-vy, abs(vx)))


def _form_metrics(poses: Sequence[PoseFrame], ball: Sequence[Detection], release: int, fps: float) -> tuple[dict, list[str]]:
    pose = _nearest_pose(poses, release)
    metrics: dict[str, float | str | None] = {
        "elbow_angle_at_release_deg": None,
        "set_point_elbow_angle_deg": None,
        "upper_arm_elevation_deg": None,
        "wrist_over_elbow_pct_shoulder_width": None,
        "release_height_body_ratio": None,
        "follow_through_extension_deg": None,
        "release_angle_2d_deg": _trajectory_release_angle(ball, release, fps),
    }
    cues: list[str] = []
    if pose:
        shoulder, elbow, wrist = (_point(pose, RIGHT[k]) for k in ("shoulder", "elbow", "wrist"))
        if shoulder and elbow and wrist:
            metrics["elbow_angle_at_release_deg"] = _round(joint_angle(shoulder, elbow, wrist))
            metrics["upper_arm_elevation_deg"] = _round(elevation_angle(shoulder, elbow))
            ls, rs = _point(pose, "left_shoulder"), _point(pose, "right_shoulder")
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
        s, e, w = (_point(p, RIGHT[k]) for k in ("shoulder", "elbow", "wrist"))
        if s and e and w:
            angle = joint_angle(s, e, w)
            if angle is not None and release - fps <= p.frame <= release:
                pre_angles.append(angle)
            if angle is not None and release <= p.frame <= release + 0.5 * fps:
                post_angles.append(angle)
    metrics["set_point_elbow_angle_deg"] = _round(min(pre_angles, default=None))
    metrics["follow_through_extension_deg"] = _round(max(post_angles, default=None))

    elbow = metrics["elbow_angle_at_release_deg"]
    extension = metrics["follow_through_extension_deg"]
    release_angle = metrics["release_angle_2d_deg"]
    if isinstance(elbow, float) and elbow < 135:
        cues.append("The shooting elbow was still notably bent at release; check for an earlier, smoother extension.")
    if isinstance(extension, float) and extension < 155:
        cues.append("Follow-through did not reach full arm extension in the visible camera plane.")
    if isinstance(release_angle, float) and release_angle < 40:
        cues.append("The observed launch path was relatively flat; consider whether more arc is comfortable and repeatable.")
    if not cues:
        cues.append("No large arm-mechanics outlier was detected in this view.")
    return metrics, cues


def _round(value: float | None) -> float | None:
    return round(value, 1) if value is not None and math.isfinite(value) else None


def analyze_shots(
    ball: Sequence[Detection],
    poses: Sequence[PoseFrame],
    fps: float,
    rim: tuple[float, float, float, float] | None,
    net_motion: dict[int, float] | None = None,
) -> list[ShotResult]:
    """Segment arcs and calculate form/outcome. Coordinates are normalized 0..1."""
    if len(ball) < 4:
        return []
    ordered = sorted(ball, key=lambda b: b.frame)
    # Candidate apexes: strong upward approach, downward departure, and enough prominence.
    candidates: list[int] = []
    gap = max(1, round(fps * 0.12))
    for i in range(1, len(ordered) - 1):
        before, current, after = ordered[max(0, i - gap)], ordered[i], ordered[min(len(ordered) - 1, i + gap)]
        if before.y - current.y > 0.025 and after.y - current.y > 0.025:
            if not candidates or current.frame - candidates[-1] > fps * 1.0:
                candidates.append(current.frame)
            elif current.y < next(b.y for b in ordered if b.frame == candidates[-1]):
                candidates[-1] = current.frame

    results: list[ShotResult] = []
    for apex_frame in candidates:
        apex_idx = min(range(len(ordered)), key=lambda i: abs(ordered[i].frame - apex_frame))
        lo = max(0, apex_idx - round(fps * 1.25))
        hi = min(len(ordered) - 1, apex_idx + round(fps * 1.5))
        segment = ordered[lo : hi + 1]
        release = find_release(segment, poses, segment[0].frame, apex_frame)
        outcome, confidence, evidence = "unknown", 0.0, ["Ball arc detected"]
        if rim:
            rx, ry, rw, rh = rim
            rim_y = ry + 0.45 * rh
            descending = [b for b in segment if b.frame >= apex_frame]
            crossings = []
            for p1, p2 in zip(descending, descending[1:]):
                x = interpolate_crossing((p1.x, p1.y), (p2.x, p2.y), rim_y)
                if x is not None and p2.frame - p1.frame <= max(3, fps * 0.25):
                    crossings.append((p2.frame, x))
            inside = [(f, x) for f, x in crossings if rx - 0.08 * rw <= x <= rx + 1.08 * rw]
            flow = net_motion or {}
            if crossings:
                crossing_frames = [f for f, _ in crossings]
                motion_score = max(
                    (value for frame, value in flow.items()
                     if any(crossing <= frame <= crossing + .35 * fps for crossing in crossing_frames)),
                    default=0.0,
                )
            else:
                # Detector occlusion is common directly over the rim. Look for a
                # delayed net response during the post-apex part of this shot.
                motion_score = max(
                    (value for frame, value in flow.items()
                     if apex_frame <= frame <= segment[-1].frame),
                    default=0.0,
                )
            if inside:
                outcome = "made"
                confidence = min(0.98, 0.68 + 0.18 * min(1.0, motion_score / 2.5))
                evidence += ["Ball center crossed the rim plane downward inside the rim", f"Net motion score: {motion_score:.1f}× baseline"]
            elif crossings:
                outcome = "missed"
                confidence = 0.74
                evidence.append("Descending ball crossed the rim plane outside the rim")
            elif motion_score > 2.2:
                outcome = "likely made"
                confidence = 0.58
                evidence.append("Ball was lost near the basket, but net motion increased")
            else:
                evidence.append("No reliable rim-plane crossing was visible")
        else:
            evidence.append("Outcome unavailable because the rim was not marked")
        metrics, cues = _form_metrics(poses, segment, release, fps)
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
        ))
    return results
