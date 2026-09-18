from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Sequence

import numpy as np

from app.geometry import distance, elevation_angle, interpolate_crossing, joint_angle
from app.models import Detection, PoseFrame, ShotResult

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


def find_release(ball: Sequence[Detection], poses: Sequence[PoseFrame], start: int, apex: int, aspect_ratio: float = 1.0, handedness: str = "right") -> int:
    candidates: list[tuple[float, int]] = []
    by_frame = {b.frame: b for b in ball}
    for pose in poses:
        if not start <= pose.frame <= apex:
            continue
        wrist = _point(pose, f"{handedness}_wrist", aspect_ratio)
        detection = by_frame.get(pose.frame)
        if wrist and detection:
            proximity = distance(wrist, (detection.x * aspect_ratio, detection.y))
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


def analyze_shots(
    ball: Sequence[Detection],
    poses: Sequence[PoseFrame],
    fps: float,
    rim: tuple[float, float, float, float] | None,
    net_motion: dict[int, float] | None = None,
    aspect_ratio: float = 1.0,
    handedness: str = "right",
) -> list[ShotResult]:
    """Segment arcs and calculate form/outcome. Coordinates are normalized 0..1."""
    if fps <= 0 or not math.isfinite(fps) or aspect_ratio <= 0 or not math.isfinite(aspect_ratio):
        raise ValueError("fps and aspect_ratio must be finite and positive")
    if handedness not in {"left", "right"}:
        raise ValueError("handedness must be left or right")
    if len(ball) < 4:
        return []
    ordered = sorted(ball, key=lambda b: b.frame)
    # Candidate apexes: strong upward approach, downward departure, and enough prominence.
    candidates: list[int] = []
    # A quarter-second approach/departure captures smooth apexes at normal
    # frame rates without requiring large movement between adjacent frames.
    gap = max(1, round(fps * 0.25))
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
            pose = _nearest_pose(poses, current.frame, fps)
            shoulders = [_point(pose, name) for name in ("left_shoulder", "right_shoulder")] if pose else []
            # Only reject an obvious below-shoulder bounce, with both shoulders visible.
            if len(shoulders) == 2 and all(shoulders) and current.y > max(p[1] for p in shoulders) + .05:
                continue
            if not candidates or current.frame - candidates[-1] > fps * 1.0:
                candidates.append(current.frame)
            elif current.y < next(b.y for b in ordered if b.frame == candidates[-1]):
                candidates[-1] = current.frame

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
        release = find_release(segment, poses, segment[0].frame, apex_frame, aspect_ratio, handedness)
        release_ball = next((b for b in segment if b.frame == release), None)
        release_contact = release_ball is not None and any(
            p.frame == release and (wrist := _point(p, f"{handedness}_wrist", aspect_ratio)) is not None
            and distance(wrist, (release_ball.x * aspect_ratio, release_ball.y)) <= .12
            for p in poses
        )
        outcome, confidence, evidence = "unknown", 0.0, ["Ball arc detected"]
        evidence.append(
            "Release time estimated from visible shooting-hand/ball proximity"
            if release_contact else
            "Release contact unavailable: timestamp uses the start of the visible arc and may precede or follow actual release"
        )
        if rim:
            rx, ry, rw, rh = rim
            rim_y = ry + 0.45 * rh
            descending = [b for b in segment if b.frame >= apex_frame]
            crossings = []
            for p1, p2 in zip(descending, descending[1:]):
                if p2.y <= p1.y:
                    continue
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
        metrics, cues = _form_metrics(poses, segment, release, fps, aspect_ratio, handedness)
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
