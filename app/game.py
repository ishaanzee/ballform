"""Conservative single-camera shot-space measurements for 1v1 or crowded games.

The score is a review heuristic, not a make probability or validated player grade.
All distances are aspect-corrected image-plane distances / shooter's torso length.
"""
from __future__ import annotations

import math
from statistics import mean

import numpy as np

from app.models import Detection, PoseFrame, ShotResult

LIMITATIONS = [
    "Projected image-plane distances in shooter torso lengths; camera angle and depth can hide separation.",
    "Shot-space score is an unvalidated review heuristic, not expected points, make probability, or player ability.",
    "In crowded footage, opponents are inferred from jersey-colour appearance; similar uniforms, lighting and occlusion can make the matchup unavailable.",
    "No court calibration: shot distance, defender depth, possession outcome and true 3D spacing are not measured.",
]
METHOD = {
    "version": "shot-space-v3-broadcast",
    "label": "Shot-space score",
    "formula": "0.65 × separation component + 0.35 × contest-clearance component",
    "separation_component": "100 × clamp((projected hip separation / shooter torso − 0.5) / 2.5, 0, 1)",
    "contest_component": "100 × clamp((nearest selected-defender wrist to ball / shooter torso − 0.15) / 1.35, 0, 1)",
    "matchup_selection": "Shooter: raised-hand release contact supported by recent tracked possession when available. Defender: nearby opposing raised-hand contest first; otherwise nearest projected opposing hip centre.",
    "weights": {"separation": 0.65, "contest_clearance": 0.35},
    "confidence_note": "Evidence confidence combines landmark visibility, ball confidence, timing, ball-owner margin, jersey-group separation and defender-selection margin. It is heuristic reliability, not a calibrated probability.",
    "note": "Thresholds are transparent heuristics, not calibrated against professional game outcomes. Separation change is descriptive and not scored.",
}


def _point(pose: PoseFrame, name: str, aspect: float):
    point = pose.landmarks.get(name)
    if (not point or len(point) < 3 or not all(math.isfinite(value) for value in point)
            or point[2] < .65 or not (0 <= point[0] <= 1 and 0 <= point[1] <= 1)):
        return None
    return point[0] * aspect, point[1]


def _body(pose: PoseFrame, aspect: float, min_torso: float = .008):
    points = [_point(pose, name, aspect)
              for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if any(point is None for point in points):
        return None
    shoulder = tuple((points[0][axis] + points[1][axis]) / 2 for axis in (0, 1))
    hip = tuple((points[2][axis] + points[3][axis]) / 2 for axis in (0, 1))
    torso = math.dist(shoulder, hip)
    return (hip, torso) if torso >= min_torso else None


def _component(value: float, low: float, span: float) -> float:
    return round(100 * max(0., min(1., (value - low) / span)), 1)


def _raised_wrist(pose, side, aspect):
    wrist = _point(pose, side + "_wrist", aspect)
    shoulders = [_point(pose, name, aspect) for name in ("left_shoulder", "right_shoulder")]
    body = _body(pose, aspect)
    if wrist is None or not all(shoulders) or body is None:
        return None
    shoulder_y = (shoulders[0][1] + shoulders[1][1]) / 2
    return wrist if wrist[1] <= shoulder_y + .15 * body[1] else None


def _possession_votes(player_frames, balls, release_s, aspect):
    """Visible hand contact before flight protects against a background hand crossing the arc."""
    votes = {}
    for frame in player_frames:
        if not release_s - .35 <= frame['time_s'] <= release_s - .04:
            continue
        ball = min(balls, key=lambda b: abs(b.time_s-frame['time_s']), default=None)
        if ball is None or abs(ball.time_s-frame['time_s']) > .025 or ball.confidence < .45:
            continue
        candidates = []
        for pose in frame['players']:
            body = _body(pose, aspect)
            if pose.track_id is None or body is None:
                continue
            wrists = [_point(pose, side+'_wrist', aspect) for side in ('left', 'right')]
            proximity = min((math.dist(w, (ball.x*aspect, ball.y))/body[1] for w in wrists if w), default=math.inf)
            candidates.append((proximity, pose.track_id))
        candidates.sort()
        if (candidates and candidates[0][0] <= .65
                and (len(candidates) < 2 or candidates[1][0]-candidates[0][0] >= .2)):
            identity = candidates[0][1]
            votes[identity] = votes.get(identity, 0) + 1
    return votes


def _appearance_groups(players: list[PoseFrame], shooter: int) -> tuple[dict[int, int], float] | None:
    """Dominant jersey groups, allowing a smaller third group to stay unassigned."""
    descriptors = {
        index: np.asarray(pose.appearance, dtype=float)
        for index, pose in enumerate(players)
        if pose.appearance is not None
        and len(pose.appearance) == 3
        and all(math.isfinite(value) for value in pose.appearance)
    }
    if shooter not in descriptors or len(descriptors) < 3:
        return None
    pair = max(
        ((left, right) for left in descriptors for right in descriptors if left < right),
        key=lambda item: float(np.linalg.norm(descriptors[item[0]] - descriptors[item[1]])),
        default=None,
    )
    if pair is None:
        return None
    centers = [descriptors[pair[0]].copy(), descriptors[pair[1]].copy()]
    # In crowded games a third coherent colour group often belongs to officials.
    # Keep it unassigned, rather than forcibly assigning every person to a team.
    if len(descriptors) >= 7:
        third = max(descriptors, key=lambda i: min(np.linalg.norm(descriptors[i]-c) for c in centers))
        if min(np.linalg.norm(descriptors[third]-c) for c in centers) >= .12:
            centers.append(descriptors[third].copy())
    assignments: dict[int, int] = {}
    for _ in range(8):
        assignments = {
            index: int(np.argmin([np.linalg.norm(value-center) for center in centers]))
            for index, value in descriptors.items()
        }
        if len(set(assignments.values())) < 2:
            return None
        centers = [
            np.mean([descriptors[index] for index, group in assignments.items() if group == target], axis=0)
            if any(group == target for group in assignments.values()) else centers[target]
            for target in range(len(centers))
        ]
    if len(centers) == 3:
        counts = {group: sum(value == group for value in assignments.values()) for group in range(3)}
        teams = sorted(counts, key=lambda group: counts[group], reverse=True)[:2]
        if min(counts[group] for group in teams) < 2:
            return None
        remapping = {group: teams.index(group) for group in teams}
        other = next(group for group in range(3) if group not in teams)
        nearest = min(teams, key=lambda group: float(np.linalg.norm(centers[other]-centers[group])))
        # Shadows/white balance can split one uniform into two shades. Do not
        # throw away a brightly lit defender as if that shade proved a third team.
        if np.linalg.norm(centers[other]-centers[nearest]) < .18:
            remapping[other] = teams.index(nearest)
        if assignments[shooter] not in remapping:
            return None
        assignments = {index: remapping[group] for index, group in assignments.items() if group in remapping}
        centers = [np.mean([descriptors[index] for index, group in assignments.items() if group == target], axis=0)
                   for target in (0, 1)]
    separation = float(np.linalg.norm(centers[0] - centers[1]))
    radii = [
        float(np.linalg.norm(descriptors[index] - centers[group]))
        for index, group in assignments.items()
    ]
    if separation < .12 or separation < 1.6 * max(radii, default=0.):
        return None
    confidence = max(.35, min(1., (separation - max(radii, default=0.)) / .35))
    return assignments, confidence


def _match_prior(frame: dict, near: dict, shooter_pose: PoseFrame, defender_pose: PoseFrame,
                 aspect: float, torso: float) -> tuple[PoseFrame, PoseFrame] | None:
    """Find the shooter and defender poses in an earlier frame."""
    prior_players: list[PoseFrame] = frame["players"]
    if shooter_pose.track_id is not None and defender_pose.track_id is not None:
        old_shooter = next((pose for pose in prior_players if pose.track_id == shooter_pose.track_id), None)
        old_defender = next((pose for pose in prior_players if pose.track_id == defender_pose.track_id), None)
        return (old_shooter, old_defender) if old_shooter and old_defender else None
    if len(near["players"]) != 2 or len(prior_players) != 2:
        return None
    current = [_body(pose, aspect) for pose in near["players"]]
    old = [_body(pose, aspect) for pose in prior_players]
    if any(item is None for item in current + old):
        return None
    costs = [sum(math.dist(current[i][0], old[(i + swap) % 2][0]) for i in range(2)) for swap in range(2)]
    if abs(costs[0] - costs[1]) / torso < .5:
        return None
    swap = min(range(2), key=lambda index: costs[index])
    shooter_index = near["players"].index(shooter_pose)
    defender_index = near["players"].index(defender_pose)
    return prior_players[(shooter_index + swap) % 2], prior_players[(defender_index + swap) % 2]


def _zoom_ratio(frame: dict, near: dict, pairs: list[tuple[PoseFrame, PoseFrame]], aspect: float) -> float | None:
    """Median earlier/current torso ratio over players seen in both frames.

    A camera zoom rescales everyone, while crouching or turning changes one
    player's projected torso, so the median separates zoom from posture.
    """
    by_id = {pose.track_id: pose for pose in frame["players"] if pose.track_id is not None}
    pairs = list(pairs) + [(by_id[pose.track_id], pose) for pose in near["players"]
                           if pose.track_id in by_id and all(pose is not current for _, current in pairs)]
    ratios = []
    for old, current in pairs:
        old_body, current_body = _body(old, aspect), _body(current, aspect)
        if old_body and current_body:
            ratios.append(old_body[1] / current_body[1])
    return float(np.median(ratios)) if ratios else None


def _separation_before(player_frames: list[dict], near: dict, shooter_pose: PoseFrame,
                       defender_pose: PoseFrame, aspect: float, torso: float):
    """Median shooter-defender separation 0.3-0.7 s before release, in release-frame torso units.

    Identity comes from persistent track IDs (or the two-player swap test), so
    individual posture changes no longer void the trend; frames are rescaled
    by the whole view's zoom and skipped when that zoom exceeds 20%.
    """
    tracked = shooter_pose.track_id is not None and defender_pose.track_id is not None
    if tracked:
        # Identity continuity: both tracks present on most frames through release.
        span = [frame for frame in player_frames if 0 < near["time_s"] - frame["time_s"] <= .7]
        present = [frame for frame in span
                   if {shooter_pose.track_id, defender_pose.track_id}
                   <= {pose.track_id for pose in frame["players"]}]
        if not span or len(present) < .5 * len(span):
            return None
    samples = []
    for frame in player_frames:
        elapsed = near["time_s"] - frame["time_s"]
        if not .3 <= elapsed <= .7:
            continue
        matched = _match_prior(frame, near, shooter_pose, defender_pose, aspect, torso)
        if matched is None:
            continue
        old_shooter, old_defender = matched
        old_shooter_body, old_defender_body = _body(old_shooter, aspect), _body(old_defender, aspect)
        if not old_shooter_body or not old_defender_body:
            continue
        zoom = _zoom_ratio(frame, near, [(old_shooter, shooter_pose), (old_defender, defender_pose)], aspect)
        if zoom is None or not .8 <= zoom <= 1.25:
            continue
        samples.append((math.dist(old_shooter_body[0], old_defender_body[0]) / zoom / torso, elapsed))
    if not samples:
        return None
    return float(np.median([value for value, _ in samples])), float(np.median([t for _, t in samples]))


def _contest_hands(player_frames: list[dict], near: dict, defender_pose: PoseFrame,
                   balls: list[Detection], release_ball: Detection, aspect: float, window: float = .1):
    """Each defender wrist's clearance to the ball at the frame nearest release where both are seen.

    A hand briefly lost to occlusion or low keypoint confidence on the release
    frame is usually visible a frame or two either side. Returns
    [(clearance_px_units, offset_s, keypoint_confidence) | None] for left, right.
    """
    frames = sorted((frame for frame in player_frames if abs(frame["time_s"] - near["time_s"]) <= window),
                    key=lambda frame: abs(frame["time_s"] - near["time_s"]))
    hands = []
    for side in ("left", "right"):
        found = None
        for frame in frames:
            if frame is near:
                pose = defender_pose
            elif defender_pose.track_id is not None:
                pose = next((p for p in frame["players"] if p.track_id == defender_pose.track_id), None)
            else:
                pose = None
            wrist = _point(pose, side + "_wrist", aspect) if pose is not None else None
            # The release frame keeps the ball the caller already validated.
            ball = release_ball if frame is near else min(
                balls, key=lambda item: abs(item.time_s - frame["time_s"]), default=None)
            if (wrist is None or ball is None or (frame is not near and (
                    abs(ball.time_s - frame["time_s"]) > .025 or ball.confidence < .5))):
                continue
            found = (math.dist((ball.x * aspect, ball.y), wrist), abs(frame["time_s"] - near["time_s"]),
                     pose.landmarks[side + "_wrist"][2])
            break
        hands.append(found)
    return hands


def analyze_game_shots(shots: list[ShotResult], player_frames: list[dict],
                       balls: list[Detection], fps: float, aspect_ratio: float,
                       min_torso: float = .008) -> dict:
    """Attach game evidence; support 2–10 visible players and abstain on ambiguity."""
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive and finite")
    for shot in shots:
        game = {
            "score": None, "confidence": 0., "status": "insufficient_evidence",
            "score_range": None,
            "metrics": {
                "visible_players": None, "separation_torso": None,
                "contest_clearance_torso": None, "separation_change_torso": None,
                "defender_selection_margin_torso": None,
                "visible_hand_clearance_torso": None,
            },
            "components": {}, "evidence": [], "limitations": list(LIMITATIONS),
            "release_frame": None,
            "players": {
                "ball_carrier_track_id": None,
                "shooter_track_id": None,
                "defender_track_id": None,
            },
        }
        shot.game = game
        evidence = game["evidence"]
        if any(item.startswith("Release contact unavailable") for item in shot.evidence):
            evidence.append("Shot-space score withheld because release contact could not be established.")
            continue
        near = min(player_frames, key=lambda frame: abs(frame["time_s"] - shot.release_s), default=None)
        if near is None or abs(near["time_s"] - shot.release_s) > .15:
            evidence.append("No player observations within 150 ms of estimated release.")
            continue
        game["release_frame"] = near["frame"]
        players: list[PoseFrame] = near["players"]
        game["metrics"]["visible_players"] = len(players)
        if len(players) < 2:
            evidence.append("At least two visible players are required for shot-space analysis.")
            continue
        bodies = [_body(pose, aspect_ratio, min_torso) for pose in players]
        ball = min(balls, key=lambda item: abs(item.time_s - near["time_s"]), default=None)
        if (ball is None or abs(ball.time_s - near["time_s"]) > .08
                or abs(ball.time_s - shot.release_s) > .15 or ball.confidence < .5
                or not all(math.isfinite(value) for value in (ball.x, ball.y, ball.confidence))
                or not (0 <= ball.x <= 1 and 0 <= ball.y <= 1)):
            evidence.append("No confident ball observation aligned with the release pose.")
            continue
        ball_point = (ball.x * aspect_ratio, ball.y)
        wrists = [[_point(pose, side + "_wrist", aspect_ratio) for side in ("left", "right")]
                  for pose in players]
        distances = [
            min((math.dist(ball_point, wrist) for wrist in pose_wrists if wrist is not None),
                default=math.inf)
            for pose_wrists in wrists
        ]
        # A background player's lowered hand can overlap an airborne ball in the
        # image. Shooter candidates need a raised release hand; prior possession
        # provides additional identity evidence when hands overlap at release.
        release_distances = [min((math.dist(ball_point, wrist) for side in ('left', 'right')
                                 if (wrist := _raised_wrist(pose, side, aspect_ratio)) is not None),
                                default=math.inf) for pose in players]
        ranked_owners = sorted((distance, index) for index, distance in enumerate(release_distances)
                               if math.isfinite(distance) and bodies[index] is not None)
        if not ranked_owners:
            evidence.append("No visible player wrist could be associated with the ball.")
            continue
        shooter_distance, shooter = ranked_owners[0]
        votes = _possession_votes(player_frames, balls, shot.release_s, aspect_ratio)
        ordered_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        temporal_owner = None
        if ordered_votes and ordered_votes[0][1] >= 2 and (len(ordered_votes) == 1 or ordered_votes[0][1] >= 1.5*ordered_votes[1][1]):
            temporal_owner = next((index for _, index in ranked_owners if players[index].track_id == ordered_votes[0][0]), None)
        if temporal_owner is not None:
            shooter = temporal_owner
            shooter_distance = release_distances[shooter]
        shooter_body = bodies[shooter]
        torso = shooter_body[1]
        second_distance = min((d for d, i in ranked_owners if i != shooter), default=math.inf)
        ownership_margin = (second_distance - shooter_distance) / torso
        hidden_threat = any(
            bodies[index] is not None and not math.isfinite(distances[index])
            and math.dist(bodies[index][0], ball_point) / torso < 1.25
            for index in range(len(players)) if index != shooter
        )
        if shooter_distance / torso > .9 or (ownership_margin < .3 and temporal_owner is None) or hidden_threat:
            evidence.append("Ball-to-wrist association is ambiguous; shooter identity withheld.")
            continue
        shooter_pose = players[shooter]
        if temporal_owner is not None:
            evidence.append(f"Shooter identity supported by {votes[shooter_pose.track_id]} preceding hand–ball observations.")
        game["players"]["ball_carrier_track_id"] = shooter_pose.track_id
        game["players"]["shooter_track_id"] = shooter_pose.track_id

        team_confidence = 1.
        if len(players) == 2:
            defender_candidates = [1 - shooter]
            selection_method = "the only other visible player"
        else:
            groups = _appearance_groups(players, shooter)
            if groups is None:
                evidence.append(
                    f"{len(players)} players were visible, but jersey appearance did not separate into two reliable groups; defender identity withheld."
                )
                continue
            assignments, team_confidence = groups
            shooter_group = assignments[shooter]
            defender_candidates = [
                index for index, group in assignments.items()
                if group != shooter_group and bodies[index] is not None
            ]
            selection_method = "the nearest body in the jersey-appearance group opposite the shooter"
            if not defender_candidates:
                evidence.append("No visible opponent candidate had sufficient body landmarks.")
                continue

        separations = sorted(
            (math.dist(shooter_body[0], bodies[index][0]) / torso, index)
            for index in defender_candidates
            if bodies[index] is not None and .55 <= bodies[index][1] / torso <= 1.8
        )
        if not separations:
            evidence.append("No opponent candidate had a comparable visible body scale.")
            continue
        separation, defender = separations[0]
        active_contests = sorted(
            (min((math.dist(ball_point, wrist)/torso for side in ('left', 'right')
                  if (wrist := _raised_wrist(players[index], side, aspect_ratio)) is not None),
                 default=math.inf), index)
            for spacing, index in separations if spacing <= 3.5
        )
        if active_contests and active_contests[0][0] <= 1.5:
            defender = active_contests[0][1]
            separation = next(spacing for spacing, index in separations if index == defender)
            selection_method = "a nearby opponent with the closest raised contest hand"
        selection_margin = max(0., min((abs(spacing-separation) for spacing, index in separations if index != defender), default=1.5))
        if len(players) > 2:
            assignments = groups[0]
            unknown_closer = any(
                index not in assignments and index != shooter and bodies[index] is not None
                and math.dist(shooter_body[0], bodies[index][0]) / torso <= separation + .15
                for index in range(len(players))
            )
            if unknown_closer:
                evidence.append("A nearby player had no reliable jersey descriptor; defender identity withheld.")
                continue
        defender_pose = players[defender]
        game["players"]["defender_track_id"] = defender_pose.track_id
        game["metrics"]["separation_torso"] = round(separation, 3)
        game["metrics"]["defender_selection_margin_torso"] = round(selection_margin, 3)
        game["components"]["separation"] = _component(separation, .5, 2.5)
        evidence.append(
            f"{len(players)} players visible. Ball carrier/shooter"
            f"{f' P{shooter_pose.track_id}' if shooter_pose.track_id else ''} associated by raised-hand ball contact; "
            f"defender{f' P{defender_pose.track_id}' if defender_pose.track_id else ''} selected as {selection_method}."
        )
        prior_pair = _separation_before(player_frames, near, shooter_pose, defender_pose,
                                        aspect_ratio, torso)
        if prior_pair is not None:
            before, elapsed = prior_pair
            game["metrics"]["separation_change_torso"] = round(separation - before, 3)
            evidence.append(f"Separation change is measured against the median over 0.3–0.7 s before release "
                            f"(median {elapsed:.2f} s), using persistent player identities and correcting for zoom.")
        hands = _contest_hands(player_frames, near, defender_pose, balls, ball, aspect_ratio)
        offset = max((hand[1] for hand in hands if hand is not None), default=0.)
        if offset > 0:
            evidence.append(f"A defender hand hidden on the release frame was measured on the nearest frame where it "
                            f"was visible, at most {offset:.2f} s from release.")
        if any(hand is None for hand in hands):
            evidence.append("Both selected-defender wrists must be visible within 0.1 s of release to measure the contest; score withheld.")
            game["confidence"] = round(.4 * team_confidence, 2)
            visible = [hand[0] for hand in hands if hand is not None]
            if visible:
                clearance = min(visible) / torso
                game["metrics"]["visible_hand_clearance_torso"] = round(clearance, 3)
                base = .65 * game["components"]["separation"]
                game["score_range"] = {
                    "lower": round(base, 1),
                    "upper": round(base + .35 * _component(clearance, .15, 1.35), 1),
                    "reason": "One defender wrist is unobserved. Range covers all possible positions of that hand, conditional on the measured separation and visible hand; it is not a statistical confidence interval.",
                }
                game["status"] = "partial"
            continue
        clearance = min(hand[0] for hand in hands) / torso
        game["metrics"]["contest_clearance_torso"] = round(clearance, 3)
        game["components"]["contest_clearance"] = _component(clearance, .15, 1.35)
        game["score"] = round(
            .65 * game["components"]["separation"]
            + .35 * game["components"]["contest_clearance"], 1
        )
        required = [
            pose.landmarks[name][2]
            for pose in (shooter_pose, defender_pose)
            for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        ]
        required.extend(hand[2] for hand in hands)
        closest_wrist = min(
            (index for index, wrist in enumerate(wrists[shooter]) if wrist is not None),
            key=lambda index: math.dist(ball_point, wrists[shooter][index]),
        )
        required.append(shooter_pose.landmarks[
            ("left", "right")[closest_wrist] + "_wrist"
        ][2])
        ownership_factor = .7 + .3 * min(1., max(0., (ownership_margin - .3) / .7))
        defender_factor = .72 + .28 * min(1., max(0., selection_margin / .6))
        timing_factor = ((1 - abs(near["time_s"] - shot.release_s))
                         * (1 - abs(ball.time_s - near["time_s"])) * (1 - offset))
        game["confidence"] = round(
            min(.85, ball.confidence, *required)
            * ownership_factor * defender_factor * team_confidence * timing_factor, 2
        )
        game["status"] = "measured"
        evidence.append(
            "Hip-centre separation and selected-defender wrist-to-ball clearance measured in the image plane."
        )
        if prior_pair is None:
            evidence.append(
                "Separation change unavailable: preceding observations or player identity continuity were insufficient."
            )
    scores = [shot.game["score"] for shot in shots if shot.game["score"] is not None]
    return {
        "total_shots": len(shots), "scored_shots": len(scores),
        "mean_score": round(mean(scores), 1) if scores else None,
        "method": METHOD, "limitations": list(LIMITATIONS),
    }


COURT_METRICS = ("shot_distance_ft", "shot_zone", "shooter_court_x_ft", "shooter_court_y_ft",
                 "separation_ft", "contest_clearance_ft", "visible_hand_clearance_ft")
COURT_LIMITATION = (
    "Court calibration: feet are measured on the floor through a homography from user-marked court landmarks, "
    "followed through camera motion. Player positions come from the feet (ankles, or the pose box bottom when ankles "
    "are hidden); contest clearance in feet assumes the defender's hand and the ball are at the shooter's depth. "
    "Measurements outside the marked landmarks are extrapolated and less accurate."
)


def _torso_px(pose: PoseFrame, width: int, height: int) -> float | None:
    points = [pose.landmarks.get(name) for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if any(p is None or p[2] < .5 for p in points):
        return None
    shoulder = ((points[0][0] + points[1][0]) / 2 * width, (points[0][1] + points[1][1]) / 2 * height)
    hip = ((points[2][0] + points[3][0]) / 2 * width, (points[2][1] + points[3][1]) / 2 * height)
    return math.dist(shoulder, hip)


def _grounded_position(player_frames: list[dict], track_id: int, near: dict, court_map, window: float = .8):
    """Court position (feet) of a player on the floor at or just before release, or a reason it is unavailable.

    An airborne foot maps through the floor homography to a point beyond the
    player, so for a jump the last grounded frames before take-off are used.
    """
    from app.court import floor_point, lowest_foot_y, takeoff

    width, height = court_map.width, court_map.height
    samples = sorted(((frame, pose) for frame in player_frames
                      if 0 <= near["time_s"] - frame["time_s"] <= window
                      for pose in frame["players"] if pose.track_id == track_id),
                     key=lambda item: item[0]["time_s"])
    if not samples or samples[-1][0] is not near:
        return "the player was not tracked on the release frame"
    if court_map.reliable(near["frame"]) is None:
        return "the court mapping was not reliable on the release frame"
    foot_y, torso = [], []
    for frame, pose in samples:
        point, low = floor_point(pose, width, height), lowest_foot_y(pose, height)
        steady = court_map.to_anchor_image(frame["frame"], (point[0][0], low)) if point and low is not None else None
        foot_y.append(steady[1] if steady else None)
        if (length := _torso_px(pose, width, height)) is not None:
            torso.append(length)
    jump = takeoff(foot_y, float(np.median(torso)) if torso else 0.)
    if jump is None:
        return "the feet were not visible before release"
    used = jump.grounded if jump.jumped else [len(samples) - 1]
    points, methods = [], set()
    for index in used:
        frame, pose = samples[index]
        found = floor_point(pose, width, height)
        court_point = court_map.to_court(frame["frame"], found[0]) if found else None
        if court_point is not None:
            points.append(court_point)
            methods.add(found[1])
    if not points:
        return "no grounded frame had a reliable court mapping"
    first = samples[used[0]][0]
    return (tuple(float(v) for v in np.median(np.asarray(points), axis=0)),
            {"jumped": jump.jumped, "rise_torso": jump.rise_torso, "frames": [samples[i][0]["frame"] for i in used],
             "before_release_s": near["time_s"] - samples[used[-1]][0]["time_s"],
             "first_frame": first["frame"], "method": " / ".join(sorted(methods))})


def _contest_points(player_frames: list[dict], near: dict, defender_id: int, balls: list[Detection],
                    window: float = .1):
    """Per defender hand: (frame, wrist pixel, ball pixel) nearest release, with the rules of _contest_hands."""
    release_ball = min(balls, key=lambda item: abs(item.time_s - near["time_s"]), default=None)
    frames = sorted((frame for frame in player_frames if abs(frame["time_s"] - near["time_s"]) <= window),
                    key=lambda frame: abs(frame["time_s"] - near["time_s"]))
    hands = []
    for side in ("left", "right"):
        found = None
        for frame in frames:
            pose = next((p for p in frame["players"] if p.track_id == defender_id), None)
            wrist = pose.landmarks.get(side + "_wrist") if pose is not None else None
            ball = release_ball if frame is near else min(
                balls, key=lambda item: abs(item.time_s - frame["time_s"]), default=None)
            if (wrist is None or wrist[2] < .65 or ball is None or (frame is not near and (
                    abs(ball.time_s - frame["time_s"]) > .025 or ball.confidence < .5))):
                continue
            found = (frame["frame"], (wrist[0], wrist[1]), (ball.x, ball.y))
            break
        hands.append(found)
    return hands


def add_court_metrics(shots: list[ShotResult], player_frames: list[dict], balls: list[Detection], court_map,
                      summary: dict | None = None) -> None:
    """Add floor measurements in feet next to the torso-length metrics, which stay unchanged."""
    from app.court import three_point_margin, vertical_plane_distance, zone

    court = court_map.court
    if summary is not None:
        summary["limitations"] = [COURT_LIMITATION if item.startswith("No court calibration") else item
                                  for item in summary["limitations"]]
    by_frame = {frame["frame"]: frame for frame in player_frames}
    for shot in shots:
        game = shot.game
        if game is None:
            continue
        metrics, evidence = game["metrics"], game["evidence"]
        metrics.update({key: None for key in COURT_METRICS})
        game["limitations"] = [COURT_LIMITATION if item.startswith("No court calibration") else item
                               for item in game["limitations"]]
        near = by_frame.get(game["release_frame"])
        shooter_id = game["players"]["shooter_track_id"]
        if near is None or shooter_id is None:
            evidence.append("Court: shot distance unavailable because the shooter was not identified at release.")
            continue
        shooter = _grounded_position(player_frames, shooter_id, near, court_map)
        if isinstance(shooter, str):
            evidence.append(f"Court: shot distance unavailable because {shooter}.")
            continue
        spot, detail = shooter
        distance = math.dist(spot, court.rim)
        shot_zone = zone(spot, court)
        metrics.update({"shot_distance_ft": round(distance, 1), "shot_zone": shot_zone,
                        "shooter_court_x_ft": round(spot[0], 1), "shooter_court_y_ft": round(spot[1], 1)})
        if detail["jumped"]:
            where = (f"the shooter's feet ({detail['method']}) on the last grounded frame"
                     f"{'s' if len(detail['frames']) > 1 else ''} before take-off, "
                     f"{detail['before_release_s']:.2f} s before release (frame {detail['frames'][-1]}); "
                     f"the feet then rose {detail['rise_torso']:.1f} torso lengths")
        else:
            where = (f"the shooter's feet ({detail['method']}) on the release frame; no take-off was detected, "
                     "so the shot was treated as taken from the floor")
        margin = three_point_margin(spot, court)
        evidence.append(f"Court: shot distance {distance:.1f} ft to the floor point under the rim, measured from {where}. "
                        f"{court.dims['label']} {'zone ' + shot_zone.replace('_', ' ') if shot_zone else 'zone unavailable (off the calibrated half)'}"
                        f", {abs(margin):.1f} ft {'behind' if margin >= 0 else 'inside'} the three-point line.")
        extrapolated = court_map.calibration.extrapolation_ft(spot)
        if extrapolated > 2:
            evidence.append(f"Court: the shooter stood {extrapolated:.0f} ft outside the marked landmarks, so the "
                            "position is extrapolated; mark a landmark nearer the shot for a firmer measurement.")
        defender_id = game["players"]["defender_track_id"]
        if defender_id is None or metrics.get("separation_torso") is None:
            continue
        defender = _grounded_position(player_frames, defender_id, near, court_map)
        if isinstance(defender, str):
            evidence.append(f"Court: separation in feet unavailable because {defender}.")
        else:
            metrics["separation_ft"] = round(math.dist(spot, defender[0]), 1)
            evidence.append("Court: separation in feet is measured on the floor from the shooter's take-off spot to "
                            f"the defender's feet ({'last grounded frames' if defender[1]['jumped'] else 'release frame'}).")
        hands = []
        for found in _contest_points(player_frames, near, defender_id, balls):
            camera = court_map.camera(found[0]) if found else None
            if found is None or camera is None:
                hands.append(None)
                continue
            frame, wrist, ball = found
            size = (court_map.width, court_map.height)
            hands.append(vertical_plane_distance(camera, spot, (wrist[0] * size[0], wrist[1] * size[1]),
                                                 (ball[0] * size[0], ball[1] * size[1])))
        visible = [hand for hand in hands if hand is not None]
        if metrics.get("contest_clearance_torso") is not None and len(visible) == 2:
            metrics["contest_clearance_ft"] = round(min(visible), 1)
        elif metrics.get("visible_hand_clearance_torso") is not None and visible:
            metrics["visible_hand_clearance_ft"] = round(min(visible), 1)
        if visible:
            evidence.append("Court: contest clearance in feet is approximate; it assumes the defender's hand and the "
                            "ball are at the shooter's depth, using the camera recovered from the floor mapping.")
