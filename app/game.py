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


def _prior_pair(player_frames: list[dict], near: dict, shooter_pose: PoseFrame,
                defender_pose: PoseFrame, aspect: float, torso: float):
    target_time = near["time_s"] - .5
    prior = min((frame for frame in player_frames if frame["time_s"] < near["time_s"]),
                key=lambda frame: abs(frame["time_s"] - target_time), default=None)
    if prior is None or abs(prior["time_s"] - target_time) > .12:
        return None
    prior_players: list[PoseFrame] = prior["players"]
    old_shooter = old_defender = None
    if shooter_pose.track_id is not None and defender_pose.track_id is not None:
        old_shooter = next((pose for pose in prior_players
                            if pose.track_id == shooter_pose.track_id), None)
        old_defender = next((pose for pose in prior_players
                             if pose.track_id == defender_pose.track_id), None)
    elif len(near["players"]) == 2 and len(prior_players) == 2:
        current = [_body(pose, aspect) for pose in near["players"]]
        old = [_body(pose, aspect) for pose in prior_players]
        if all(item is not None for item in current + old):
            costs = [sum(math.dist(current[i][0], old[(i + swap) % 2][0]) for i in range(2))
                     for swap in range(2)]
            if abs(costs[0] - costs[1]) / torso >= .5:
                swap = min(range(2), key=lambda index: costs[index])
                shooter_index = near["players"].index(shooter_pose)
                defender_index = near["players"].index(defender_pose)
                old_shooter = prior_players[(shooter_index + swap) % 2]
                old_defender = prior_players[(defender_index + swap) % 2]
    if old_shooter is None or old_defender is None:
        return None
    old_shooter_body, old_defender_body = _body(old_shooter, aspect), _body(old_defender, aspect)
    current_shooter_body, current_defender_body = _body(shooter_pose, aspect), _body(defender_pose, aspect)
    if not all((old_shooter_body, old_defender_body, current_shooter_body, current_defender_body)):
        return None
    movements = [
        math.dist(old_shooter_body[0], current_shooter_body[0]) / torso,
        math.dist(old_defender_body[0], current_defender_body[0]) / torso,
    ]
    scale_changes = [
        abs(old_shooter_body[1] / current_shooter_body[1] - 1),
        abs(old_defender_body[1] / current_defender_body[1] - 1),
    ]
    if max(movements) > 1.5 or max(scale_changes) > .2:
        return None
    before = math.dist(old_shooter_body[0], old_defender_body[0]) / torso
    return before, near["time_s"] - prior["time_s"]


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
        prior_pair = _prior_pair(player_frames, near, shooter_pose, defender_pose,
                                 aspect_ratio, torso)
        if prior_pair is not None:
            before, elapsed = prior_pair
            game["metrics"]["separation_change_torso"] = round(separation - before, 3)
            evidence.append(f"Separation change over {elapsed:.2f} s uses persistent player track IDs.")
        if any(wrist is None for wrist in wrists[defender]):
            evidence.append("Both selected-defender wrists must be visible to measure the contest; score withheld.")
            game["confidence"] = round(.4 * team_confidence, 2)
            visible = [wrist for wrist in wrists[defender] if wrist is not None]
            if visible:
                clearance = min(math.dist(ball_point, wrist) for wrist in visible) / torso
                game["metrics"]["visible_hand_clearance_torso"] = round(clearance, 3)
                base = .65 * game["components"]["separation"]
                game["score_range"] = {
                    "lower": round(base, 1),
                    "upper": round(base + .35 * _component(clearance, .15, 1.35), 1),
                    "reason": "One defender wrist is unobserved. Range covers all possible positions of that hand, conditional on the measured separation and visible hand; it is not a statistical confidence interval.",
                }
                game["status"] = "partial"
            continue
        clearance = min(math.dist(ball_point, wrist) for wrist in wrists[defender]) / torso
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
        required.extend(defender_pose.landmarks[side + "_wrist"][2] for side in ("left", "right"))
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
                         * (1 - abs(ball.time_s - near["time_s"])))
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
