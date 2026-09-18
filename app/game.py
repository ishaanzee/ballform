"""Conservative, single-camera 1v1 shot-space measurements.

The score is a review heuristic, not a make probability or validated player grade.
All distances are aspect-corrected image-plane distances / shooter's torso length.
"""
from __future__ import annotations

import math
from statistics import mean

from app.models import Detection, PoseFrame, ShotResult

LIMITATIONS = [
    "Projected image-plane distances in shooter torso lengths; camera angle and depth can hide separation.",
    "Shot-space score is an unvalidated review heuristic, not expected points, make probability, or player ability.",
    "No court calibration: shot distance, defender depth, possession outcome and true 3D spacing are not measured.",
]
METHOD = {
    "version": "shot-space-v1",
    "label": "Shot-space score",
    "formula": "0.65 × separation component + 0.35 × contest-clearance component",
    "separation_component": "100 × clamp((projected hip separation / shooter torso − 0.5) / 2.5, 0, 1)",
    "contest_component": "100 × clamp((nearest defender wrist to ball / shooter torso − 0.15) / 1.35, 0, 1)",
    "weights": {"separation": 0.65, "contest_clearance": 0.35},
    "confidence_note": "Evidence confidence uses minimum required landmark visibility and ball confidence, penalized for timing offsets and small shooter-association margins; it is heuristic reliability, not a calibrated probability.",
    "note": "Thresholds are transparent heuristics, not calibrated against professional game outcomes. Separation change is descriptive and not scored.",
}


def _point(pose: PoseFrame, name: str, aspect: float):
    p = pose.landmarks.get(name)
    if not p or len(p) < 3 or not all(math.isfinite(v) for v in p) or p[2] < .65:
        return None
    if not (0 <= p[0] <= 1 and 0 <= p[1] <= 1):
        return None
    return (p[0] * aspect, p[1])


def _body(pose, aspect):
    points = [_point(pose, k, aspect) for k in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if any(p is None for p in points):
        return None
    shoulder = tuple((points[0][i] + points[1][i]) / 2 for i in (0, 1))
    hip = tuple((points[2][i] + points[3][i]) / 2 for i in (0, 1))
    torso = math.dist(shoulder, hip)
    return (hip, torso) if torso >= .04 else None


def _component(value, low, span):
    return round(100 * max(0., min(1., (value - low) / span)), 1)


def analyze_game_shots(shots: list[ShotResult], player_frames: list[dict],
                       balls: list[Detection], fps: float, aspect_ratio: float) -> dict:
    """Attach ``game`` evidence to shots; abstain when identity/visibility is unclear."""
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive and finite")
    for shot in shots:
        game = {"score": None, "confidence": 0., "status": "insufficient_evidence",
                "metrics": {"separation_torso": None, "contest_clearance_torso": None,
                            "separation_change_torso": None},
                "components": {}, "evidence": [], "limitations": list(LIMITATIONS),
                "release_frame": None}
        shot.game = game
        evidence = game["evidence"]
        if any(item.startswith("Release contact unavailable") for item in shot.evidence):
            evidence.append("Shot-space score withheld because release contact could not be established.")
            continue
        near = min(player_frames, key=lambda f: abs(f["time_s"] - shot.release_s), default=None)
        if near is None or abs(near["time_s"] - shot.release_s) > .15:
            evidence.append("No player observations within 150 ms of estimated release.")
            continue
        game["release_frame"] = near["frame"]
        players = near["players"]
        if len(players) != 2:
            evidence.append("Exactly two visible players are required to identify the 1v1 matchup.")
            continue
        bodies = [_body(p, aspect_ratio) for p in players]
        if any(b is None for b in bodies):
            evidence.append("Shoulder/hip visibility is insufficient for reliable torso normalization.")
            continue
        ball = min(balls, key=lambda b: abs(b.time_s - near["time_s"]), default=None)
        if (ball is None or abs(ball.time_s - near["time_s"]) > .08
                or abs(ball.time_s - shot.release_s) > .15 or ball.confidence < .5
                or not all(math.isfinite(v) for v in (ball.x, ball.y, ball.confidence))
                or not (0 <= ball.x <= 1 and 0 <= ball.y <= 1)):
            evidence.append("No confident ball observation aligned with the release pose.")
            continue
        bp = (ball.x * aspect_ratio, ball.y)
        wrists = [[_point(p, side + "_wrist", aspect_ratio) for side in ("left", "right")] for p in players]
        # Both players need at least one wrist to rule out a competing ball association.
        if any(all(w is None for w in ws) for ws in wrists):
            evidence.append("Player wrist occlusion prevents reliable shooter identification.")
            continue
        distances = [min(math.dist(bp, w) for w in ws if w is not None) for ws in wrists]
        shooter = min(range(2), key=lambda i: distances[i])
        defender = 1 - shooter
        torso = bodies[shooter][1]
        if distances[shooter] / torso > .9 or (distances[defender] - distances[shooter]) / torso < .3:
            evidence.append("Ball-to-wrist association is ambiguous; shooter identity withheld.")
            continue
        if max(b[1] for b in bodies) / min(b[1] for b in bodies) > 1.8:
            evidence.append("Player torso scales differ strongly; perspective or pose distortion prevents comparison.")
            continue
        separation = math.dist(bodies[shooter][0], bodies[defender][0]) / torso
        game["metrics"]["separation_torso"] = round(separation, 3)
        game["components"]["separation"] = _component(separation, .5, 2.5)
        evidence.append(f"Shooter associated by nearest visible wrist; pose offset {abs(near['time_s'] - shot.release_s):.3f} s from estimated release.")
        if any(w is None for w in wrists[defender]):
            evidence.append("Both defender wrists must be visible to measure nearest-hand contest; score withheld.")
            game["confidence"] = .4
            continue
        clearance = min(math.dist(bp, w) for w in wrists[defender]) / torso
        game["metrics"]["contest_clearance_torso"] = round(clearance, 3)
        game["components"]["contest_clearance"] = _component(clearance, .15, 1.35)
        game["score"] = round(.65 * game["components"]["separation"] + .35 * game["components"]["contest_clearance"], 1)
        required = [p.landmarks[name][2] for p in players
                    for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
        required.extend(players[defender].landmarks[side + "_wrist"][2] for side in ("left", "right"))
        closest_wrist = min((i for i, w in enumerate(wrists[shooter]) if w is not None),
                            key=lambda i: math.dist(bp, wrists[shooter][i]))
        required.append(players[shooter].landmarks[("left", "right")[closest_wrist] + "_wrist"][2])
        margin = (distances[defender] - distances[shooter]) / torso
        association_factor = .7 + .3 * min(1., max(0., (margin - .3) / .7))
        timing_factor = (1 - abs(near["time_s"] - shot.release_s)) * (1 - abs(ball.time_s - near["time_s"]))
        game["confidence"] = round(min(.85, ball.confidence, *required) * association_factor * timing_factor, 2)
        game["status"] = "measured"
        evidence.append("Hip-center separation and nearest defender wrist-to-ball clearance measured in the image plane.")
        prior = min((f for f in player_frames if f["time_s"] < near["time_s"]),
                    key=lambda f: abs(f["time_s"] - (near["time_s"] - .5)), default=None)
        if prior is not None and abs(prior["time_s"] - (near["time_s"] - .5)) <= .12 and len(prior["players"]) == 2:
            old = [_body(p, aspect_ratio) for p in prior["players"]]
            if all(b is not None for b in old):
                costs = [sum(math.dist(bodies[i][0], old[(i + swap) % 2][0]) for i in range(2)) for swap in range(2)]
                swap = min(range(2), key=lambda i: costs[i])
                movements = [math.dist(bodies[i][0], old[(i + swap) % 2][0]) / torso for i in range(2)]
                scale_changes = [abs(old[(i + swap) % 2][1] / bodies[i][1] - 1) for i in range(2)]
                if (abs(costs[0] - costs[1]) / torso >= .5 and max(movements) <= 1.5
                        and max(scale_changes) <= .2):
                    before = math.dist(old[0][0], old[1][0]) / torso
                    game["metrics"]["separation_change_torso"] = round(separation - before, 3)
                    evidence.append(f"Separation change over {near['time_s'] - prior['time_s']:.2f} s; conservative nearest-position association, no persistent player IDs.")
        if game["metrics"]["separation_change_torso"] is None:
            evidence.append("Separation change unavailable: preceding observations or player matching insufficient.")
    scores = [s.game["score"] for s in shots if s.game["score"] is not None]
    return {"total_shots": len(shots), "scored_shots": len(scores),
            "mean_score": round(mean(scores), 1) if scores else None,
            "method": METHOD, "limitations": list(LIMITATIONS)}
