"""Offline ball-handler decoding over a whole camera segment.

The online BallHandlerTracker only sees the past, so it confirms passes late
and drops a dribbler when a grace timer expires. Analysis is offline, so this
scores every tracked player (and "nobody") on every frame and picks the most
consistent sequence with Viterbi decoding: keeping a handler is free, changing
one costs, and a handler without supporting evidence slowly decays to nobody.
"""
from __future__ import annotations

import math

import numpy as np

from app.models import Detection, PoseFrame
from app.tracking import HandlerDecision, body_geometry

# Per-frame evidence is bounded so a few frames cannot outweigh a switch:
# a player's frame score is at most EVIDENCE_WEIGHT.
EVIDENCE_WEIGHT = 1.5
CONTACT = 1.0          # ball within reach of a visible wrist (scaled by ball confidence)
# Contact fades from full credit at FULL_REACH to none at CONTACT_REACH (torso
# lengths). A loose ball bouncing past a hand in the image passes at ~.7-.8;
# held balls and dribble contacts are mostly under ~.5.
FULL_REACH = .4
CONTACT_REACH = .9
RIVAL_MARGIN = .3      # a rival hand this much closer takes all contact credit
UNPOSED_REACH = .45    # ball in the hands/torso area of a detector player with no pose
DRIBBLE_ZONE = .5      # ball low and beside the body, typically mid-dribble
POSSESSION_BOX = .8    # detector's player-in-possession class, scaled by its confidence
BALL_ELSEWHERE = -.6   # ball visible and clearly away from this player
LOOSE_BALL = .6        # "nobody" when a confident ball is near no one, or when a
                       # player without a pose is closer to it than any posed player
NO_EVIDENCE = -.04     # holding with the ball hidden; ~1 s before ending wins
OFF_FRAME = -.25       # held player has no pose on this frame
# Taking the ball from a loose state (a catch) is cheap; taking it from another
# player costs more, so a brief blip on a defender next to a dribbler (a round
# trip through "nobody" costs 4 * START_OR_END) needs ~4 frames of full evidence,
# while a 4-frame catch-and-shoot out of a loose ball still registers.
START_OR_END = 1.25
SWITCH_PLAYER = 4.5


def _wrists(pose: PoseFrame, aspect: float) -> list[tuple[float, float]]:
    return [(p[0] * aspect, p[1]) for name in ("left_wrist", "right_wrist")
            if (p := pose.landmarks.get(name)) is not None and p[2] >= .5]


def _reach(pose: PoseFrame, ball: Detection | None, aspect: float) -> float:
    """Closest visible wrist to the ball, in this player's torso lengths."""
    geometry = body_geometry(pose, aspect)
    if ball is None or ball.confidence < .45 or geometry is None:
        return math.inf
    point = (ball.x * aspect, ball.y)
    return min((math.dist(w, point) / geometry[1] for w in _wrists(pose, aspect)), default=math.inf)


def _unposed_reach(ball: Detection | None, box) -> float:
    """Pseudo reach for a player the detector sees but pose does not (e.g. occluded)."""
    if ball is None or ball.confidence < .45:
        return math.inf
    x1, y1, x2, y2 = box
    # Hands and torso region; the lowest quarter would catch a floor ball
    # beside someone's feet.
    inside = x1 <= ball.x <= x2 and y1 + .1 * (y2 - y1) <= ball.y <= y1 + .75 * (y2 - y1)
    return UNPOSED_REACH if inside else math.inf


def _contact(reach: float, rival_reach: float) -> float:
    """Contact credit in [0, 1]: graded by reach and shared with a closer rival hand."""
    grade = min(1.0, max(0.0, (CONTACT_REACH - reach) / (CONTACT_REACH - FULL_REACH)))
    share = min(1.0, max(0.0, (rival_reach - reach + RIVAL_MARGIN) / (2 * RIVAL_MARGIN)))
    return grade * share


def _evidence(pose: PoseFrame, ball: Detection | None, boxes, aspect: float,
              rival_reach: float = math.inf) -> tuple[float, bool]:
    """Score one player on one frame; the flag marks direct (not inferred) evidence."""
    support, direct = 0.0, False
    geometry = body_geometry(pose, aspect)
    if boxes and geometry is not None:
        hip = (geometry[0][0] / aspect, geometry[0][1])
        for confidence, (x1, y1, x2, y2) in boxes:
            if x1 <= hip[0] <= x2 and y1 <= hip[1] <= y2:
                support, direct = POSSESSION_BOX * confidence, True
                break
    if ball is None or ball.confidence < .45 or geometry is None:
        return EVIDENCE_WEIGHT * support, direct
    hip, torso = geometry
    point = (ball.x * aspect, ball.y)
    reach = _reach(pose, ball, aspect)
    if reach <= CONTACT_REACH:
        contact = CONTACT * ball.confidence * _contact(reach, rival_reach)
        return EVIDENCE_WEIGHT * max(support, contact), direct or contact > 0
    # Low ball beside the hips: between waist and a little below the feet.
    beside = abs(point[0] - hip[0]) <= 1.0 * torso
    low = hip[1] - .3 * torso <= point[1] <= hip[1] + 2.4 * torso
    if beside and low:
        return EVIDENCE_WEIGHT * max(support, DRIBBLE_ZONE), direct
    if support:
        return EVIDENCE_WEIGHT * support, direct
    return (BALL_ELSEWHERE if math.dist(point, hip) > 2.5 * torso else 0.0), direct


def decode_handlers(frames: list[dict], aspect: float) -> list[HandlerDecision]:
    """frames: [{"players": [PoseFrame], "ball": Detection|None, "possession": [(conf, xyxy)],
    "unposed": [(conf, xyxy)]}]; unposed detector players compete for the ball but
    cannot be highlighted, so a ball in their hands decodes as nobody."""
    ids = sorted({p.track_id for f in frames for p in f["players"] if p.track_id is not None})
    if not frames:
        return []
    states = [None, *ids]
    index = {track_id: i for i, track_id in enumerate(states)}
    n, s = len(frames), len(states)
    emission = np.zeros((n, s))
    direct = np.zeros((n, s), dtype=bool)
    for t, frame in enumerate(frames):
        ball = frame.get("ball")
        emission[t, 1:] = OFF_FRAME
        # The loose-ball bonus needs the ball out of every posed player's reach;
        # weak contact credit is not a loose ball (e.g. a ball held overhead).
        near_anyone = False
        tracked = [pose for pose in frame["players"] if pose.track_id is not None]
        reaches = [_reach(pose, ball, aspect) for pose in tracked]
        unposed = min((_unposed_reach(ball, box) for _, box in frame.get("unposed", [])), default=math.inf)
        for i, pose in enumerate(tracked):
            rival = min([unposed, *(r for j, r in enumerate(reaches) if j != i)])
            score, seen = _evidence(pose, ball, frame.get("possession"), aspect, rival)
            k = index[pose.track_id]
            emission[t, k] = score if score else NO_EVIDENCE
            direct[t, k] = seen
            near_anyone |= score >= .5 or reaches[i] <= CONTACT_REACH
        unposed_holds = unposed < min(reaches, default=math.inf)
        if ball is not None and ball.confidence >= .6 and (unposed_holds or not near_anyone):
            emission[t, 0] = LOOSE_BALL
    transition = np.full((s, s), -SWITCH_PLAYER)
    transition[0, :] = transition[:, 0] = -START_OR_END
    np.fill_diagonal(transition, 0.0)
    score = emission[0].copy()
    back = np.zeros((n, s), dtype=np.int32)
    for t in range(1, n):
        candidates = score[:, None] + transition
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], np.arange(s)] + emission[t]
    path = [int(np.argmax(score))]
    for t in range(n - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    decisions = []
    for t, k in enumerate(path):
        track_id = states[k]
        visible = track_id is not None and any(p.track_id == track_id for p in frames[t]["players"])
        if not visible:
            decisions.append(HandlerDecision(None, "none"))
        else:
            decisions.append(HandlerDecision(track_id, "observed" if direct[t, k] else "held"))
    return decisions
