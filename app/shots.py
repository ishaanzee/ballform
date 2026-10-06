"""Game-mode shot coverage beyond the raised-hand jump-shot arc.

``analyze_shots`` finds jump shots from a ball arc plus visible raised-hand
release contact. This module labels every game shot with a type and adds the
attempts that path misses:

* Arc shots that reach the basket within RIM_FLIGHT_S of release: rim
  finishes that drew a small arc, typed tip or "layup or dunk".
* Hidden-release jump shots: the detector's jump-shot class and a ball arc
  above the last visible hand contact, when the release itself was hidden.
* Rim attempts (layups, dunks, tips, putbacks): the ball reaches the basket,
  seen as a confident ball-in-basket detection or as the ball entering the
  rim's area, soon after a player's hand contact.
* Layup-dunk class attempts: when neither a rim nor a basket detection exists,
  a layup-dunk class run followed by the ball rising above the player's head.

One attempt is one hand contact followed by the ball reaching the basket. A
basket event with no new contact since the previous attempt got there (a ball
rattling on the rim, a rebound) belongs to that attempt. The shooter for these
attempts is the last player in unambiguous hand contact, checked against the
decoded ball handler; a disagreement withholds the shooter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.models import Detection, PoseFrame, ShotResult
from app.possession import _unposed_reach
from app.scoring import RimInput, _rim_at, arc_apexes, rim_by_arrival, rim_outcome
from app.tracking import body_geometry

# Wrist-to-ball distance in torso lengths, as game.py's possession votes; a
# second hand within the margin makes the contact ambiguous.
CONTACT_REACH = .65
CONTACT_MARGIN = .2
# A ball above a raised hand sits on the fingertips, a hand's length past the
# wrist: tippers' wrists were 0.7-0.97 torso lengths from the ball (3b4f,
# 699d, ec07).
FINGERTIP_REACH = .9
# A layup, dunk or tip reaches the basket this soon after the last touch.
CONTACT_TO_BASKET_S = 1.2
# A single low-confidence ball-in-basket frame fires on an empty net (3074e
# frames 54 and 136); a lone detection must be this confident.
BASKET_CONFIDENCE = .5
# Detector class runs this close to an existing shot's release belong to it.
SHOT_WINDOW_S = .7
# A shot's flight can claim basket events for this long after release.
FLIGHT_S = 3.0
# A contact this close to the basket (torso lengths) is at the rim.
AT_RIM = 1.2
TIP_WINDOW_S = 2.5
# The ball can reach the rim's area still in the shooter's hands (a dunk, a
# layup carried up). The same player's contacts up to this long after it, with
# gaps of at most HOLD_GAP_S, are still that hold.
HOLD_PAST_BASKET_S = .4
HOLD_GAP_S = .2
# A rim attempt and an arc shot released this close together are one attempt
# (the arc caught the gather); labeled tips came at least 0.8 s after the
# previous shot reached the basket.
SAME_ATTEMPT_S = .5
# A shot in flight stops claiming basket events once one player has this many
# sampled contacts with the ball after it (caught, not brushed: the 2fcb fan
# pass-overs touched it on two).
CAUGHT_CONTACTS = 4
# Labeled tips touched the ball 0.80-0.90 s after the previous attempt reached
# the basket, so a rim attempt's make seen later than this may be a follow-up's.
FOLLOW_UP_S = .8
# An arc shot whose ball reaches the basket this soon after release was let go
# at the rim. On the labeled broadcast clips, jump shots took 1.05-1.38 s,
# floaters 0.50-0.65 s, and 11 of 14 rim finishes 0.30 s or less (the other
# three had a release estimate 0.3-0.6 s early).
RIM_FLIGHT_S = .4
# Such a finish within this long of the previous shot reaching the basket is a
# tip: labeled tips came 0.80-0.90 s after, a gathered putback dunk 1.48 s.
TIP_AFTER_REACH_S = 1.
# A touch bends the ball's path. In 2D a ball flying past a hand, e.g. a fan's
# behind the baseline, stays on a parabola: the largest residual around the
# contact was 0.022 there versus 0.050-0.065 at three real releases (units of
# frame height; ball radius about 0.016).
FREE_FLIGHT_RADII = 2.5
RIM_TYPES = {"layup", "dunk", "layup or dunk", "tip"}
# An arc's apex comes within this long of release, and its descent within as
# long again of the apex.
ARC_APEX_S = 1.5
# A ball coming down this many rim widths to the side of the rim's centre is
# beside the basket, not at it.
AWAY_RIM_WIDTHS = 1.
# A blocked shot never reaches the basket either: a confident shot-block class
# between release and apex keeps such an arc (76d9, Durant's block).
BLOCK_CONFIDENCE = .8
# An arc's rise must be seen: the ball going up this many of its radii between
# confident detections at most 0.1 s apart, from release to apex. Labeled shots
# were seen rising 2.5 radii or more; arcs drawn from stray detections across a
# lost ball 0-0.03 (cd04, c467, 567b, d3d6: a net read as the ball).
ARC_RISE_RADII = 1.
# No attempt is released this soon after a make. On the dev labels the next
# real attempt came 1.28 s or more after a "made" call (25e9, after a false
# make); hands on the ball falling through the net came 0.15-0.32 s after (b212, 25e9).
DEAD_BALL_S = 1.
# With no rim or basket detection to check the ball against, a layup-dunk
# class run must be this confident. On the dev labels the path found no real
# attempt; its two finds were one rebound tip-out seen in two overlapping clips
# (8e57, dd76), at peaks 0.51 and 0.57.
LAYUP_CLASS_PEAK = .7
# A touch this soon after a shot reaches the basket is part of its arrival (a
# contest, a hand behind the rim in 2D: 20f2 0.07 s), not a follow-up; the
# ball has to come off the rim first. The earliest labeled follow-up touched
# it 0.25 s after (883b, a miss caught short of the rim).
SETTLE_S = .15


@dataclass
class Run:
    start: int
    end: int
    count: int
    peak: float
    box: tuple[float, float, float, float]


@dataclass
class Contact:
    frame: int
    track_id: int
    pose: PoseFrame
    torso: float
    ball: Detection
    raised: bool


@dataclass
class BasketEvent:
    frame: int
    sources: list[str]
    peak: float
    location: tuple[float, float]


@dataclass
class Anchor:
    release: int
    reached: int | None
    shot: ShotResult


def event_runs(frames: list[dict], name: str, fps: float, gap_s: float = .2) -> list[Run]:
    """Runs of one detector event class over sampled frames, merging gaps up to ``gap_s``."""
    runs: list[Run] = []
    for frame in frames:
        hits = [(conf, tuple(box)) for kind, conf, box in frame.get("events", []) if kind == name]
        if not hits:
            continue
        conf, box = max(hits)
        if runs and frame["frame"] - runs[-1].end <= gap_s * fps:
            run = runs[-1]
            run.end, run.count = frame["frame"], run.count + 1
            if conf > run.peak:
                run.peak, run.box = conf, box
        else:
            runs.append(Run(frame["frame"], frame["frame"], 1, conf, box))
    return runs


def hand_contacts(frames: list[dict], balls: list[Detection], aspect: float) -> list[Contact]:
    """Frames where exactly one tracked player's wrist is on a confident ball."""
    by_frame = {ball.frame: ball for ball in balls}
    contacts = []
    for frame in frames:
        ball = by_frame.get(frame["frame"])
        if ball is None or ball.confidence < .45:
            continue
        point = (ball.x * aspect, ball.y)
        ranked = []
        for pose in frame["players"]:
            geometry = body_geometry(pose, aspect)
            if geometry is None:
                continue
            wrists = [(p[0] * aspect, p[1]) for name in ("left_wrist", "right_wrist")
                      if (p := pose.landmarks.get(name)) is not None and p[2] >= .5]
            if wrists:
                wrist = min(wrists, key=lambda w: math.dist(w, point))
                ranked.append((math.dist(wrist, point) / geometry[1], pose, geometry[1], wrist))
        ranked.sort(key=lambda item: item[0])
        if not ranked or ranked[0][0] > FINGERTIP_REACH:
            continue
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < CONTACT_MARGIN:
            continue
        # A detector player without a pose holding the ball is a hidden rival.
        if any(_unposed_reach(ball, box) < math.inf for _, box in frame.get("unposed", [])):
            continue
        reach, pose, torso, wrist = ranked[0]
        shoulders = [p for name in ("left_shoulder", "right_shoulder")
                     if (p := pose.landmarks.get(name)) is not None and p[2] >= .5]
        if pose.track_id is None or len(shoulders) < 2:
            continue
        raised = wrist[1] <= (shoulders[0][1] + shoulders[1][1]) / 2 + .15 * torso
        if reach > CONTACT_REACH and not (raised and ball.y < wrist[1]):
            continue
        contacts.append(Contact(frame["frame"], pose.track_id, pose, torso, ball, raised))
    return contacts


def _near_box(ball: Detection, box: tuple[float, float, float, float]) -> bool:
    """The ball is within one box size of a detector box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return x1 - w <= ball.x <= x2 + w and y1 - h <= ball.y <= y2 + h


def _in_rim_area(ball: Detection, box: tuple[float, float, float, float]) -> bool:
    """The ball is at the basket: over or just beside the rim box (x, y, w, h), up to a net's length below."""
    return (box[0] - .25 * box[2] <= ball.x <= box[0] + 1.25 * box[2]
            and box[1] - 1.5 * box[3] <= ball.y <= box[1] + 1.2 * box[3])


def basket_events(frames: list[dict], balls: list[Detection], rim: RimInput, fps: float,
                  contact_frames: list[int] = ()) -> list[BasketEvent]:
    """Moments the ball reaches the basket: ball-in-basket detections and rim-area entries.

    Neither means a make: lone ball-in-basket detections also fire on an empty
    net, and a ball in the rim's area can still miss. Detections within
    0.5 s merge into one event unless a hand touched the ball in between (a tip).
    """
    events = []
    by_frame = {ball.frame: ball for ball in balls}
    for run in event_runs(frames, "ball_in_basket", fps):
        # An empty net fires too: drop the run when the ball was seen confidently
        # elsewhere on its frames, and never at the box.
        seen = [ball for frame in range(run.start, run.end + 1)
                if (ball := by_frame.get(frame)) is not None and ball.confidence >= .45]
        if seen and not any(_near_box(ball, run.box) for ball in seen):
            continue
        if run.peak >= BASKET_CONFIDENCE or run.count >= 2:
            x1, y1, x2, y2 = run.box
            events.append(BasketEvent(run.start, ["ball_in_basket"], run.peak, ((x1 + x2) / 2, (y1 + y2) / 2)))
    if rim is not None:
        # An entry needs the ball confidently outside first; a low-confidence
        # frame inside the area is not an exit and re-entry.
        was_inside = False
        for ball in balls:
            box = _rim_at(rim, ball.frame)
            if box is None or ball.confidence < .45:
                continue
            inside = _in_rim_area(ball, box)
            if inside and not was_inside:
                events.append(BasketEvent(ball.frame, ["rim_area"], 0., (box[0] + box[2] / 2, box[1] + .45 * box[3])))
            was_inside = inside
    merged: list[BasketEvent] = []
    for event in sorted(events, key=lambda e: e.frame):
        if (merged and event.frame - merged[-1].frame <= .5 * fps
                and not any(merged[-1].frame < frame < event.frame for frame in contact_frames)):
            last = merged[-1]
            last.sources = sorted(set(last.sources) | set(event.sources))
            last.peak = max(last.peak, event.peak)
            if "rim_area" in event.sources:
                last.location = event.location
        else:
            merged.append(event)
    return merged


def _torso_distance(contact: Contact, location: tuple[float, float], aspect: float) -> tuple[float, float]:
    """Horizontal and vertical offsets of the contact ball from a basket point, in torso lengths."""
    return ((contact.ball.x - location[0]) * aspect / contact.torso,
            (contact.ball.y - location[1]) / contact.torso)


def _net_motion_peak(net_motion, start: int, end: int) -> float:
    scores = [float(value.get("strength", 0.)) if isinstance(value, dict) else float(value)
              for frame, value in (net_motion or {}).items() if start <= frame <= end]
    return max(scores, default=0.)


def _handler_before(frames: list[dict], frame: int, fps: float, seconds: float = 1.) -> int | None:
    """The last observed decoded ball handler in the ``seconds`` up to ``frame``."""
    for data in reversed(frames):
        if data["frame"] > frame:
            continue
        if frame - data["frame"] > seconds * fps:
            break
        handler = data.get("handler")
        if handler is not None and handler.source == "observed" and handler.track_id is not None:
            return handler.track_id
    return None


def free_flight(balls: list[Detection], frame: int, fps: float, aspect: float) -> bool:
    """True when the ball stays on one parabola through ``frame``: nothing touched it.

    Needs three confident observations on each side within 0.3 s; otherwise the
    contact is not rejected.
    """
    points = [b for b in balls if abs(b.frame - frame) <= .3 * fps and b.confidence >= .45]
    if sum(b.frame < frame for b in points) < 3 or sum(b.frame > frame for b in points) < 3:
        return False
    t = np.asarray([(b.frame - frame) / fps for b in points])
    x = np.asarray([b.x * aspect for b in points])
    y = np.asarray([b.y for b in points])
    residual = np.hypot(x - np.polyval(np.polyfit(t, x, 1), t), y - np.polyval(np.polyfit(t, y, 2), t))
    radius = float(np.mean([b.radius for b in points])) * aspect
    return float(residual.max()) < max(.01, FREE_FLIGHT_RADII * radius)


def came_down_away(balls: list[Detection], apex: int | None, rim: RimInput, fps: float,
                   reach: int | None = None) -> bool:
    """True when the ball was below the rim, away from it, from the arc's apex on
    and before reaching the basket (frame ``reach``, if it ever did): a pass, not a shot.

    A shot's ball gets to the basket before it comes back down. A ball lost
    before it came back down (occlusion, blur, no rim known) proves nothing.
    """
    if apex is None:
        return False
    for ball in balls:
        if ball.frame < apex or ball.confidence < .45:
            continue
        if ball.frame > apex + ARC_APEX_S * fps or (reach is not None and ball.frame >= reach):
            break
        box = _rim_at(rim, ball.frame)
        if (box is not None and ball.y > box[1] + .45 * box[3]
                and abs(ball.x - box[0] - box[2] / 2) > AWAY_RIM_WIDTHS * box[2]):
            return True
    return False


def _seen_rising(balls: list[Detection], start: int, end: int, fps: float) -> float:
    """How far the ball was seen going up between two frames, in ball radii: upward steps
    between confident detections at most 0.1 s apart (a radius of 0.01 frame heights when
    the detector gave none)."""
    seen = [b for b in balls if start <= b.frame <= end and b.confidence >= .45]
    rise = sum(max(0., a.y - b.y) for a, b in zip(seen, seen[1:]) if b.frame - a.frame <= .1 * fps)
    radius = float(np.mean([b.radius for b in seen])) if seen else 0.
    return rise / (radius or .01)


def _held_since(contacts: list[Contact], contact: Contact, frame: int, fps: float) -> bool:
    """The contact's player has had the ball since ``frame``, with gaps of at most HOLD_GAP_S."""
    held = contact.frame
    for earlier in reversed(contacts):
        if earlier.frame >= held or earlier.track_id != contact.track_id:
            continue
        if held - earlier.frame > HOLD_GAP_S * fps or earlier.frame < frame:
            break
        held = earlier.frame
    return held <= frame


def _flight_supported(contact: Contact, apex: Detection) -> bool:
    """The arc path's rule: apex 0.75 torso above the shoulders and 0.75 torso of rise."""
    shoulders = [contact.pose.landmarks[name][1] for name in ("left_shoulder", "right_shoulder")]
    return (apex.y < sum(shoulders) / 2 - .75 * contact.torso
            and contact.ball.y - apex.y >= .75 * contact.torso)


def _class_note(run: Run | None, label: str) -> list[str]:
    if run is None:
        return []
    return [f"Detector {label} class fired on {run.count} sampled frames (peak {run.peak:.2f}), frames {run.start}–{run.end}"]


def _overlapping(runs: list[Run], start: int, end: int) -> Run | None:
    return max((run for run in runs if run.start <= end and run.end >= start), key=lambda r: r.peak, default=None)


def _outcome(balls, start: int, event: BasketEvent | None, apex_frame: int, rim: RimInput,
             net_motion, fps: float) -> tuple[str, float, list[str], int | None]:
    # A dunk or a layup carried up crosses the rim plane before the hold ends
    # (``start``), so the descent is followed from the top of the ball's path.
    segment = [b for b in balls if min(start, apex_frame) <= b.frame <= apex_frame + 1.5 * fps]
    if not rim_by_arrival(rim, apex_frame, fps) or not segment:
        evidence = ["Outcome unavailable because the rim was not marked"]
        if event is not None and "ball_in_basket" in event.sources:
            evidence.append(f"The detector's ball-in-basket class fired (peak {event.peak:.2f}) at frame {event.frame}; "
                            "lone detections also fire on an empty net, so it is not treated as a make without a rim.")
        return "unknown", 0., evidence, None
    outcome, confidence, evidence, frame = rim_outcome(segment, apex_frame, rim, net_motion, fps, balls)
    if event is not None and "ball_in_basket" in event.sources:
        motion = _net_motion_peak(net_motion, event.frame, round(event.frame + .5 * fps))
        evidence.append(f"Detector ball-in-basket class fired (peak {event.peak:.2f}) at frame {event.frame}")
        if outcome == "unknown" and event.peak >= BASKET_CONFIDENCE and motion >= 2.2:
            outcome, confidence, frame = "likely made", .6, event.frame
            evidence.append(f"No rim-plane crossing was visible; ball-in-basket detection with net-specific motion "
                            f"({motion:.1f}× baseline) supports a make")
    return outcome, confidence, evidence, frame


def _jump_type(shot_release: int, c6: list[Run], c7: list[Run], fps: float) -> tuple[str, list[str]]:
    """Floater only when the detector's layup-dunk class matches or beats jump-shot around release."""
    window = round(.4 * fps)
    jump = _overlapping(c6, shot_release - window, shot_release + window)
    rim = _overlapping(c7, shot_release - window, shot_release + window)
    if rim is not None and rim.peak >= .5 and rim.peak >= (jump.peak if jump else 0.):
        return "floater", ["Floater: arc shot with the detector's layup-dunk class at least as confident as jump-shot around release",
                           *_class_note(rim, "layup-dunk"), *_class_note(jump, "jump-shot")]
    return "jump shot", _class_note(jump, "jump-shot")


def _shot(balls, start: int, release: int, end: int, fps: float, outcome, shot_type: str,
          evidence: list[str], attempt: dict) -> ShotResult:
    span = [b for b in balls if start <= b.frame <= end] or [b for b in balls if b.frame == release]
    first, last = (span[0], span[-1]) if span else (None, None)
    outcome_label, confidence, outcome_evidence, outcome_frame = outcome
    return ShotResult(
        number=0,
        start_s=round(first.time_s if first else release / fps, 2),
        release_s=round(release / fps, 2),
        end_s=round(last.time_s if last else release / fps, 2),
        outcome=outcome_label,
        outcome_confidence=round(confidence, 2),
        evidence=[*evidence, *outcome_evidence],
        metrics={},
        cues=[],
        outcome_frame=outcome_frame,
        shot_type=shot_type,
        attempt=attempt,
    )


def missed_before_follow_up(anchors: list[Anchor], fps: float) -> None:
    """Call an unknown attempt missed when a later attempt, let go soon after it
    reached the basket, is seen going through the rim.

    A make ends the possession: the ball goes to the other team under the
    basket it just went through. A tip or putback let go between SAME_ATTEMPT_S
    and TIP_WINDOW_S after the first attempt reached the basket means that
    attempt stayed out (699d, a326, ec07: the tip went in).
    """
    for anchor in anchors:
        shot = anchor.shot
        if shot.outcome != "unknown":
            continue
        reached = anchor.reached if anchor.reached is not None else anchor.release
        follow = next((later for later in sorted(anchors, key=lambda a: a.release)
                       if reached + SAME_ATTEMPT_S * fps <= later.release <= reached + TIP_WINDOW_S * fps
                       and later.shot.outcome == "made" and later.shot.outcome_frame is not None
                       and later.shot.outcome_frame > later.release), None)
        if follow is None:
            continue
        shot.outcome, shot.outcome_confidence, shot.outcome_frame = "missed", .6, follow.release
        shot.evidence.append(f"Another attempt, let go {(follow.release - reached) / fps:.2f} s after this one reached "
                             f"the basket, went through the rim: this one stayed out")


def find_attempts(shots: list[ShotResult], balls: list[Detection], frames: list[dict], fps: float,
                  rim: RimInput, aspect: float, net_motion=None) -> list[ShotResult]:
    """Type the arc shots and add the attempts they miss, sorted by release (one camera segment)."""
    balls = sorted(balls, key=lambda b: b.frame)
    frames = sorted(frames, key=lambda f: f["frame"])
    contacts = hand_contacts(frames, balls, aspect)
    events = basket_events(frames, balls, rim, fps, [c.frame for c in contacts])
    c6, c7, c8 = (event_runs(frames, name, fps) for name in ("jump_shot", "layup_dunk", "shot_block"))
    window = round(SHOT_WINDOW_S * fps)
    by_frame = {b.frame: b for b in balls}
    apexes = arc_apexes(balls, [], fps, game_mode=True)

    anchors = []
    previous_reach = kept_reach = None
    for shot in sorted(shots, key=lambda s: s.release_s):
        release = round(shot.release_s * fps)
        apex = next((f for f in apexes if release <= f <= release + ARC_APEX_S * fps), None)
        # The ball flies free from release to apex. A hand still bending its path
        # later means the release estimate caught the gather or a pump fake.
        late = [c for c in contacts if apex is not None and release < c.frame < apex - .1 * fps
                and not free_flight(balls, c.frame, fps, aspect)]
        if late:
            shot.evidence.append(f"Release moved from frame {release} to the last hand contact before the arc apex, "
                                 f"frame {late[-1].frame}")
            release = late[-1].frame
            shot.release_s = round(release / fps, 2)
        if apex is not None and _seen_rising(balls, release, apex, fps) < ARC_RISE_RADII:
            # The rise rests on stray detections bridged by interpolation or a gap
            # (a net read as the ball after the ball was lost): nobody saw it go up.
            continue
        # From broadcast height a layup off the glass still draws a small arc,
        # so the arc path finds it; its flight time says it was a finish.
        reach = next((e.frame for e in events if release - .2 * fps <= e.frame <= release + FLIGHT_S * fps), None)
        if came_down_away(balls, apex, rim, fps, reach):
            block = _overlapping(c8, release, apex or release)
            if block is None or block.peak < BLOCK_CONFIDENCE:
                # A pass. The ball still reached the basket later (some other
                # attempt), which a tip right after it is measured from.
                previous_reach = reach if reach is not None else previous_reach
                continue
            shot.evidence += ["Blocked: the ball came down away from the rim, with the detector's shot-block class "
                              "firing between release and apex", *_class_note(block, "shot-block")]
        if reach is not None and reach == kept_reach:
            # One attempt reaching the basket once: the previous arc (e.g. a
            # dunk's gather and its slam read as two arcs).
            continue
        kept_reach = reach
        if reach is not None and reach - release <= RIM_FLIGHT_S * fps:
            flight = f"the ball reached the basket {max(0, reach - release) / fps:.2f} s after release"
            if (previous_reach is not None and previous_reach < reach
                    and release - previous_reach <= TIP_AFTER_REACH_S * fps):
                shot.shot_type = "tip"
                notes = [f"Tip: {flight}, {(release - previous_reach) / fps:.2f} s after the previous shot got there"]
            else:
                shot.shot_type = "layup or dunk"
                notes = [f"Rim finish: {flight}; a jump shot or floater takes longer",
                         *_class_note(_overlapping(c7, release - window, reach), "layup-dunk")]
        else:
            shot.shot_type, notes = _jump_type(release, c6, c7, fps)
        shot.evidence += notes
        previous_reach = reach if reach is not None else previous_reach
        anchors.append(Anchor(release, None, shot))

    # Hidden-release jump shots: jump-shot class plus a supported arc, no known shot nearby.
    for run in c6:
        if run.count < 3 or run.peak < .7 or any(run.start - window <= a.release <= run.end + window for a in anchors):
            continue
        apex = next((f for f in apexes if run.start <= f <= run.end + 1.5 * fps), None)
        prior = [c for c in contacts if apex is not None and apex - 1.25 * fps <= c.frame < apex
                 and c.frame >= run.start - .5 * fps]
        if (apex is None or not prior or not _flight_supported(prior[-1], by_frame[apex])
                or _seen_rising(balls, prior[-1].frame, apex, fps) < ARC_RISE_RADII):
            continue
        contact = prior[-1]
        reach = next((e.frame for e in events if contact.frame <= e.frame <= contact.frame + FLIGHT_S * fps), None)
        if came_down_away(balls, apex, rim, fps, reach):
            continue
        handler = _handler_before(frames, contact.frame, fps)
        shooter = contact.track_id if handler in (None, contact.track_id) else None
        shot_type, notes = _jump_type(contact.frame, c6, c7, fps)
        evidence = ["Ball arc detected",
                    f"Release contact hidden: release time is the last visible hand contact (frame {contact.frame}), "
                    f"{(apex - contact.frame) / fps:.2f} s before the arc apex, and may precede the actual release",
                    *notes]
        evidence.append(f"Shooter P{contact.track_id} is the last player in hand contact" +
                        ("" if shooter is not None else f", but the decoded ball handler was P{handler}; shooter withheld"))
        shot = _shot(balls, contact.frame, contact.frame, round(apex + 1.5 * fps), fps,
                     _outcome(balls, contact.frame, None, apex, rim, net_motion, fps), shot_type, evidence,
                     {"path": "hidden_release", "contact_frame": contact.frame, "apex_frame": apex,
                      "shooter_track_id": shooter})
        anchors.append(Anchor(contact.frame, None, shot))
    anchors.sort(key=lambda a: a.release)

    # Rim attempts: a basket event preceded by a new hand contact.
    last_release = -1
    for event in events:
        if event.frame <= last_release:
            continue  # reached while the previous rim attempt's shooter still held the ball
        current = max((a for a in anchors if a.release <= event.frame), key=lambda a: a.release, default=None)
        recent = [c for c in contacts if event.frame - CONTACT_TO_BASKET_S * fps <= c.frame <= event.frame
                  and (current is None or c.frame > current.release + .15 * fps)]
        caught = bool(recent) and sum(c.track_id == recent[-1].track_id for c in recent) >= CAUGHT_CONTACTS
        if (current is not None and current.reached is None and event.frame - current.release <= FLIGHT_S * fps
                and not caught):
            # A shot in flight claims its arrival at the basket. A hand touching it
            # on the way is usually a contest or, in 2D, a background hand the ball
            # passes over (2fcb frame 320, a fan behind the baseline). A ball the
            # same player then held is no longer that shot's (cd04: a false arc
            # claimed Dosunmu's layup after his offensive rebound).
            current.reached = event.frame
            if "ball_in_basket" in event.sources and not any("ball-in-basket" in e for e in current.shot.evidence):
                current.shot.evidence.append(
                    f"Detector ball-in-basket class fired (peak {event.peak:.2f}) at frame {event.frame}; "
                    "it does not change the outcome, which needs the rim")
            continue
        if current is not None and current.reached is not None:
            recent = [c for c in recent if c.frame > current.reached + SETTLE_S * fps]
        if not recent:
            continue
        # No free-flight test here: a ball carried up to the rim, or tipped, moves
        # as smoothly as one flying past a hand. On the dev labels it rejected 23
        # of 32 missed rim attempts and floaters; the pass-overs it was added for
        # are claimed by the shot in flight or dropped as empty-net detections.
        contact = recent[-1]
        if current is not None and current.shot.attempt is not None and _held_since(contacts, contact, current.release, fps):
            # The previous rim attempt's shooter never let go (1b06: the blocked
            # dunker held the ball on the rim, then came down with it).
            continue
        dx, dy = _torso_distance(contact, event.location, aspect)
        at_rim = math.hypot(dx, dy) <= AT_RIM
        path = [b for b in balls if contact.frame <= b.frame <= event.frame]
        if not at_rim:
            rise = contact.ball.y - min(b.y for b in path)
            bounced = any(b.y > contact.ball.y + .5 * contact.torso for b in path)
            if rise < .5 * contact.torso or bounced:
                continue
        # A dunk or a layup carried up enters the rim's area still in the
        # hands; the release is the end of that hold.
        for later in contacts:
            if (contact.frame < later.frame <= event.frame + HOLD_PAST_BASKET_S * fps
                    and later.track_id == contact.track_id and later.frame - contact.frame <= HOLD_GAP_S * fps):
                contact = later
        if contact.frame > event.frame:
            dx, dy = _torso_distance(contact, event.location, aspect)
            at_rim = math.hypot(dx, dy) <= AT_RIM
        last_release = contact.frame
        # An arc found within moments of this contact is the same attempt; the
        # later of the two releases is the one the ball actually left on.
        twin = next((a for a in anchors if a.shot.attempt is None
                     and abs(a.release - contact.frame) <= SAME_ATTEMPT_S * fps), None)
        if twin is not None:
            if twin.release >= contact.frame:
                continue
            anchors.remove(twin)
        previous_reach = max((a.reached for a in anchors if a.reached is not None and a.reached < contact.frame),
                             default=None)
        handler = _handler_before(frames, contact.frame, fps)
        evidence = [f"Rim attempt: last hand contact by P{contact.track_id} at frame {contact.frame}; the ball reached the "
                    f"basket {max(0, event.frame - contact.frame) / fps:.2f} s later ({', '.join(event.sources).replace('_', '-')})"]
        if (previous_reach is not None and contact.frame - previous_reach <= TIP_WINDOW_S * fps
                and at_rim and contact.raised and handler != contact.track_id):
            shot_type = "tip"
            evidence.append(f"Tip: raised-hand touch at the rim {(contact.frame - previous_reach) / fps:.2f} s after the "
                            "previous attempt reached the basket")
        elif at_rim and event.frame - contact.frame <= .35 * fps:
            # In 2D a layup let go at the rim looks like a dunk: on the dev labels the
            # contact's height against the basket did not separate them.
            shot_type = "layup or dunk"
            evidence.append("Layup or dunk: last contact at the basket, reaching it within 0.35 s")
        elif not at_rim and event.frame - contact.frame > RIM_FLIGHT_S * fps:
            # Let go away from the rim with as long a flight as an arc shot's: a
            # floater or jump shot whose arc was not found (f267, 5adf).
            shot_type, notes = _jump_type(contact.frame, c6, c7, fps)
            evidence += [f"{shot_type.capitalize()}: let go away from the rim, reaching it "
                         f"{(event.frame - contact.frame) / fps:.2f} s later", *notes]
        else:
            shot_type = "layup"
            if previous_reach is not None and contact.frame - previous_reach <= TIP_WINDOW_S * fps:
                evidence.append(f"Putback: {(contact.frame - previous_reach) / fps:.2f} s after the previous attempt "
                                "reached the basket")
        shooter = contact.track_id if shot_type == "tip" or handler in (None, contact.track_id) else None
        evidence.append(f"Shooter P{contact.track_id} is the last player in hand contact" +
                        (", matching the decoded ball handler" if handler == contact.track_id else
                         "" if shooter is not None else f", but the decoded ball handler was P{handler}; shooter withheld"))
        evidence += _class_note(_overlapping(c7, contact.frame - window, event.frame), "layup-dunk")
        evidence += _class_note(_overlapping(c8, contact.frame, event.frame + window), "shot-block")
        top = min(path + [b for b in balls if event.frame < b.frame <= max(event.frame, contact.frame) + .3 * fps],
                  key=lambda b: b.y, default=contact.ball)
        shot = _shot(balls, contact.frame - round(.5 * fps), contact.frame, round(event.frame + 1.5 * fps), fps,
                     _outcome(balls, contact.frame, event, top.frame, rim, net_motion, fps), shot_type, evidence,
                     {"path": "rim_attempt", "contact_frame": contact.frame, "basket_frame": event.frame,
                      "basket_sources": event.sources, "shooter_track_id": shooter})
        # The ball-in-basket class can fire on the net while the ball is still in
        # flight (5adf): time the make from the ball first seen at the rim.
        arrived = next((b.frame for b in balls if b.frame >= event.frame and b.confidence >= .45
                        and (box := _rim_at(rim, b.frame)) is not None and _in_rim_area(b, box)), event.frame)
        if shot.outcome in {"made", "likely made"} and shot.outcome_frame - arrived > FOLLOW_UP_S * fps:
            # As late as a tip after it: the make may be an unseen follow-up touch's (f267, a tip left on
            # the rim and tipped in again).
            shot.evidence.append(f"The ball went through {(shot.outcome_frame - arrived) / fps:.2f} s after "
                                 "reaching the basket, as late as a follow-up tip, so the make is not credited")
            shot.outcome, shot.outcome_confidence, shot.outcome_frame = "unknown", 0., None
        anchors.append(Anchor(contact.frame, event.frame, shot))
        anchors.sort(key=lambda a: a.release)

    # Layup-dunk class without a basket event (no rim, ball-in-basket missed).
    for run in c7:
        if (run.count < 2 or run.peak < LAYUP_CLASS_PEAK
                or any(run.start - window <= a.release <= run.end + CONTACT_TO_BASKET_S * fps for a in anchors)
                or any(run.start <= e.frame <= run.end + CONTACT_TO_BASKET_S * fps for e in events)):
            continue
        prior = [c for c in contacts if run.start - .5 * fps <= c.frame <= run.end + .3 * fps]
        if not prior:
            continue
        contact = prior[-1]
        if free_flight(balls, contact.frame, fps, aspect):
            continue
        after = [b for b in balls if contact.frame < b.frame <= contact.frame + fps]
        others = [c for c in contacts if contact.frame < c.frame <= contact.frame + .3 * fps]
        shoulders = sum(contact.pose.landmarks[name][1] for name in ("left_shoulder", "right_shoulder")) / 2
        # The ball leaves the hands upward, above the head, with no catch right after.
        if (not after or min(b.y for b in after) > shoulders - .5 * contact.torso or others
                or came_down_away(balls, min(after, key=lambda b: b.y).frame, rim, fps)):
            continue
        handler = _handler_before(frames, contact.frame, fps)
        shooter = contact.track_id if handler in (None, contact.track_id) else None
        evidence = [f"Layup or dunk: detector layup-dunk class with the ball leaving P{contact.track_id}'s hands "
                    f"upward at frame {contact.frame}; the ball was not seen reaching the basket",
                    *_class_note(run, "layup-dunk"),
                    f"Shooter P{contact.track_id} is the last player in hand contact" +
                    ("" if shooter is not None else f", but the decoded ball handler was P{handler}; shooter withheld")]
        top = min(after, key=lambda b: b.y)
        shot = _shot(balls, contact.frame - round(.5 * fps), contact.frame, round(contact.frame + 1.5 * fps), fps,
                     _outcome(balls, contact.frame, None, top.frame, rim, net_motion, fps), "layup or dunk", evidence,
                     {"path": "layup_dunk_class", "contact_frame": contact.frame, "shooter_track_id": shooter})
        anchors.append(Anchor(contact.frame, None, shot))

    kept = []
    for anchor in sorted(anchors, key=lambda a: a.release):
        # A make ends the possession: the ball drops through the net and has to be
        # inbounded, so a "release" right after it is a hand on the falling ball.
        if any(made.shot.outcome == "made" and made.shot.outcome_frame is not None
               and 0 <= anchor.release - made.shot.outcome_frame <= DEAD_BALL_S * fps for made in kept):
            continue
        kept.append(anchor)
    missed_before_follow_up(kept, fps)
    ordered = sorted((a.shot for a in kept), key=lambda s: s.release_s)
    for number, shot in enumerate(ordered, 1):
        shot.number = number
    return ordered
