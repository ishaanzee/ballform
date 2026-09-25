"""Game-mode shot coverage beyond the raised-hand jump-shot arc.

``analyze_shots`` finds jump shots from a ball arc plus visible raised-hand
release contact. This module labels every game shot with a type and adds the
attempts that path misses:

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
from app.scoring import RimInput, _rim_at, arc_apexes, rim_outcome
from app.tracking import body_geometry

# Wrist-to-ball distance in torso lengths, as game.py's possession votes; a
# second hand within the margin makes the contact ambiguous.
CONTACT_REACH = .65
CONTACT_MARGIN = .2
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
# A touch bends the ball's path. In 2D a ball flying past a hand, e.g. a fan's
# behind the baseline, stays on a parabola: the largest residual around the
# contact was 0.022 there versus 0.050-0.065 at three real releases (units of
# frame height; ball radius about 0.016).
FREE_FLIGHT_RADII = 2.5
RIM_TYPES = {"layup", "dunk", "layup or dunk", "tip"}


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
        if not ranked or ranked[0][0] > CONTACT_REACH:
            continue
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < CONTACT_MARGIN:
            continue
        # A detector player without a pose holding the ball is a hidden rival.
        if any(_unposed_reach(ball, box) < math.inf for _, box in frame.get("unposed", [])):
            continue
        _, pose, torso, wrist = ranked[0]
        shoulders = [p for name in ("left_shoulder", "right_shoulder")
                     if (p := pose.landmarks.get(name)) is not None and p[2] >= .5]
        if pose.track_id is None or len(shoulders) < 2:
            continue
        raised = wrist[1] <= (shoulders[0][1] + shoulders[1][1]) / 2 + .15 * torso
        contacts.append(Contact(frame["frame"], pose.track_id, pose, torso, ball, raised))
    return contacts


def basket_events(frames: list[dict], balls: list[Detection], rim: RimInput, fps: float,
                  contact_frames: list[int] = ()) -> list[BasketEvent]:
    """Moments the ball reaches the basket: ball-in-basket detections and rim-area entries.

    Neither means a make: lone ball-in-basket detections also fire on an empty
    net, and a ball in the rim's area can still miss. Detections within
    0.5 s merge into one event unless a hand touched the ball in between (a tip).
    """
    events = []
    for run in event_runs(frames, "ball_in_basket", fps):
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
            inside = (box[0] - .25 * box[2] <= ball.x <= box[0] + 1.25 * box[2]
                      and box[1] - 1.5 * box[3] <= ball.y <= box[1] + 1.2 * box[3])
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
    segment = [b for b in balls if start <= b.frame <= apex_frame + 1.5 * fps]
    if _rim_at(rim, apex_frame) is None or not segment:
        evidence = ["Outcome unavailable because the rim was not marked"]
        if event is not None and "ball_in_basket" in event.sources:
            evidence.append(f"The detector's ball-in-basket class fired (peak {event.peak:.2f}) at frame {event.frame}; "
                            "lone detections also fire on an empty net, so it is not treated as a make without a rim.")
        return "unknown", 0., evidence, None
    outcome, confidence, evidence, frame = rim_outcome(segment, apex_frame, rim, net_motion, fps)
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


def find_attempts(shots: list[ShotResult], balls: list[Detection], frames: list[dict], fps: float,
                  rim: RimInput, aspect: float, net_motion=None) -> list[ShotResult]:
    """Type the arc shots and add the attempts they miss, sorted by release (one camera segment)."""
    balls = sorted(balls, key=lambda b: b.frame)
    frames = sorted(frames, key=lambda f: f["frame"])
    contacts = hand_contacts(frames, balls, aspect)
    events = basket_events(frames, balls, rim, fps, [c.frame for c in contacts])
    c6, c7, c8 = (event_runs(frames, name, fps) for name in ("jump_shot", "layup_dunk", "shot_block"))
    window = round(SHOT_WINDOW_S * fps)

    anchors = []
    for shot in shots:
        release = round(shot.release_s * fps)
        shot.shot_type, notes = _jump_type(release, c6, c7, fps)
        shot.evidence += notes
        anchors.append(Anchor(release, None, shot))

    # Hidden-release jump shots: jump-shot class plus a supported arc, no known shot nearby.
    by_frame = {b.frame: b for b in balls}
    apexes = arc_apexes(balls, [], fps, game_mode=True)
    for run in c6:
        if run.count < 3 or run.peak < .7 or any(run.start - window <= a.release <= run.end + window for a in anchors):
            continue
        apex = next((f for f in apexes if run.start <= f <= run.end + 1.5 * fps), None)
        prior = [c for c in contacts if apex is not None and apex - 1.25 * fps <= c.frame < apex
                 and c.frame >= run.start - .5 * fps]
        if apex is None or not prior or not _flight_supported(prior[-1], by_frame[apex]):
            continue
        contact = prior[-1]
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
    for event in events:
        current = max((a for a in anchors if a.release <= event.frame), key=lambda a: a.release, default=None)
        recent = [c for c in contacts if event.frame - CONTACT_TO_BASKET_S * fps <= c.frame <= event.frame
                  and (current is None or c.frame > current.release + .15 * fps)]
        if current is not None and current.reached is None and event.frame - current.release <= FLIGHT_S * fps:
            # A shot in flight claims its arrival at the basket. A hand touching it
            # on the way is usually a contest or, in 2D, a background hand the ball
            # passes over (2fcb frame 320, a fan behind the baseline).
            current.reached = event.frame
            if "ball_in_basket" in event.sources and not any("ball-in-basket" in e for e in current.shot.evidence):
                current.shot.evidence.append(
                    f"Detector ball-in-basket class fired (peak {event.peak:.2f}) at frame {event.frame}; "
                    "it does not change the outcome, which needs the rim")
            continue
        if current is not None and current.reached is not None:
            recent = [c for c in recent if c.frame > current.reached]
        if not recent:
            continue
        contact = recent[-1]
        if free_flight(balls, contact.frame, fps, aspect):
            continue
        dx, dy = _torso_distance(contact, event.location, aspect)
        at_rim = math.hypot(dx, dy) <= AT_RIM
        path = [b for b in balls if contact.frame <= b.frame <= event.frame]
        if not at_rim:
            rise = contact.ball.y - min(b.y for b in path)
            bounced = any(b.y > contact.ball.y + .5 * contact.torso for b in path)
            if rise < .5 * contact.torso or bounced:
                continue
        previous_reach = max((a.reached for a in anchors if a.reached is not None and a.reached < contact.frame),
                             default=None)
        handler = _handler_before(frames, contact.frame, fps)
        evidence = [f"Rim attempt: last hand contact by P{contact.track_id} at frame {contact.frame}; the ball reached the "
                    f"basket {(event.frame - contact.frame) / fps:.2f} s later ({', '.join(event.sources).replace('_', '-')})"]
        if (previous_reach is not None and contact.frame - previous_reach <= TIP_WINDOW_S * fps
                and at_rim and contact.raised and handler != contact.track_id):
            shot_type = "tip"
            evidence.append(f"Tip: raised-hand touch at the rim {(contact.frame - previous_reach) / fps:.2f} s after the "
                            "previous attempt reached the basket")
        elif at_rim and dy <= .3 and event.frame - contact.frame <= .35 * fps:
            shot_type = "dunk"
            evidence.append("Dunk: last contact at or above the basket, reaching it within 0.35 s")
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
        top = min(path + [b for b in balls if event.frame < b.frame <= event.frame + .3 * fps], key=lambda b: b.y)
        shot = _shot(balls, contact.frame - round(.5 * fps), contact.frame, round(event.frame + 1.5 * fps), fps,
                     _outcome(balls, contact.frame, event, top.frame, rim, net_motion, fps), shot_type, evidence,
                     {"path": "rim_attempt", "contact_frame": contact.frame, "basket_frame": event.frame,
                      "basket_sources": event.sources, "shooter_track_id": shooter})
        anchors.append(Anchor(contact.frame, event.frame, shot))
        anchors.sort(key=lambda a: a.release)

    # Layup-dunk class without a basket event (no rim, ball-in-basket missed).
    for run in c7:
        if (run.count < 2 or run.peak < .5
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
        if not after or min(b.y for b in after) > shoulders - .5 * contact.torso or others:
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

    ordered = sorted((a.shot for a in anchors), key=lambda s: s.release_s)
    for number, shot in enumerate(ordered, 1):
        shot.number = number
    return ordered
