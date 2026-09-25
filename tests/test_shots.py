import pytest

from app.models import Detection, PoseFrame, ShotResult
from app.shots import find_attempts
from app.tracking import HandlerDecision

FPS = 30
BASKET = (.5, .1)


def pose(frame, x, track_id, wrist=None):
    """Torso .2 tall (shoulders y .4, hips .6); both wrists at ``wrist`` or low by the hips."""
    wx, wy = wrist or (x, .62)
    landmarks = {"left_shoulder": (x - .03, .4, .95), "right_shoulder": (x + .03, .4, .95),
                 "left_hip": (x - .03, .6, .95), "right_hip": (x + .03, .6, .95),
                 "left_wrist": (wx, wy, .95), "right_wrist": (wx, wy, .95)}
    return PoseFrame(frame, frame / FPS, landmarks, track_id=track_id)


def lerp(a, b, amount):
    return tuple(a[i] + amount * (b[i] - a[i]) for i in (0, 1))


def scene(path, holders, events=(), handlers=None, n=None):
    """path: frame -> ball (x, y); holders: frame -> (track_id, wrist) for a hand on the ball.

    P1 stands at x .4 and P2 at x .7; a holder's wrists are moved onto ``wrist``.
    """
    frames, balls = [], []
    for frame in range(n or max(path) + 1):
        players = []
        for track_id, x in ((1, .4), (2, .7)):
            holder = holders.get(frame)
            players.append(pose(frame, x, track_id, holder[1] if holder and holder[0] == track_id else None))
        handler = (handlers or {}).get(frame)
        frames.append({"frame": frame, "time_s": frame / FPS, "players": players, "unposed": [],
                       "handler": HandlerDecision(handler, "observed") if handler else None,
                       "events": [event[1:] for event in events if event[0] == frame]})
        if frame in path:
            balls.append(Detection(frame, frame / FPS, *path[frame], .9))
    return balls, frames


def basket_box(conf):
    return ("ball_in_basket", conf, (BASKET[0] - .02, BASKET[1] - .02, BASKET[0] + .02, BASKET[1] + .02))


def layup(handler=1, rim_conf=.8):
    """P1 holds the ball at (.42, .35) to frame 10; it rises to the basket by frame 20."""
    path = {f: (.42, .35) for f in range(11)}
    path.update({f: lerp((.42, .35), BASKET, (f - 10) / 10) for f in range(11, 21)})
    path.update({f: lerp(BASKET, (.5, .3), (f - 20) / 10) for f in range(21, 31)})
    holders = {f: (1, (.42, .35)) for f in range(11)}
    events = [(20, *basket_box(rim_conf)), (21, *basket_box(rim_conf))]
    return scene(path, holders, events, {f: handler for f in range(11)})


def test_layup_reaching_the_basket_is_a_shot_by_the_last_hand_contact():
    balls, frames = layup()
    shots = find_attempts([], balls, frames, FPS, None, 1.)
    assert len(shots) == 1
    shot = shots[0]
    assert shot.shot_type == "layup"
    assert shot.release_s == round(10 / FPS, 2)
    assert shot.attempt["path"] == "rim_attempt" and shot.attempt["shooter_track_id"] == 1
    assert shot.outcome == "unknown"
    assert any("not treated as a make without a rim" in item for item in shot.evidence)
    assert any("matching the decoded ball handler" in item for item in shot.evidence)


def test_handler_disagreement_withholds_the_shooter():
    balls, frames = layup(handler=2)
    shot, = find_attempts([], balls, frames, FPS, None, 1.)
    assert shot.attempt["shooter_track_id"] is None
    assert any("shooter withheld" in item for item in shot.evidence)


def test_lone_low_confidence_basket_detection_is_ignored():
    balls, frames = scene({f: (.42, .35) for f in range(30)}, {f: (1, (.42, .35)) for f in range(30)},
                          [(20, *basket_box(.4))])
    assert find_attempts([], balls, frames, FPS, None, 1.) == []


def test_dribble_under_the_basket_is_not_a_shot():
    # The ball goes to the floor and back up to the hand; a confident
    # ball-in-basket detection (e.g. of another ball) follows.
    path = {f: (.42, .5) for f in range(11)}
    path.update({f: lerp((.42, .5), (.43, .95), (f - 10) / 5) for f in range(11, 16)})
    path.update({f: lerp((.43, .95), (.42, .5), (f - 15) / 5) for f in range(16, 21)})
    holders = {f: (1, (.42, .5)) for f in [*range(11), 20]}
    balls, frames = scene(path, holders, [(21, *basket_box(.8)), (22, *basket_box(.8))], n=25)
    assert find_attempts([], balls, frames, FPS, None, 1.) == []


def test_dunk_is_a_contact_at_the_basket():
    path = {f: lerp((.45, .35), (.5, .12), f / 18) for f in range(19)}
    path.update({f: (.5, .12 + .02 * (f - 18)) for f in range(19, 30)})
    holders = {f: (1, path[f]) for f in range(19)}
    balls, frames = scene(path, holders, [(20, *basket_box(.8))], {f: 1 for f in range(19)})
    shot, = find_attempts([], balls, frames, FPS, None, 1.)
    assert shot.shot_type == "dunk"
    assert shot.attempt["contact_frame"] == 18


def arc_shot():
    """A jump shot released at frame 0 that reaches the basket at frame 20, then falls."""
    return ShotResult(1, 0., 0., 1., "unknown", 0., ["Ball arc detected"], {})


def test_rim_rattle_and_rebound_belong_to_the_jump_shot():
    path = {f: lerp((.3, .3), BASKET, f / 20) for f in range(21)}
    path.update({f: (.5, .1) for f in range(21, 40)})
    path.update({f: lerp(BASKET, (.7, .45), (f - 40) / 10) for f in range(40, 51)})
    path.update({f: (.7, .45) for f in range(51, 70)})
    # P2 catches the rebound low; ball-in-basket fires twice as the ball sits on the rim.
    holders = {f: (2, (.7, .45)) for f in range(50, 70)}
    events = [(20, *basket_box(.8)), (38, *basket_box(.9))]
    balls, frames = scene(path, holders, events, {f: 2 for f in range(52, 70)})
    shots = find_attempts([arc_shot()], balls, frames, FPS, None, 1.)
    assert len(shots) == 1 and shots[0].shot_type == "jump shot"
    assert any("does not change the outcome" in item for item in shots[0].evidence)


def test_tip_after_the_ball_reaches_the_basket():
    path = {f: lerp((.3, .3), BASKET, f / 20) for f in range(21)}
    path.update({f: lerp(BASKET, (.55, .14), (f - 20) / 6) for f in range(21, 27)})
    path.update({f: lerp((.55, .14), BASKET, (f - 26) / 4) for f in range(27, 40)})
    holders = {26: (2, (.55, .14))}
    events = [(20, *basket_box(.8)), (30, *basket_box(.8))]
    balls, frames = scene(path, holders, events, n=40)
    shots = find_attempts([arc_shot()], balls, frames, FPS, None, 1.)
    assert [shot.shot_type for shot in shots] == ["jump shot", "tip"]
    assert shots[1].attempt["shooter_track_id"] == 2 and shots[1].number == 2


def test_hand_over_the_ball_near_the_rim_in_flight_is_not_a_new_attempt():
    # 2fcb: the falling jump shot passed over a raised hand behind the baseline
    # just before reaching the rim, which read as a dunk.
    path = {f: lerp((.3, .3), BASKET, f / 20) for f in range(25)}
    holders = {f: (2, path[f]) for f in (17, 18)}
    balls, frames = scene(path, holders, [(20, *basket_box(.8)), (21, *basket_box(.8))], {17: 2, 18: 2})
    shots = find_attempts([arc_shot()], balls, frames, FPS, None, 1.)
    assert [shot.shot_type for shot in shots] == ["jump shot"]


def test_contest_mid_flight_is_not_a_new_attempt():
    far_basket = (.8, .1)
    path = {f: lerp((.3, .3), far_basket, f / 20) for f in range(25)}
    holders = {8: (2, path[8])}
    box = ("ball_in_basket", .8, (.78, .08, .82, .12))
    balls, frames = scene(path, holders, [(20, *box), (21, *box)])
    assert len(find_attempts([arc_shot()], balls, frames, FPS, None, 1.)) == 1


def test_layup_dunk_class_needs_the_ball_to_leave_upward():
    holders = {f: (1, (.42, .45)) for f in range(11)}
    events = [(f, "layup_dunk", .7, (.35, .3, .45, .7)) for f in range(5, 11)]
    # Pass: the ball goes across to P2's hands.
    path = {f: (.42, .45) for f in range(11)}
    path.update({f: lerp((.42, .45), (.7, .45), (f - 10) / 6) for f in range(11, 30)})
    pass_holders = {**holders, **{f: (2, (.7, .45)) for f in range(16, 30)}}
    balls, frames = scene(path, pass_holders, events)
    assert find_attempts([], balls, frames, FPS, None, 1.) == []
    # Finish: the ball rises above P1's head and nobody catches it.
    path.update({f: lerp((.42, .45), (.45, .05), (f - 10) / 10) for f in range(11, 30)})
    balls, frames = scene(path, holders, events)
    shot, = find_attempts([], balls, frames, FPS, None, 1.)
    assert shot.shot_type == "layup or dunk" and shot.attempt["path"] == "layup_dunk_class"


def test_hidden_release_jump_shot_uses_the_jump_shot_class_and_arc():
    path = {f: (.4, .45) for f in range(6)}
    path.update({f: lerp((.4, .45), (.5, .05), (f - 5) / 15) for f in range(6, 21)})
    path.update({f: lerp((.5, .05), (.6, .3), (f - 20) / 15) for f in range(21, 36)})
    holders = {f: (1, (.4, .45)) for f in range(6)}
    events = [(f, "jump_shot", .9, (.35, .3, .45, .7)) for f in range(9)]
    balls, frames = scene(path, holders, events, {f: 1 for f in range(6)})
    shot, = find_attempts([], balls, frames, FPS, None, 1.)
    assert shot.shot_type == "jump shot"
    assert shot.attempt == {"path": "hidden_release", "contact_frame": 5, "apex_frame": 20, "shooter_track_id": 1}
    assert any(item.startswith("Release contact hidden") for item in shot.evidence)
    # Without the detector's jump-shot class the same arc is left to the raised-hand path.
    balls, frames = scene(path, holders, [], {f: 1 for f in range(6)})
    assert find_attempts([], balls, frames, FPS, None, 1.) == []


@pytest.mark.parametrize("layup_conf, expected", [(.7, "floater"), (.5, "jump shot")])
def test_floater_needs_layup_dunk_class_at_least_as_confident_as_jump_shot(layup_conf, expected):
    events = ([(f, "layup_dunk", layup_conf, (.35, .3, .45, .7)) for f in range(5, 13)]
              + [(f, "jump_shot", .6, (.35, .3, .45, .7)) for f in range(5, 13)])
    balls, frames = scene({f: (.9, .9) for f in range(20)}, {}, events)
    shot = ShotResult(1, 0., 10 / FPS, 1., "unknown", 0., [], {})
    assert find_attempts([shot], balls, frames, FPS, None, 1.)[0].shot_type == expected


def test_layup_outcome_uses_the_rim_when_one_exists():
    path = {f: (.42, .35) for f in range(11)}
    path.update({f: lerp((.42, .35), (.5, .05), (f - 10) / 10) for f in range(11, 21)})
    path.update({f: lerp((.5, .05), (.5, .2), (f - 20) / 6) for f in range(21, 35)})
    holders = {f: (1, (.42, .35)) for f in range(11)}
    balls, frames = scene(path, holders, handlers={f: 1 for f in range(11)})
    shot, = find_attempts([], balls, frames, FPS, (.47, .08, .06, .04), 1.)
    assert shot.shot_type == "layup" and shot.attempt["basket_sources"] == ["rim_area"]
    assert shot.outcome == "made"


def test_putback_after_a_rebound_is_a_new_layup():
    path = {f: lerp((.3, .3), BASKET, f / 20) for f in range(21)}
    path.update({f: lerp(BASKET, (.66, .35), (f - 20) / 10) for f in range(21, 31)})
    path.update({f: (.66, .35) for f in range(31, 41)})
    path.update({f: lerp((.66, .35), BASKET, (f - 40) / 10) for f in range(41, 55)})
    holders = {f: (2, (.66, .35)) for f in range(30, 41)}
    events = [(20, *basket_box(.8)), (50, *basket_box(.8))]
    balls, frames = scene(path, holders, events, {f: 2 for f in range(32, 41)})
    shots = find_attempts([arc_shot()], balls, frames, FPS, None, 1.)
    assert [shot.shot_type for shot in shots] == ["jump shot", "layup"]
    assert shots[1].attempt["shooter_track_id"] == 2
    assert any(item.startswith("Putback") for item in shots[1].evidence)


def test_ball_passing_a_hand_on_a_parabola_is_not_a_touch():
    # After the jump shot reaches the basket, the ball drops past a raised hand
    # behind the rim without changing course (2fcb frame 326).
    path = {f: lerp((.3, .3), BASKET, f / 20) for f in range(21)}
    path.update({f: (.5 + .004 * (f - 20), .1 + .003 * (f - 20) + .0004 * (f - 20) ** 2) for f in range(21, 45)})
    holders = {f: (2, path[f]) for f in (28, 29)}
    events = [(20, *basket_box(.8)), (30, *basket_box(.8)), (31, *basket_box(.8))]
    balls, frames = scene(path, holders, events)
    assert [shot.shot_type for shot in find_attempts([arc_shot()], balls, frames, FPS, None, 1.)] == ["jump shot"]


def test_rim_area_entry_survives_a_low_confidence_frame():
    path = {f: lerp((.42, .35), (.5, .05), f / 10) for f in range(11)}
    path.update({f: lerp((.5, .05), (.5, .2), (f - 10) / 8) for f in range(11, 25)})
    balls, frames = scene(path, {})
    balls[12].confidence = .3
    from app.shots import basket_events
    # A hand contact between the two frames (2fcb) would stop the entries merging.
    assert len(basket_events(frames, balls, (.47, .08, .06, .04), FPS, [12])) == 1
