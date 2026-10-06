import pytest

from app.models import Detection, PoseFrame, ShotResult
from app.shots import Anchor, find_attempts, missed_before_follow_up
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


def test_contact_at_the_basket_is_a_layup_or_dunk():
    # In 2D a dunk and a layup let go at the rim look alike.
    path = {f: lerp((.45, .35), (.5, .12), f / 18) for f in range(19)}
    path.update({f: (.5, .12 + .02 * (f - 18)) for f in range(19, 30)})
    holders = {f: (1, path[f]) for f in range(19)}
    balls, frames = scene(path, holders, [(20, *basket_box(.8))], {f: 1 for f in range(19)})
    shot, = find_attempts([], balls, frames, FPS, None, 1.)
    assert shot.shot_type == "layup or dunk"
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


def arc_shots_reaching(releases_and_reaches):
    """Arc shots released at the given frames, each reaching the basket at its paired frame."""
    path = {f: (.45, .3) for f in range(max(r for _, r in releases_and_reaches) + 10)}
    path.update({reach: BASKET for _, reach in releases_and_reaches})
    events = [(reach, *basket_box(.8)) for _, reach in releases_and_reaches]
    balls, frames = scene(path, {}, events)
    shots = [ShotResult(i + 1, 0., release / FPS, 1., "unknown", 0., ["Ball arc detected"], {})
             for i, (release, _) in enumerate(releases_and_reaches)]
    return find_attempts(shots, balls, frames, FPS, None, 1.)


def test_arc_shot_reaching_the_basket_right_after_release_is_a_rim_finish():
    shot, = arc_shots_reaching([(10, 16)])
    assert shot.shot_type == "layup or dunk"
    assert any(item.startswith("Rim finish: the ball reached the basket 0.20 s") for item in shot.evidence)


def test_quick_rim_finish_right_after_the_previous_shot_got_there_is_a_tip():
    first, tip = arc_shots_reaching([(0, 20), (40, 44)])
    assert (first.shot_type, tip.shot_type) == ("jump shot", "tip")
    first, putback = arc_shots_reaching([(0, 20), (60, 64)])
    assert putback.shot_type == "layup or dunk"


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


RIM = (.47, .08, .06, .04)


def arc(apex, end):
    """An arc released at frame 0 from (.2, .5), peaking at ``apex`` on frame 15 and ending at ``end`` on frame 30."""
    path = {f: lerp((.2, .5), apex, f / 15) for f in range(16)}
    path.update({f: lerp(apex, end, (f - 15) / 15) for f in range(16, 31)})
    return path


def test_pass_coming_down_beside_the_rim_is_not_a_shot():
    # The ball peaks at rim height and comes down in a teammate's hands, far from the basket.
    balls, frames = scene(arc((.35, .2), (.5, .5)), {})
    assert find_attempts([arc_shot()], balls, frames, FPS, RIM, 1.) == []
    # Without a rim nothing says where the basket is, so the arc stays a shot.
    assert len(find_attempts([arc_shot()], balls, frames, FPS, None, 1.)) == 1
    # A shot gets to the basket before it comes down.
    balls, frames = scene(arc((.42, .02), (.5, .12)), {})
    shot, = find_attempts([arc_shot()], balls, frames, FPS, RIM, 1.)
    assert shot.shot_type == "jump shot"


def test_ball_in_basket_with_the_ball_seen_elsewhere_is_ignored():
    # 5adf: the empty net fired while the tracked ball was mid-pass across the court.
    from app.shots import basket_events
    balls, frames = scene({f: (.2, .4) for f in range(10)}, {}, [(5, *basket_box(.9)), (6, *basket_box(.9))])
    assert basket_events(frames, balls, None, FPS) == []
    balls, frames = scene({f: BASKET if f in (5, 6) else (.2, .4) for f in range(10)}, {},
                          [(5, *basket_box(.9)), (6, *basket_box(.9))])
    assert len(basket_events(frames, balls, None, FPS)) == 1


def test_hand_on_the_ball_after_the_release_estimate_moves_the_release():
    # The arc path timed the release at the gather (frame 0); P1 held the ball overhead to frame 8.
    path = {f: (.4, .3) for f in range(9)}
    path.update({f: lerp((.4, .3), (.45, .05), (f - 8) / 12) for f in range(9, 21)})
    path.update({f: lerp((.45, .05), BASKET, (f - 20) / 8) for f in range(21, 35)})
    holders = {f: (1, (.4, .3)) for f in range(9)}
    balls, frames = scene(path, holders, [(28, *basket_box(.8)), (29, *basket_box(.8))])
    shot, = find_attempts([arc_shot()], balls, frames, FPS, None, 1.)
    assert shot.release_s == round(8 / FPS, 2) and shot.shot_type == "jump shot"
    assert any(item.startswith("Release moved from frame 0") for item in shot.evidence)


def test_blocked_shot_coming_down_away_stays_a_shot():
    events = [(f, "shot_block", .9, (.3, .1, .4, .3)) for f in range(5, 12)]
    balls, frames = scene(arc((.35, .2), (.5, .5)), {}, events)
    shot, = find_attempts([arc_shot()], balls, frames, FPS, RIM, 1.)
    assert any(item.startswith("Blocked") for item in shot.evidence)


def test_two_arcs_reaching_the_basket_at_the_same_moment_are_one_attempt():
    # 8628: a tip-dunk's catch above the rim and its slam read as two arcs.
    shot, = arc_shots_reaching([(10, 16), (18, 16)])
    assert shot.release_s == pytest.approx(10 / FPS)


def test_ball_carried_into_the_rim_area_is_released_at_the_end_of_the_hold():
    # 1b06, a80c: the ball enters the rim's area still in the hands and moves as
    # smoothly as a ball in flight; the release is the last contact of the hold.
    path = {f: lerp((.45, .4), (.5, .06), f / 24) for f in range(25)}
    path.update({f: lerp((.5, .06), (.5, .3), (f - 24) / 12) for f in range(25, 40)})
    holders = {f: (1, path[f]) for f in range(25)}
    balls, frames = scene(path, holders, handlers={f: 1 for f in range(25)})
    shot, = find_attempts([], balls, frames, FPS, RIM, 1.)
    assert shot.attempt["path"] == "rim_attempt" and shot.attempt["contact_frame"] == 24
    assert shot.attempt["basket_frame"] < 24 and shot.shot_type == "layup or dunk"


def test_dunk_through_the_rim_before_the_hold_ends_is_made():
    # 8c88, a80c: the hands are still on the ball as it goes down through the rim.
    path = {f: lerp((.45, .4), (.5, .06), f / 24) for f in range(25)}
    path.update({f: lerp((.5, .06), (.5, .14), (f - 24) / 4) for f in range(25, 29)})
    path.update({f: lerp((.5, .14), (.5, .4), (f - 28) / 10) for f in range(29, 45)})
    holders = {f: (1, path[f]) for f in range(29)}
    balls, frames = scene(path, holders, handlers={f: 1 for f in range(29)})
    shot, = find_attempts([], balls, frames, FPS, RIM, 1.)
    assert shot.attempt["contact_frame"] == 28 and shot.outcome == "made"
    assert shot.outcome_frame < 28


def test_rim_attempt_moments_after_an_arc_replaces_it():
    # b212: the arc path caught the gather; the same hands finished at the rim 0.4 s later.
    path = {f: lerp((.4, .3), BASKET, f / 6) for f in range(7)}
    path.update({f: (.5, .15) for f in range(7, 13)})
    path.update({f: lerp((.5, .15), BASKET, (f - 12) / 3) for f in range(13, 25)})
    holders = {f: (1, (.5, .15)) for f in range(8, 13)}
    events = [(6, *basket_box(.8)), (15, *basket_box(.8)), (16, *basket_box(.8))]
    balls, frames = scene(path, holders, events, {f: 1 for f in range(8, 13)})
    shot, = find_attempts([arc_shot()], balls, frames, FPS, None, 1.)
    assert shot.attempt["path"] == "rim_attempt" and shot.release_s == round(12 / FPS, 2)


def test_rim_attempt_make_long_after_reaching_the_basket_is_not_credited():
    # f267: a tip left on the rim was tipped in again by an unseen touch.
    path = {f: (.5, .35) for f in range(11)}
    path.update({f: (.5, .01 + .34 / 900 * (f - 40) ** 2) for f in range(11, 65)})
    holders = {f: (1, (.5, .35)) for f in range(11)}
    balls, frames = scene(path, holders, handlers={f: 1 for f in range(11)})
    shot, = find_attempts([], balls, frames, FPS, RIM, 1.)
    assert shot.outcome == "unknown"
    assert any("so the make is not credited" in item for item in shot.evidence)


def test_make_is_timed_from_the_ball_reaching_the_rim_not_an_early_basket_detection():
    # 5adf: the ball-in-basket class fired on the net while the ball was still in flight.
    path = {f: (.3, .35) for f in range(11)}
    path.update({f: lerp((.3, .35), (.5, -.3), (f - 10) / 15) for f in range(11, 26)})
    path.update({f: lerp((.5, -.3), (.5, .15), (f - 25) / 24) for f in range(26, 50)})
    path.update({f: lerp((.5, .15), (.5, .35), (f - 49) / 10) for f in range(50, 60)})
    holders = {f: (1, (.3, .35)) for f in range(11)}
    events = [(f, *basket_box(.9)) for f in range(19, 50, 5)]
    balls, frames = scene(path, holders, events, {f: 1 for f in range(11)})
    shot, = find_attempts([], balls, frames, FPS, RIM, 1.)
    assert shot.attempt["basket_frame"] == 19
    assert shot.outcome == "made"


def test_ball_caught_after_a_shot_ends_its_flight():
    # cd04: a false arc claimed the layup of the player who rebounded it.
    path = {f: lerp((.3, .3), (.55, .25), f / 10) for f in range(11)}
    path.update({f: (.55, .25) for f in range(11, 17)})
    path.update({f: lerp((.55, .25), BASKET, (f - 16) / 4) for f in range(17, 30)})
    holders = {f: (2, (.55, .25)) for f in range(10, 17)}
    balls, frames = scene(path, holders, [(20, *basket_box(.8)), (21, *basket_box(.8))], {f: 2 for f in range(10, 17)})
    shots = find_attempts([arc_shot()], balls, frames, FPS, None, 1.)
    assert [shot.attempt and shot.attempt["path"] for shot in shots] == [None, "rim_attempt"]
    assert shots[1].release_s == round(16 / FPS, 2)


@pytest.mark.parametrize("wrist, found", [((.55, .30), True), ((.55, -.02), False)])
def test_tip_on_the_fingertips_above_a_raised_hand(wrist, found):
    # 3b4f, 699d: the tipper's wrist is a hand's length below the ball. The
    # same distance with the ball below the hand is not a touch.
    path = {f: lerp((.3, .3), BASKET, f / 20) for f in range(21)}
    path.update({f: lerp(BASKET, (.55, .14), (f - 20) / 6) for f in range(21, 27)})
    path.update({f: lerp((.55, .14), BASKET, (f - 26) / 4) for f in range(27, 40)})
    events = [(20, *basket_box(.8)), (30, *basket_box(.8))]
    balls, frames = scene(path, {26: (2, wrist)}, events, n=40)
    shots = find_attempts([arc_shot()], balls, frames, FPS, None, 1.)
    assert [shot.shot_type for shot in shots] == (["jump shot", "tip"] if found else ["jump shot"])


def test_rim_attempt_let_go_away_from_the_rim_with_a_long_flight_is_a_shot():
    # f267, 5adf: an arc the arc path missed, found from the contact and the basket.
    path = {f: (.42, .35) for f in range(11)}
    path.update({f: lerp((.42, .35), BASKET, (f - 10) / 15) for f in range(11, 26)})
    path.update({f: lerp(BASKET, (.5, .3), (f - 25) / 10) for f in range(26, 36)})
    holders = {f: (1, (.42, .35)) for f in range(11)}
    balls, frames = scene(path, holders, [(25, *basket_box(.8)), (26, *basket_box(.8))], {f: 1 for f in range(11)})
    shot, = find_attempts([], balls, frames, FPS, None, 1.)
    assert shot.attempt["path"] == "rim_attempt" and shot.shot_type == "jump shot"


def attempt(release, reached, outcome, outcome_frame=None):
    shot = ShotResult(1, release / FPS, release / FPS, release / FPS + 1, outcome, 0., [], {},
                      outcome_frame=outcome_frame)
    return Anchor(release, reached, shot)


@pytest.mark.parametrize("tip_release, outcome", [
    (36, "missed"),   # 699d, a326, ec07: a tip 0.53 s after it reached the basket went in
    (22, "unknown"),  # too soon after: the same attempt seen twice
    (110, "unknown"),  # long after: a new possession
])
def test_attempt_followed_by_a_made_tip_missed(tip_release, outcome):
    first, tip = attempt(10, 20, "unknown"), attempt(tip_release, None, "made", tip_release + 5)
    missed_before_follow_up([first, tip], FPS)
    assert first.shot.outcome == outcome
    assert tip.shot.outcome == "made"


def test_follow_up_that_did_not_go_in_says_nothing():
    first, tip = attempt(10, 20, "unknown"), attempt(36, None, "missed", 45)
    missed_before_follow_up([first, tip], FPS)
    assert first.shot.outcome == "unknown"


def test_arc_whose_rise_was_not_seen_is_not_a_shot():
    # cd04: the ball was lost after a rebound; one stray detection near the rim,
    # bridged by interpolation, drew an arc nobody threw.
    balls, frames = scene(arc((.42, .02), (.5, .12)), {})
    for ball in balls:
        if 1 <= ball.frame < 15:
            ball.confidence = .3
    assert find_attempts([arc_shot()], balls, frames, FPS, RIM, 1.) == []
    balls, frames = scene(arc((.42, .02), (.5, .12)), {})
    assert len(find_attempts([arc_shot()], balls, frames, FPS, RIM, 1.)) == 1


def test_hand_on_the_ball_falling_through_the_net_is_not_a_new_attempt():
    # 25e9: after the dunk the ball dropped through the net into a player's hands,
    # and a stray detection at the rim read as a putback.
    made = ShotResult(1, 0., 0., 1., "made", .9, ["Ball arc detected"], {}, outcome_frame=20)
    path = {f: lerp((.3, .3), BASKET, f / 20) for f in range(21)}
    path.update({f: lerp(BASKET, (.5, .4), (f - 20) / 8) for f in range(21, 29)})
    path.update({f: lerp((.5, .4), BASKET, (f - 28) / 3) for f in range(29, 40)})
    holders = {f: (2, (.5, .4)) for f in range(26, 29)}
    events = [(20, *basket_box(.8)), (31, *basket_box(.8))]
    balls, frames = scene(path, holders, events, n=40)
    assert [shot.number for shot in find_attempts([made], balls, frames, FPS, None, 1.)] == [1]
    made = ShotResult(1, 0., 0., 1., "missed", .9, ["Ball arc detected"], {}, outcome_frame=20)
    assert len(find_attempts([made], balls, frames, FPS, None, 1.)) == 2
