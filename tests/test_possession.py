from app.models import Detection, PoseFrame
from app.possession import decode_handlers


def player(track_id, x, frame=0):
    """Upright player with hips at (x, .6), torso .1 tall, wrists at hip height."""
    landmarks = {"left_shoulder": (x - .02, .5, .9), "right_shoulder": (x + .02, .5, .9),
                 "left_hip": (x - .02, .6, .9), "right_hip": (x + .02, .6, .9),
                 "left_wrist": (x - .04, .6, .9), "right_wrist": (x + .04, .6, .9)}
    return PoseFrame(frame, frame / 30, landmarks, track_id)


def ball(x, y, frame=0):
    return Detection(frame, frame / 30, x, y, .9, .01)


def frames(balls, players=(1, 2), possession=None):
    return [{"players": [player(1, .3, t), player(2, .7, t)][:len(players)],
             "ball": b, "possession": (possession or {}).get(t, [])}
            for t, b in enumerate(balls)]


def ids(decisions):
    return [d.track_id for d in decisions]


def test_hidden_dribble_between_contacts_keeps_the_handler():
    balls = [ball(.34, .6)] + [None] * 18 + [ball(.34, .6)]
    assert ids(decode_handlers(frames(balls), 1.0)) == [1] * 20


def test_pass_switches_at_the_catch_with_nobody_in_flight():
    balls = ([ball(.34, .6, t) for t in range(5)] + [ball(.5, .3, t) for t in range(5, 9)]
             + [ball(.66, .6, t) for t in range(9, 15)])
    assert ids(decode_handlers(frames(balls), 1.0)) == [1] * 5 + [None] * 4 + [2] * 6


def test_ball_visible_away_from_everyone_ends_possession():
    balls = [ball(.34, .6, t) for t in range(6)] + [ball(.5, .1, t) for t in range(6, 30)]
    decoded = ids(decode_handlers(frames(balls), 1.0))
    assert decoded[:6] == [1] * 6 and set(decoded[8:]) == {None}


def test_low_dribble_beside_the_body_counts_as_possession():
    # Ball at the floor beside player 1: far from both wrists, inside the dribble zone.
    balls = [ball(.34, .6, 0)] + [ball(.36, .78, t) if t % 2 else ball(.34, .6, t) for t in range(1, 40)]
    assert set(ids(decode_handlers(frames(balls), 1.0))) == {1}


def test_possession_box_alone_identifies_the_handler():
    boxes = {t: [(.8, (.6, .4, .8, .9))] for t in range(12)}
    decoded = decode_handlers(frames([None] * 12, possession=boxes), 1.0)
    assert ids(decoded) == [2] * 12
    assert {d.source for d in decoded} == {"observed"}
