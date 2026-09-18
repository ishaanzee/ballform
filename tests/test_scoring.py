from app.models import Detection
from app.scoring import analyze_shots


def ball(frame, x, y):
    return Detection(frame, frame / 10, x, y, .9, .01)


def test_made_shot_crosses_rim_on_descent():
    track = [
        ball(0, .30, .70), ball(1, .35, .56), ball(2, .40, .40), ball(3, .46, .25),
        ball(4, .51, .18), ball(5, .53, .23), ball(6, .54, .32), ball(7, .55, .43),
        ball(8, .55, .55),
    ]
    shots = analyze_shots(track, [], 10, (.48, .36, .14, .08), {7: 3.0})
    assert len(shots) == 1
    assert shots[0].outcome == "made"
    assert shots[0].outcome_confidence > .7


def test_unknown_without_rim():
    track = [ball(0, .3, .7), ball(1, .4, .5), ball(2, .5, .2), ball(3, .6, .5), ball(4, .7, .7)]
    shots = analyze_shots(track, [], 5, None)
    assert shots[0].outcome == "unknown"
