import json

import numpy as np
import pytest

from app import outcome_model
from app.models import Detection, ShotResult
from app.outcome_model import FEATURES, GBM, OutcomeModel, call_outcomes, clip_context, remember

RIM = (.45, .30, .10, .05)  # left, top, width, height (normalized)
FPS = 30.


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + .5 * X[:, 1] > 0).astype(int)
    X[rng.random((n, 4)) < .1] = np.nan  # missing measurements
    return X, y


def test_gbm_learns_and_round_trips_through_json():
    X, y = _data()
    model = GBM(n_estimators=60, learning_rate=.1, max_depth=3, min_leaf=5).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert ((p >= .5) == y).mean() > .9
    loaded = GBM.from_dict(json.loads(json.dumps(model.to_dict())))
    assert np.array_equal(loaded.predict_proba(X), model.predict_proba(X))
    # a row of all-missing values still gets a probability
    assert 0 < loaded.predict_proba(np.full((1, 4), np.nan))[0, 1] < 1


def test_gbm_is_deterministic():
    X, y = _data(seed=1)
    a = GBM(n_estimators=30).fit(X, y).to_dict()
    b = GBM(n_estimators=30).fit(X, y).to_dict()
    assert json.dumps(a) == json.dumps(b)


def test_outcome_model_saves_and_loads(tmp_path):
    X, y = _data(n=120, seed=2)
    names = ["a", "b", "c", "d"]
    model = OutcomeModel(GBM(n_estimators=20).fit(X, y), names, .2, .8, {"note": "test"})
    path = tmp_path / "m.json"
    model.save(path)
    loaded = OutcomeModel.load(path)
    rows = [dict(zip(names, x)) for x in X[:10]]
    assert np.array_equal(loaded.p_made(rows), model.p_made(rows))
    assert (loaded.t_lo, loaded.t_hi, loaded.meta) == (.2, .8, {"note": "test"})
    assert [loaded.call(p) for p in (.1, .5, .9)] == ["missed", "unknown", "made"]


class Constant:
    """Stands in for a trained model: the same p(made) for every shot."""

    def __init__(self, p):
        self.p, self.t_lo, self.t_hi = p, .2, .8

    def p_made(self, rows):
        return np.full(len(rows), self.p)

    def call(self, p):
        return OutcomeModel.call(self, p)


def _through_the_rim():
    """A ball coming down from an apex at frame 10 through the middle of RIM."""
    balls = [Detection(f, f / FPS, .5, .10 + .02 * (f - 10), .9, .01) for f in range(10, 26)]
    shot = ShotResult(1, 0., 10 / FPS, 25 / FPS, "unknown", 0., ["Ball arc detected"], {})
    return shot, balls


def _clip(balls):
    frames = [{"frame": b.frame, "events": []} for b in balls]
    return clip_context(1280, 720, balls, frames)


def test_features_cover_the_model_inputs_and_tolerate_gaps():
    shot, balls = _through_the_rim()
    remember(shot, 10, balls, balls)
    feats = call_outcomes([shot], RIM, {}, FPS, _clip(balls), model=Constant(.5))[0]
    assert set(FEATURES) <= set(feats)
    assert feats["cross_inside"] == 1. and abs(feats["cross_center_off"]) < .1
    # no ball after the crossing: missing values, not errors
    sparse = ShotResult(2, 0., 10 / FPS, 25 / FPS, "unknown", 0., [], {})
    remember(sparse, 10, balls[:3], balls[:3])
    feats = call_outcomes([sparse], RIM, None, FPS, _clip(balls[:3]), model=Constant(.5))[0]
    assert np.isnan(feats["cross_u"]) and feats["n_post"] == 0


def test_model_call_replaces_the_rules_where_the_rim_was_known():
    shot, balls = _through_the_rim()
    remember(shot, 10, balls, balls)
    call_outcomes([shot], RIM, {}, FPS, _clip(balls), model=Constant(.93))
    assert (shot.outcome, shot.outcome_confidence) == ("made", .93)
    assert 10 < shot.outcome_frame <= 25  # timed at the crossing
    assert "p(made) = 0.93" in shot.evidence[-1] and "rules called it unknown" in shot.evidence[-1]
    call_outcomes([shot], RIM, {}, FPS, _clip(balls), model=Constant(.5))
    assert (shot.outcome, shot.outcome_frame) == ("unknown", None)


def test_no_rim_keeps_the_rules_result():
    shot, balls = _through_the_rim()
    shot.outcome, shot.outcome_confidence = "missed", .66
    # never measured with a rim (no remember), or no rim at all: the model abstains
    assert call_outcomes([shot], RIM, {}, FPS, _clip(balls), model=Constant(.99)) == [None]
    remember(shot, 10, balls, balls)
    assert call_outcomes([shot], None, {}, FPS, _clip(balls), model=Constant(.99)) == [None]
    assert (shot.outcome, shot.outcome_confidence, shot.evidence) == ("missed", .66, ["Ball arc detected"])


@pytest.mark.parametrize("switch, outcome", [("0", "missed"), ("1", "made")])
def test_env_switch_turns_the_model_off(monkeypatch, switch, outcome):
    monkeypatch.setenv("BALLFORM_OUTCOME_MODEL", switch)
    monkeypatch.setattr(outcome_model, "load_model", lambda *a: Constant(.95))
    shot, balls = _through_the_rim()
    shot.outcome = "missed"
    remember(shot, 10, balls, balls)
    feats = call_outcomes([shot], RIM, {}, FPS, _clip(balls))
    assert shot.outcome == outcome and feats[0] is not None


def test_shipped_model_reads_the_measured_features():
    model = outcome_model.load_model()
    if model is None:
        pytest.skip("no trained model shipped")
    assert set(model.features) <= set(FEATURES)
    assert 0 < model.t_lo < model.t_hi < 1
    shot, balls = _through_the_rim()
    remember(shot, 10, balls, balls)
    feats = call_outcomes([shot], RIM, {}, FPS, _clip(balls), model=model)[0]
    assert 0 <= model.p_made([feats])[0] <= 1
