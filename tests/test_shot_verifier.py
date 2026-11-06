import numpy as np
import pytest

from app import shot_verifier
from app.models import Detection, ShotResult
from app.outcome_model import GBM
from app.shot_verifier import FEATURES, Verifier, kept_shots, shot_features

FPS = 30.


class Constant:
    def __init__(self, values, threshold=.3):
        self.values, self.threshold = values, threshold

    def p_real(self, rows):
        return np.array(self.values[:len(rows)], float)


def _shots(n=3):
    return [ShotResult(i + 1, 2. * i, 2. * i + .5, 2. * i + 1, "missed", .7, ["Ball arc detected"], {},
                       shot_type="jump shot") for i in range(n)]


def test_features_for_every_shot():
    shots = _shots()
    frames = [{"frame": f, "events": [("jump_shot", .8, (0, 0, 1, 1))] if f == 15 else []} for f in range(0, 200, 3)]
    balls = [Detection(f, f / FPS, .5, .5, .9) for f in range(0, 200, 3)]
    rows = shot_features(shots, (.45, .3, .1, .05), FPS, frames, balls, [], [100], [None, {"has_cross": 1.}, None])
    assert len(rows) == 3 and all(set(FEATURES) <= set(row) for row in rows)
    assert rows[0]["cls_jump"] == .8 and rows[1]["rim_has_cross"] == 1. and np.isnan(rows[0]["rim_has_cross"])
    assert rows[0]["next_gap_s"] == 2. and rows[2]["next_gap_s"] == 10.


def test_drops_unlikely_shots_and_renumbers():
    shots = _shots()
    kept = kept_shots(shots, [{}] * 3, "moving", Constant([.9, .1, .8]))
    assert kept == [0, 2] and [shots[i].number for i in kept] == [1, 2]


def test_fixed_cameras_keep_every_shot():
    shots = _shots()
    assert kept_shots(shots, [{}] * 3, "elevated", Constant([0., 0., 0.])) == [0, 1, 2]


@pytest.mark.parametrize("switch, kept", [("0", [0, 1, 2]), ("1", [0, 2])])
def test_env_switch_turns_the_verifier_off(monkeypatch, switch, kept):
    monkeypatch.setenv("BALLFORM_SHOT_VERIFIER", switch)
    monkeypatch.setattr(shot_verifier, "load_model", lambda *a: Constant([.9, .1, .8]))
    assert kept_shots(_shots(), [{}] * 3, "moving") == kept


def test_verifier_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, len(FEATURES)))
    y = (X[:, 0] > 0).astype(int)
    model = Verifier(GBM(n_estimators=10).fit(X, y), FEATURES, .25)
    model.save(tmp_path / "v.json")
    loaded = Verifier.load(tmp_path / "v.json")
    rows = [dict(zip(FEATURES, x)) for x in X[:5]]
    assert np.array_equal(loaded.p_real(rows), model.p_real(rows)) and loaded.threshold == .25


def test_shipped_verifier_reads_the_measured_features():
    model = shot_verifier.load_model()
    if model is None:
        pytest.skip("no trained verifier shipped")
    assert set(model.features) <= set(FEATURES) and 0 <= model.threshold < 1
