"""Learned check that a predicted shot is a real attempt (game mode).

A second small gradient-boosted tree model (app/outcome_model.GBM) gives p(real) for each
shot the rules found, from what was seen around its release: how it was found, the
detector's classes, hand contacts, ball and rim visibility, the neighbouring shots and the
rim measurements. Shots under the model's threshold are dropped; nothing is ever added.
Trained on broadcast clips, it is applied to a camera following play only.
Trained by scripts/train_shot_verifier.py on the dev labels; set BALLFORM_SHOT_VERIFIER=0
to turn it off.
"""
from __future__ import annotations

import bisect
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.outcome_model import GBM, MADE, SEEN
from app.scoring import _rim_at

MODEL_PATH = Path(__file__).with_name("shot_verifier.json")
NAN = float("nan")
PATHS = ["arc", "rim_attempt", "hidden_release", "layup_dunk_class"]
TYPES = ["jump shot", "floater", "layup or dunk", "tip", "layup"]
FLAGS = {"release_moved": "Release moved from frame", "hidden_release": "Release contact hidden",
         "shooter_withheld": "shooter withheld", "putback": "Putback", "blocked": "Blocked:"}
FROM_RIM = ["has_cross", "apex_above_rim_w", "rim_w_px", "ball_vis_after_apex", "bib_peak", "net_peak_window",
            "arrive_after_release_s"]
FEATURES = (
    [f"path_{p}" for p in PATHS] + [f"type_{t}" for t in TYPES]
    + ["rules_made", "rules_missed", "outcome_conf", "cls_jump", "cls_layup", "cls_block", "cls_bib_2s",
       "cls_possession", "ball_vis_rel", "rim_vis_2s", "rim_at_release", "near_cut_s", "n_cuts", "prev_gap_s",
       "next_gap_s", "n_shots_clip", "prev_made", "next_made", "n_contacts_1s_before", "n_contact_players_1s",
       "n_contacts_after_1s"]
    + [f"flag_{k}" for k in FLAGS] + [f"rim_{k}" for k in FROM_RIM])


def enabled() -> bool:
    return os.environ.get("BALLFORM_SHOT_VERIFIER", "1") != "0"


class Verifier:
    """A GBM over ``features`` giving p(real), and the threshold under which a shot is dropped."""

    def __init__(self, gbm: GBM, features: list[str], threshold: float, meta: dict | None = None):
        self.gbm, self.features, self.threshold, self.meta = gbm, features, threshold, meta or {}

    def p_real(self, rows: list[dict]) -> np.ndarray:
        X = np.array([[float(row.get(name, NAN)) for name in self.features] for row in rows], float).reshape(
            len(rows), len(self.features))
        return self.gbm.predict_proba(X)[:, 1]

    def to_dict(self) -> dict:
        return {"version": 1, "features": self.features, "threshold": self.threshold, "meta": self.meta,
                "gbm": self.gbm.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "Verifier":
        return cls(GBM.from_dict(data["gbm"]), list(data["features"]), float(data["threshold"]), data.get("meta"))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Verifier":
        return cls.from_dict(json.loads(Path(path).read_text()))


@lru_cache(maxsize=4)
def load_model(path: str = str(MODEL_PATH)) -> Verifier | None:
    return Verifier.load(Path(path)) if Path(path).exists() else None


def _peak(by_class: dict, name: str, start: float, end: float) -> float:
    frames, confs = by_class.get(name, ([], []))
    lo, hi = bisect.bisect_left(frames, start), bisect.bisect_right(frames, end)
    return max(confs[lo:hi], default=0.)


def shot_features(shots, rim, fps: float, player_frames, observed_balls, contacts, cut_frames,
                  outcome_features=None) -> list[dict]:
    """Per shot, what was seen around its release (rules' outcomes; read before any change)."""
    by_class: dict[str, tuple[list, list]] = {}
    for frame in sorted(player_frames, key=lambda f: f["frame"]):
        best: dict[str, float] = {}
        for kind, conf, *_ in frame.get("events", []):
            best[kind] = max(best.get(kind, 0.), float(conf))
        for kind, conf in best.items():
            by_class.setdefault(kind, ([], []))
            by_class[kind][0].append(frame["frame"])
            by_class[kind][1].append(conf)
    sampled = sorted(f["frame"] for f in player_frames)
    seen = {b.frame for b in observed_balls if b.confidence >= SEEN}
    contacts = sorted(contacts, key=lambda c: c.frame)
    contact_frames = [c.frame for c in contacts]
    order = sorted(range(len(shots)), key=lambda i: shots[i].release_s)
    rank = {i: k for k, i in enumerate(order)}
    rows = []
    for i, shot in enumerate(shots):
        r = round(shot.release_s * fps)
        path = (shot.attempt or {}).get("path", "arc")
        f = {f"path_{p}": float(path == p) for p in PATHS}
        f.update({f"type_{t}": float(shot.shot_type == t) for t in TYPES})
        f["rules_made"], f["rules_missed"] = float(shot.outcome in MADE), float(shot.outcome == "missed")
        f["outcome_conf"] = float(shot.outcome_confidence)
        w = .4 * fps
        f["cls_jump"] = _peak(by_class, "jump_shot", r - w, r + w)
        f["cls_layup"] = _peak(by_class, "layup_dunk", r - w, r + w)
        f["cls_block"] = _peak(by_class, "shot_block", r, r + 1.5 * fps)
        f["cls_bib_2s"] = _peak(by_class, "ball_in_basket", r, r + 2 * fps)
        f["cls_possession"] = _peak(by_class, "player_in_possession", r - w, r)
        around = sampled[bisect.bisect_left(sampled, r - .5 * fps):bisect.bisect_right(sampled, r + fps)]
        f["ball_vis_rel"] = sum(fr in seen for fr in around) / len(around) if around else NAN
        after = sampled[bisect.bisect_left(sampled, r):bisect.bisect_right(sampled, r + 2 * fps)]
        f["rim_vis_2s"] = sum(_rim_at(rim, fr) is not None for fr in after) / len(after) if after else NAN
        f["rim_at_release"] = float(_rim_at(rim, r) is not None)
        f["near_cut_s"] = min([abs(r - c) / fps for c in cut_frames] + [5.])
        f["n_cuts"] = float(len(cut_frames))
        k = rank[i]
        prev = shots[order[k - 1]] if k > 0 else None
        nxt = shots[order[k + 1]] if k + 1 < len(order) else None
        f["prev_gap_s"] = min(shot.release_s - prev.release_s, 10.) if prev else 10.
        f["next_gap_s"] = min(nxt.release_s - shot.release_s, 10.) if nxt else 10.
        f["n_shots_clip"] = float(len(shots))
        f["prev_made"] = float(prev is not None and prev.outcome in MADE)
        f["next_made"] = float(nxt is not None and nxt.outcome in MADE)
        before = contacts[bisect.bisect_left(contact_frames, r - fps):bisect.bisect_right(contact_frames, r)]
        f["n_contacts_1s_before"] = float(len(before))
        f["n_contact_players_1s"] = float(len({c.track_id for c in before}))
        f["n_contacts_after_1s"] = float(bisect.bisect_right(contact_frames, r + fps)
                                         - bisect.bisect_right(contact_frames, r))
        f.update({f"flag_{k}": float(any(needle in e for e in shot.evidence)) for k, needle in FLAGS.items()})
        measured = outcome_features[i] if outcome_features else None
        f.update({f"rim_{k}": (measured or {}).get(k, NAN) for k in FROM_RIM})
        rows.append(f)
    return rows


def kept_shots(shots, features: list[dict], profile: str = "moving", model: Verifier | None = None) -> list[int]:
    """Indices of the shots the verifier keeps (all of them with it off or untrained, and on
    a fixed camera, which it was not trained on); the kept shots are renumbered."""
    if profile != "moving":
        return list(range(len(shots)))
    if model is None and enabled():
        model = load_model()
    if model is None or not shots:
        return list(range(len(shots)))
    kept = [i for i, value in enumerate(model.p_real(features)) if value >= model.threshold]
    for number, i in enumerate(kept, 1):
        shots[i].number = number
    return kept
