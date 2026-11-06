"""Learned make/miss call over measurements the outcome rules already take.

A small gradient-boosted tree model (numpy only) reads per-shot measurements of the
ball's way down through the rim plane, the net and the next attempt, and gives
p(made). Where a rim was known at the ball's arrival its call replaces the rules'
made/missed/unknown: made at p >= the model's high threshold, missed at p <= its low
one, else unknown. With no rim the rules' result stays (the model has nothing to read
there). Trained by scripts/train_outcome_model.py on the dev labels; set
BALLFORM_OUTCOME_MODEL=0 to turn it off.
"""
from __future__ import annotations

import bisect
import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from app import scoring
from app.scoring import _rim_at

MODEL_PATH = Path(__file__).with_name("outcome_model.json")
NAN = float("nan")
MISSING = -1e9  # a missing measurement sorts below every split threshold
SEEN = .45  # ball detections trusted for the measurements, as in the rules
MADE = {"made", "likely made"}
RIM_TYPES = {"layup", "dunk", "layup or dunk", "tip"}

FEATURES = [
    "n_cross", "n_late", "has_cross", "cross_inside", "cross_on_time", "cross_u", "cross_center_off", "cross_dt",
    "cross_conf", "arrive_dt", "arrive_after_release_s", "sat_s", "anchor_kind", "max_rise_after", "n_above_after",
    "n_beside_net", "n_in_net", "deepest_v", "deepest_u_off", "n_post", "net_peak_cross", "net_peak_anchor",
    "net_peak_window", "net_base", "ball_vis_after_apex", "rim_vis", "rim_w_px", "rim_hw", "ball_r_over_rim",
    "apex_above_rim_w", "rule_rim_bounce", "rule_bounced_out", "rule_beside_net", "bib_peak", "bib_count",
    "next_gap_s", "next_made", "is_rim_path", "is_rim_type",
]


def enabled() -> bool:
    return os.environ.get("BALLFORM_OUTCOME_MODEL", "1") != "0"


def remember(shot, apex_frame: int, segment, track) -> None:
    """Keep what the rules measured a shot's outcome from (only called with a rim known by
    arrival); read by ``outcome_features``. Not a dataclass field, so never serialized."""
    shot.__dict__["_outcome_inputs"] = (apex_frame, list(segment), track)


def _round6(value: float) -> float:
    return float(f"{value:.6g}")


class GBM:
    """Gradient-boosted regression trees on the logistic loss, histogram splits (up to
    ``bins`` cut points per feature), missing values (NaN) always on the left."""

    def __init__(self, n_estimators: int = 150, learning_rate: float = .05, max_depth: int = 3,
                 min_leaf: int = 8, lam: float = 1., bins: int = 32):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.min_leaf, self.lam, self.bins = min_leaf, lam, bins
        self.f0 = 0.
        self.trees: list[dict] = []

    def _cuts(self, X: np.ndarray) -> list[np.ndarray]:
        cuts = []
        for j in range(X.shape[1]):
            vals = np.unique(X[~np.isnan(X[:, j]), j])
            if len(vals) <= 1:
                cuts.append(np.array([]))
                continue
            if len(vals) <= self.bins + 1:
                c = (vals[1:] + vals[:-1]) / 2
            else:
                c = np.quantile(vals, np.linspace(0, 1, self.bins + 2)[1:-1])
            cuts.append(np.unique([_round6(v) for v in c]))
        return cuts

    def fit(self, X, y) -> "GBM":
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, d = X.shape
        cuts = self._cuts(X)
        nb = max(len(c) for c in cuts) + 2 if cuts else 2
        # bin 0 = missing; bin b >= 1 holds values with b - 1 cut points below them
        B = np.zeros((n, d), int)
        for j, c in enumerate(cuts):
            ok = ~np.isnan(X[:, j])
            B[ok, j] = 1 + np.searchsorted(c, X[ok, j], side="left")
        offsets = np.arange(d) * nb
        p0 = min(max(y.mean(), 1e-3), 1 - 1e-3)
        self.f0 = _round6(math.log(p0 / (1 - p0)))
        f = np.full(n, self.f0)
        self.trees = []
        for _ in range(self.n_estimators):
            p = 1 / (1 + np.exp(-f))
            g, h = p - y, p * (1 - p)
            tree = {"feature": [], "threshold": [], "left": [], "right": [], "value": []}
            self._grow(tree, B, cuts, offsets, nb, g, h, np.arange(n), self.max_depth)
            self.trees.append(tree)
            f += self.learning_rate * self._apply(tree, X)
        return self

    def _grow(self, tree, B, cuts, offsets, nb, g, h, idx, depth) -> int:
        node = len(tree["feature"])
        for key in tree:
            tree[key].append(-1 if key in ("feature", "left", "right") else 0.)
        G, H = g[idx].sum(), h[idx].sum()
        tree["value"][node] = _round6(-G / (H + self.lam))
        if depth == 0 or len(idx) < 2 * self.min_leaf:
            return node
        d = B.shape[1]
        flat = (B[idx] + offsets).ravel()
        size = d * nb
        hg = np.bincount(flat, np.repeat(g[idx], d), size).reshape(d, nb).cumsum(1)
        hh = np.bincount(flat, np.repeat(h[idx], d), size).reshape(d, nb).cumsum(1)
        hc = np.bincount(flat, None, size).reshape(d, nb).cumsum(1)
        # left = bins 0..t (t = 0 splits missing from present); right = the rest
        GL, HL, CL = hg[:, :-1], hh[:, :-1], hc[:, :-1]
        GR, HR, CR = G - GL, H - HL, len(idx) - CL
        gain = GL ** 2 / (HL + self.lam) + GR ** 2 / (HR + self.lam) - G ** 2 / (H + self.lam)
        n_cuts = np.array([len(c) for c in cuts])
        valid = (CL >= self.min_leaf) & (CR >= self.min_leaf) & (np.arange(nb - 1)[None, :] <= n_cuts[:, None])
        gain = np.where(valid, gain, -np.inf)
        j, t = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if not gain[j, t] > 1e-9:
            return node
        threshold = MISSING / 2 if t == 0 else float(cuts[j][t - 1])
        go_left = B[idx, j] <= t
        tree["feature"][node], tree["threshold"][node] = int(j), threshold
        tree["left"][node] = self._grow(tree, B, cuts, offsets, nb, g, h, idx[go_left], depth - 1)
        tree["right"][node] = self._grow(tree, B, cuts, offsets, nb, g, h, idx[~go_left], depth - 1)
        return node

    @staticmethod
    def _apply(tree: dict, X: np.ndarray) -> np.ndarray:
        X = np.where(np.isnan(X), MISSING, X)
        feature, threshold = np.asarray(tree["feature"]), np.asarray(tree["threshold"], float)
        left, right, value = np.asarray(tree["left"]), np.asarray(tree["right"]), np.asarray(tree["value"], float)
        node = np.zeros(len(X), int)
        rows = np.arange(len(X))
        while True:
            j = feature[node]
            inner = j >= 0
            if not inner.any():
                return value[node]
            go_left = X[rows, np.maximum(j, 0)] <= threshold[node]
            node = np.where(inner, np.where(go_left, left[node], right[node]), node)

    def decision(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        f = np.full(len(X), self.f0)
        for tree in self.trees:
            f += self.learning_rate * self._apply(tree, X)
        return f

    def predict_proba(self, X) -> np.ndarray:
        """Columns p(missed), p(made)."""
        p = 1 / (1 + np.exp(-self.decision(X)))
        return np.column_stack([1 - p, p])

    def to_dict(self) -> dict:
        return {"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "max_depth": self.max_depth,
                "min_leaf": self.min_leaf, "lam": self.lam, "bins": self.bins, "f0": self.f0, "trees": self.trees}

    @classmethod
    def from_dict(cls, data: dict) -> "GBM":
        model = cls(data["n_estimators"], data["learning_rate"], data["max_depth"], data["min_leaf"], data["lam"],
                    data["bins"])
        model.f0, model.trees = float(data["f0"]), data["trees"]
        return model


class OutcomeModel:
    """A GBM over ``features`` and its call thresholds."""

    def __init__(self, gbm: GBM, features: list[str], t_lo: float, t_hi: float, meta: dict | None = None):
        self.gbm, self.features, self.t_lo, self.t_hi, self.meta = gbm, features, t_lo, t_hi, meta or {}

    def p_made(self, rows: list[dict]) -> np.ndarray:
        X = np.array([[float(row.get(name, NAN)) for name in self.features] for row in rows], float).reshape(
            len(rows), len(self.features))
        return self.gbm.predict_proba(X)[:, 1]

    def call(self, p: float) -> str:
        return "made" if p >= self.t_hi else "missed" if p <= self.t_lo else "unknown"

    def to_dict(self) -> dict:
        return {"version": 1, "features": self.features, "t_lo": self.t_lo, "t_hi": self.t_hi, "meta": self.meta,
                "gbm": self.gbm.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "OutcomeModel":
        return cls(GBM.from_dict(data["gbm"]), list(data["features"]), float(data["t_lo"]), float(data["t_hi"]),
                   data.get("meta"))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path: Path) -> "OutcomeModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


@lru_cache(maxsize=4)
def load_model(path: str = str(MODEL_PATH)) -> OutcomeModel | None:
    """The shipped model, loaded once; None when the file is absent."""
    return OutcomeModel.load(Path(path)) if Path(path).exists() else None


def _strength(value) -> float:
    return float(value.get("strength", 0.)) if isinstance(value, dict) else float(value)


def _peak(net_motion, start, end) -> float:
    if start is None or end is None:
        return NAN
    return max((_strength(v) for frame, v in (net_motion or {}).items() if start <= frame <= end), default=0.)


def _sampled(clip: dict, start: float, end: float) -> list[int]:
    frames = clip["sampled"]
    return frames[bisect.bisect_left(frames, start):bisect.bisect_right(frames, end)]


def rim_features(segment, apex_frame: int, rim, net_motion, fps: float, track, clip: dict,
                 release_frame: int | None = None) -> tuple[dict, int | None]:
    """Measurements of one shot's way down to the rim (the inputs ``scoring.rim_outcome`` reads),
    and the frame they centre on (the crossing, else the arrival at the rim, else None)."""
    track = sorted(track or segment, key=lambda b: b.frame)
    width, height = clip["width"], clip["height"]
    descending = [b for b in segment if b.frame >= apex_frame]
    crossings = []
    for p1, p2 in zip(descending, descending[1:]):
        r1, r2 = _rim_at(rim, p1.frame), _rim_at(rim, p2.frame)
        if r1 is None or r2 is None:
            continue
        rel1, rel2 = p1.y - (r1[1] + .45 * r1[3]), p2.y - (r2[1] + .45 * r2[3])
        if rel1 > 0 or rel2 <= rel1 or rel2 < 0:
            continue
        a = -rel1 / max(rel2 - rel1, 1e-9)
        x = p1.x + a * (p2.x - p1.x)
        box = tuple(r1[i] + a * (r2[i] - r1[i]) for i in range(4))
        if p2.frame - p1.frame <= max(3, fps * .25):
            crossings.append((p2.frame, (x - box[0]) / box[2], min(p1.confidence, p2.confidence)))
    arrival = apex_frame + scoring.ARRIVAL_S * fps
    on_time = [c for c in crossings if c[0] <= arrival]
    late = [c for c in crossings if c[0] > arrival]
    inside = [c for c in on_time if -.08 <= c[1] <= 1.08]
    first = inside[0] if inside else on_time[0] if on_time else late[0] if late else None
    f = {"n_cross": len(on_time), "n_late": len(late), "has_cross": float(first is not None),
         "cross_inside": float(bool(inside)), "cross_on_time": float(bool(on_time)),
         "cross_u": first[1] if first else NAN, "cross_center_off": abs(first[1] - .5) if first else NAN,
         "cross_dt": (first[0] - apex_frame) / fps if first else NAN, "cross_conf": first[2] if first else NAN}
    seen = [b for b in track if b.confidence >= SEEN]
    flight = [b for b in seen if apex_frame <= b.frame <= apex_frame + 1.5 * fps]
    arrived = None
    if flight:
        top = min(flight, key=lambda b: b.y)
        arrived = next((b.frame for b in flight if b.frame >= top.frame and (box := _rim_at(rim, b.frame)) is not None
                        and box[0] - .5 * box[2] <= b.x <= box[0] + 1.5 * box[2]
                        and b.y >= box[1] - scoring.BOUNCE_REACH * box[3]), None)
    crossed = first[0] if first else None
    f["arrive_dt"] = (arrived - apex_frame) / fps if arrived is not None else NAN
    f["arrive_after_release_s"] = ((arrived - release_frame) / fps
                                   if arrived is not None and release_frame is not None else NAN)
    f["sat_s"] = (crossed - arrived) / fps if crossed is not None and arrived is not None and crossed >= arrived else NAN
    anchor = crossed if crossed is not None else arrived
    f["anchor_kind"] = 2 if crossed is not None else 1 if arrived is not None else 0
    rises, beside, in_net, deepest, deepest_u, n_post = [], 0, 0, NAN, NAN, 0
    if anchor is not None:
        for b in seen:
            if not anchor < b.frame <= anchor + .6 * fps or (box := _rim_at(rim, b.frame)) is None:
                continue
            n_post += 1
            u, v = (b.x - box[0]) / box[2], (b.y - box[1]) / box[3]
            if -1 <= u <= 2:
                rises.append(-v)
            if b.frame <= anchor + .3 * fps and .5 <= v <= 2.2:
                if 0 <= u <= 1:
                    in_net += 1
                else:
                    beside += 1
            if b.frame <= anchor + .5 * fps and (math.isnan(deepest) or v > deepest):
                deepest, deepest_u = v, abs(u - .5)
    f.update({"max_rise_after": max(rises) if rises else NAN, "n_above_after": sum(r > 0 for r in rises),
              "n_beside_net": beside, "n_in_net": in_net, "deepest_v": deepest, "deepest_u_off": deepest_u,
              "n_post": n_post})
    f["net_peak_cross"] = _peak(net_motion, crossed, None if crossed is None else crossed + .35 * fps)
    f["net_peak_anchor"] = _peak(net_motion, anchor, None if anchor is None else anchor + .6 * fps)
    f["net_peak_window"] = _peak(net_motion, apex_frame, apex_frame + 1.5 * fps)
    base = [_strength(v) for frame, v in (net_motion or {}).items() if apex_frame - 1.5 * fps <= frame < apex_frame]
    f["net_base"] = float(np.median(base)) if base else NAN
    by_frame = {b.frame: b for b in track}
    after = _sampled(clip, apex_frame, apex_frame + fps)
    f["ball_vis_after_apex"] = (sum(fr in clip["observed"] and fr in by_frame and by_frame[fr].confidence >= SEEN
                                    for fr in after) / len(after)) if after else NAN
    window = _sampled(clip, apex_frame, apex_frame + 1.5 * fps)
    f["rim_vis"] = (sum(_rim_at(rim, fr) is not None for fr in window) / len(window)) if window else NAN
    box = _rim_at(rim, apex_frame) or (_rim_at(rim, anchor) if anchor is not None else None)
    if box is None:
        box = next((b for fr in window if (b := _rim_at(rim, fr)) is not None), None)
    apex_ball = by_frame.get(apex_frame)
    f.update({"rim_w_px": NAN, "rim_hw": NAN, "ball_r_over_rim": NAN, "apex_above_rim_w": NAN})
    if box is not None and box[2] > 0:
        rim_px = box[2] * width
        radii = [b.radius for b in flight if b.radius > 0]
        f.update({"rim_w_px": rim_px, "rim_hw": box[3] * height / rim_px,
                  "ball_r_over_rim": float(np.median(radii)) * max(width, height) / rim_px if radii else NAN,
                  "apex_above_rim_w": (box[1] - apex_ball.y) * height / rim_px if apex_ball else NAN})
    f["rule_rim_bounce"] = float(scoring.rim_bounce(track, apex_frame, arrival, rim, fps) is not None)
    f["rule_bounced_out"] = float(bool(inside) and scoring.bounced_out(track, inside[0][0], rim, fps))
    f["rule_beside_net"] = float(bool(inside) and scoring.beside_net(track, inside[0][0], rim, fps))
    basket = [conf for frame, conf in clip["ball_in_basket"] if apex_frame - .3 * fps <= frame <= apex_frame + 1.5 * fps]
    f["bib_peak"] = max(basket, default=0.)
    f["bib_count"] = len(basket)
    return {k: float(v) for k, v in f.items()}, anchor


def clip_context(width: int, height: int, observed_balls, player_frames) -> dict:
    """What the measurements need from the whole clip: frame size, raw ball detections,
    sampled frames and the detector's ball-in-basket class."""
    return {"width": width, "height": height, "observed": {b.frame for b in observed_balls},
            "sampled": sorted(f["frame"] for f in player_frames),
            "ball_in_basket": [(f["frame"], float(conf)) for f in player_frames
                               for kind, conf, *_ in f.get("events", []) if kind == "ball_in_basket"]}


def outcome_features(shots, rim, net_motion, fps: float, clip: dict) -> list[tuple[dict, int | None] | None]:
    """(features, anchor frame) per shot whose outcome the rules measured with a rim known
    at arrival, else None. Read before any call is changed, so training (model off) and
    serving see the same values."""
    ordered = sorted(range(len(shots)), key=lambda i: shots[i].release_s)
    following = {i: (ordered[k + 1] if k + 1 < len(ordered) else None) for k, i in enumerate(ordered)}
    out: list = []
    for i, shot in enumerate(shots):
        inputs = shot.__dict__.get("_outcome_inputs")
        if inputs is None or rim is None:
            out.append(None)
            continue
        apex_frame, segment, track = inputs
        release = round(shot.release_s * fps)
        feats, anchor = rim_features(segment, apex_frame, rim, net_motion, fps, track, clip, release)
        anchor = anchor if anchor is not None else apex_frame
        nxt = following[i]
        feats["next_gap_s"] = min(shots[nxt].release_s - shot.release_s, 10.) if nxt is not None else 10.
        feats["next_made"] = float(nxt is not None and shots[nxt].outcome in MADE)
        feats["is_rim_path"] = float((shot.attempt or {}).get("path") in {"rim_attempt", "layup_dunk_class"})
        feats["is_rim_type"] = float(shot.shot_type in RIM_TYPES)
        out.append((feats, anchor))
    return out


def call_outcomes(shots, rim, net_motion, fps: float, clip: dict,
                  model: OutcomeModel | None = None) -> list[dict | None]:
    """Measure every shot, then, with the model on, replace the rules' call where a rim was
    known at arrival. Returns the per-shot features (None where there was no rim)."""
    measured = outcome_features(shots, rim, net_motion, fps, clip)
    if model is None and enabled():
        model = load_model()
    todo = [(shot, m) for shot, m in zip(shots, measured) if m is not None]
    if model is not None and todo:
        for (shot, (feats, anchor)), p in zip(todo, model.p_made([m[0] for _, m in todo])):
            p, call = float(p), model.call(float(p))
            shot.evidence.append(f"Learned outcome model: p(made) = {p:.2f} (made at {model.t_hi:.2f} or more, missed "
                                 f"at {model.t_lo:.2f} or less); the rules called it {shot.outcome}")
            if call == "unknown":
                shot.outcome_frame = None
            elif shot.outcome_frame is None:
                shot.outcome_frame = anchor
            shot.outcome = call
            shot.outcome_confidence = round(p if call == "made" else 1 - p if call == "missed" else 0., 2)
    return [m[0] if m is not None else None for m in measured]
