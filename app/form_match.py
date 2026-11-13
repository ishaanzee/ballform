"""Match a shooter's free-throw signature to the player templates in form_db.json.

Each feature is weighted by how well it tells players apart: the share of its variance that
lies between players rather than between one player's own reps. A feature that every player
repeats closely but that differs from player to player counts most; one that wanders from
rep to rep counts little. scripts/formdb/build.py builds the templates.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from app.form_signature import FEATURES, JOINTS, SKELETON_TIMES

DB_PATH = Path(__file__).with_name("form_db.json")
# Shoulder width over torso length. Side-on broadcast free throws measure 0.17-0.27, oblique ones 0.44+.
MAX_SIDE_SHOULDER_WIDTH = 0.35
# feature -> (what to do when yours is lower than the player's, what to do when it is higher)
ADVICE = {
    "set_elbow_deg": ("Keep your shooting arm more open at the set point", "Fold your shooting arm more at the set point"),
    "set_height": ("Bring the ball higher before you push", "Start your push from a lower set point"),
    "set_forward": ("Set the ball further out in front of your face", "Bring your set point closer to your head"),
    "release_elbow_deg": ("Extend your elbow more fully through the release", "Release with a little more bend left in the elbow"),
    "release_height": ("Reach higher at release", "Release lower and more out in front"),
    "release_arm_deg": ("Finish with your arm more upright", "Finish with your arm reaching more toward the rim"),
    "extend_time_s": ("Slow the push from set point to release", "Push from set point to release faster"),
    "lift_time_s": ("Bring the ball up more slowly", "Bring the ball up faster"),
    "knee_bend_deg": ("Bend your knees less", "Bend your knees more"),
    "knee_lead_s": ("Start your knee bend earlier before the release", "Time your knee bend closer to the release"),
    "trunk_lean_deg": ("Lean a little more toward the rim at release", "Stay more upright at release"),
    "follow_hold_s": ("Hold your follow-through longer", "Drop your follow-through sooner"),
}


def _median(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.median(kept)) if kept else None


def _spread(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return float(np.std(kept, ddof=1)) if len(kept) >= 2 else None


def median_features(reps: Sequence[dict]) -> dict[str, float | None]:
    return {name: _median([r["features"].get(name) for r in reps]) for name in FEATURES}


def median_skeleton(reps: Sequence[dict]) -> list[dict | None]:
    frames = []
    for i in range(len(SKELETON_TIMES)):
        seen = [r["skeleton"][i] for r in reps if len(r.get("skeleton") or []) > i and r["skeleton"][i]]
        frames.append({joint: [round(float(np.median([f[joint][k] for f in seen])), 3) for k in (0, 1)]
                       for joint in JOINTS} if seen else None)
    return frames


def build_db(players: Sequence[dict]) -> dict:
    """players: [{"name", "hand", "reps": [signature, ...]}] -> the form_db.json content."""
    entries = [{"name": p["name"], "hand": p.get("hand", "right"), "reps": len(p["reps"]),
                "features": median_features(p["reps"]),
                "spread": {name: _spread([r["features"].get(name) for r in p["reps"]]) for name in FEATURES},
                "skeleton": median_skeleton(p["reps"])} for p in players]
    features = {}
    for name in FEATURES:
        medians = [e["features"][name] for e in entries if e["features"][name] is not None]
        spreads = [e["spread"][name] for e in entries if e["spread"][name] is not None]
        if len(medians) < 2 or not spreads:
            continue
        between, within = float(np.std(medians, ddof=1)), float(np.sqrt(np.mean(np.square(spreads))))
        if between <= 1e-9:
            continue
        features[name] = {"scale": round(between, 4), "within": round(within, 4),
                          "weight": round(between ** 2 / (between ** 2 + within ** 2), 3)}
    return {"version": 1, "shot": "free throw", "features": features, "players": entries}


def load_db(path: Path = DB_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _distance(mine: dict, theirs: dict, features: dict) -> tuple[float | None, dict[str, float]]:
    """Weighted distance in between-player standard deviations, and each feature's share of it."""
    total = weight = 0.0
    parts = {}
    for name, info in features.items():
        a, b = mine.get(name), theirs.get(name)
        if a is None or b is None:
            continue
        parts[name] = info["weight"] * ((a - b) / info["scale"]) ** 2
        total += parts[name]
        weight += info["weight"]
    return (math.sqrt(total / weight), parts) if weight > 0 else (None, parts)


def _format(name: str, value: float) -> str:
    unit = FEATURES[name][1]
    return f"{value:.0f}°" if unit == "°" else f"{value:.2f} s" if unit == "s" else f"{value:.2f} torso lengths"


def feedback(mine: dict, player: dict, features: dict, limit: int = 3) -> list[dict]:
    """The measured differences that matter most for looking like ``player``, largest first."""
    _, parts = _distance(mine, player["features"], features)
    notes = []
    for name in sorted(parts, key=parts.get, reverse=True):
        yours, theirs = mine[name], player["features"][name]
        # Inside the player's own rep-to-rep variation there is nothing to change.
        if abs(yours - theirs) <= max(player["spread"].get(name) or 0.0, features[name]["within"]):
            continue
        advice = ADVICE[name][0 if yours < theirs else 1]
        notes.append({"feature": name, "label": FEATURES[name][0], "yours": round(yours, 3), "theirs": round(theirs, 3),
                      "text": f"{advice}: yours is {_format(name, yours)}, {player['name']}'s is {_format(name, theirs)}."})
        if len(notes) == limit:
            break
    return notes


def _comparison(mine: dict, player: dict, features: dict) -> list[dict]:
    return [{"feature": name, "label": FEATURES[name][0], "unit": FEATURES[name][1],
             "yours": None if mine[name] is None else round(mine[name], 3),
             "theirs": player["features"].get(name), "weight": features[name]["weight"]} for name in features]


def match(reps: Sequence[dict], db: dict | None = None, detail: int = 5) -> dict | None:
    """Rank the database players by similarity to these reps. The ``detail`` closest carry
    what to change to look like them, the measurements side by side and their skeleton.
    None without reps or a database."""
    db = db if db is not None else load_db()
    if not reps or not db or not db.get("players"):
        return None
    mine, features = median_features(reps), db["features"]
    ranked = []
    for player in db["players"]:
        distance, _ = _distance(mine, player["features"], features)
        if distance is not None:
            ranked.append(({"name": player["name"], "distance": round(distance, 3),
                            "similarity": round(100 * math.exp(-distance ** 2 / 2)), "reps": player["reps"]}, player))
    if not ranked:
        return None
    ranked.sort(key=lambda entry: entry[0]["distance"])
    for entry, player in ranked[:detail]:
        entry.update(feedback=feedback(mine, player, features), features=_comparison(mine, player, features),
                     skeleton=player["skeleton"])
    widths = [r["shoulder_width"] for r in reps if r.get("shoulder_width") is not None]
    notes = []
    if widths and float(np.median(widths)) > MAX_SIDE_SHOULDER_WIDTH:
        notes.append("This clip is not a side view, and the player templates are. Film from the side, level with "
                     "the free-throw line, or the match mostly reflects the camera angle.")
    if len(reps) < 3:
        notes.append(f"Based on {len(reps)} free throw{'s' if len(reps) != 1 else ''}. Five or more in one clip "
                     "gives a steadier match.")
    consistency = []
    for name, info in features.items():
        spread = _spread([r["features"].get(name) for r in reps])
        if spread is not None:
            consistency.append({"feature": name, "label": FEATURES[name][0], "unit": FEATURES[name][1],
                                "yours": round(spread, 3), "players": info["within"]})
    return {"shot": db.get("shot", "free throw"), "reps": len(reps),
            "release_times_s": [r["release_s"] for r in reps], "matches": [entry for entry, _ in ranked],
            "consistency": consistency, "notes": notes,
            "skeleton": {"times": list(SKELETON_TIMES), "yours": median_skeleton(reps)}}
