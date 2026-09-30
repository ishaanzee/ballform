"""Measure raw vs smoothed floor trajectories on calibrated game jobs.

Rebuilds each job's court map from its saved court.json (the per-frame
image-to-court homographies) and its players from observations.json, so no
video is re-analyzed. For every job it reports:

- the share of frame-to-frame steps faster than a human can run (SPEED_LIMIT),
  for the raw mapped floor points (what the shot metrics use) and for the
  smoothed trajectories (app/trajectory.py);
- a hold-out check: every third measured point is left out in turn and
  predicted from the rest;
- how far trajectories move when only the court calibration differs (jobs
  with identical observations);
- the shot metrics recomputed from saved data next to the saved result.json;
- candidate shooter and defender speeds over the 0.5 s before release, and
  how much they move under other reasonable smoother settings. These are
  measured here only; they are not added to reports (see README).

It also runs synthetic stops and cuts through a known camera to measure how
much the smoother blurs a sharp change in speed, and `--tune` grid-searches the
noise parameters by likelihood.

    uv run python scripts/measure_floor_trajectories.py [job ...] [--tune] [--json out.json]

Jobs default to every directory under data/jobs with a court.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import trajectory as T  # noqa: E402
from app.camera_motion import CourtMap, FrameMapping, follow_court  # noqa: E402
from app.court import Calibration, apply, fit, floor_point, template  # noqa: E402
from app.game import COURT_METRICS, add_court_metrics  # noqa: E402
from app.models import Detection, PoseFrame, ShotResult  # noqa: E402

# ft/s: about 20 mph (9 m/s), close to an elite sprinter's pace; no one reaches it between two
# frames of a half-court possession.
SPEED_LIMIT = 30.


def load_job(directory: Path):
    """(observations, player_frames, court_map, result) from a finished job's saved files."""
    obs = json.loads((directory / "observations.json").read_text())
    court = json.loads((directory / "court.json").read_text())
    result = json.loads((directory / "result.json").read_text()) if (directory / "result.json").exists() else None
    frames = []
    for f in obs["players"]:
        players = [PoseFrame(p["frame"], p["time_s"], {k: tuple(v) for k, v in p["landmarks"].items()},
                             p.get("track_id"), tuple(p["appearance"]) if p.get("appearance") else None,
                             tuple(p["box"]) if p.get("box") else None) for p in f["players"]]
        frames.append({"frame": f["frame"], "time_s": f["time_s"], "players": players})
    summary = court["summary"]
    mappings = {m["frame"]: FrameMapping(m["frame"], None if m["image_to_court"] is None
                                         else np.asarray(m["image_to_court"], float),
                                         m["status"], m["inliers"], m["inlier_ratio"], m["keyframe"])
                for m in court["frames"]}
    anchor = mappings[summary["anchor_frame"]].image_to_court
    ids = list(summary["per_point_error_px"])
    # Clicked pixels are not saved; only the landmark ids matter downstream (extrapolation_ft).
    calibration = Calibration(summary["standard"], anchor, np.linalg.inv(anchor), ids, np.zeros((len(ids), 2)),
                              list(summary["per_point_error_px"].values()), None, list(summary["dropped_outliers"]))
    court_map = CourtMap(calibration, mappings, summary["anchor_frame"], obs["width"], obs["height"])
    return obs, frames, court_map, result


def clip_id(directory: Path) -> str:
    """Short hash of the source video, or of the observations when the video was not kept."""
    source = directory / "input.mp4"
    data = source.read_bytes() if source.exists() else (directory / "observations.json").read_bytes()
    return ("video " if source.exists() else "obs ") + hashlib.md5(data).hexdigest()[:8]


def raw_steps(frames: list[dict], court_map) -> np.ndarray:
    """Speeds (ft/s) between consecutive analyzed frames of one track, from the unsmoothed floor points."""
    order = sorted(frames, key=lambda f: f["frame"])
    previous: dict[int, tuple] = {}
    speeds = []
    for index, frame in enumerate(order):
        current = {}
        for pose in frame["players"]:
            found = floor_point(pose, court_map.width, court_map.height) if pose.track_id is not None else None
            point = court_map.to_court(frame["frame"], found[0]) if found else None
            if point is not None:
                current[pose.track_id] = (index, frame["time_s"], point)
        for track_id, (i, t, point) in current.items():
            before = previous.get(track_id)
            if before is not None and before[0] == i - 1:
                speeds.append(math.dist(point, before[2]) / (t - before[1]))
        previous = current
    return np.asarray(speeds)


def smoothed_steps(trajectories) -> np.ndarray:
    return np.concatenate([np.linalg.norm(np.diff(t.position, axis=0), axis=1) / np.diff(t.times)
                           for t in trajectories if len(t.frames) > 1] or [np.empty(0)])


def holdout(frames: list[dict], court_map, cuts, folds: int = 3) -> dict:
    """Leave out every `folds`-th measured point in turn; error of the smoothed prediction there."""
    order = {f["frame"]: i for i, f in enumerate(sorted(frames, key=lambda f: f["frame"]))}
    by_frame = {f["frame"]: f for f in frames}
    errors, nis = [], []
    for fold in range(folds):
        exclude = {(p.track_id, f["frame"]) for f in frames for p in f["players"]
                   if p.track_id is not None and order[f["frame"]] % folds == fold}
        for t in T.floor_trajectories(frames, court_map, cuts, exclude=exclude):
            for i, frame in enumerate(t.frames):
                if (t.track_id, frame) not in exclude or t.source[i] != "-":
                    continue
                pose = next(p for p in by_frame[frame]["players"] if p.track_id == t.track_id)
                m = T.measure(pose, frame, by_frame[frame]["time_s"], court_map)
                if m is None:
                    continue
                e = m.court - t.position[i]
                errors.append(float(np.linalg.norm(e)))
                nis.append(float(e @ np.linalg.solve(t.position_cov[i] + m.cov, e)))
    errors, nis = np.asarray(errors), np.asarray(nis)
    return {"points": len(errors), "median_ft": float(np.median(errors)), "p90_ft": float(np.percentile(errors, 90)),
            "beyond_gate": float(np.mean(nis > T.GATE))}


SPEED_WINDOW_S = .5


def shot_speeds(trajectories, frames, game: dict, window: float = SPEED_WINDOW_S) -> tuple:
    """Candidate per-shot speeds, measured here only (not shipped): (shooter ft/s, defender closing ft/s).

    Shooter: mean smoothed floor speed over `window` s up to the release frame.
    Defender: mean of the part of the defender's smoothed floor velocity that
    points at the shooter over the same frames. None when a player has no
    trajectory through the window or fewer than half its frames were measured.
    """
    near = next((f for f in frames if f["frame"] == game["release_frame"]), None)
    if near is None:
        return None, None

    def window_of(track_id):
        t = next((t for t in trajectories if t.track_id == track_id and t.index(near["frame"]) is not None), None)
        if t is None:
            return None
        inside = [i for i in range(t.index(near["frame"]) + 1) if near["time_s"] - t.times[i] <= window + 1e-6]
        spacing = float(np.median(np.diff(t.times))) if len(t.times) > 1 else 0.
        if (near["time_s"] - t.times[inside[0]] < window - spacing - 1e-6
                or sum(t.source[i] in "aob" for i in inside) < .5 * len(inside)):
            return None
        return t, inside

    shooter = window_of(game["players"]["shooter_track_id"])
    if shooter is None:
        return None, None
    track, inside = shooter
    shooter_speed = round(float(np.mean(track.speed[inside])), 1)
    defender = window_of(game["players"]["defender_track_id"]) if game["players"]["defender_track_id"] else None
    if defender is None:
        return shooter_speed, None
    other, other_inside = defender
    toward = []
    for i in other_inside:
        j = track.index(other.frames[i])
        if j is not None and np.linalg.norm(gap := track.position[j] - other.position[i]) > 1e-6:
            toward.append(float(other.velocity[i] @ gap / np.linalg.norm(gap)))
    return shooter_speed, (round(float(np.mean(toward)), 1) if len(toward) >= .5 * len(other_inside) else None)


def shot_check(frames, court_map, obs, result, trajectories) -> list[dict]:
    """Recompute the court metrics from saved data and compare them with result.json."""
    rows = []
    balls = [Detection(**b) for b in obs["balls"]]
    for saved in (result or {}).get("shots", []):
        if not saved.get("game"):
            continue
        game = copy.deepcopy(saved["game"])
        game["evidence"] = [e for e in game["evidence"] if not e.startswith("Court:")]
        shot = ShotResult(saved["number"], saved["start_s"], saved["release_s"], saved["end_s"], saved["outcome"],
                          saved["outcome_confidence"], [], {}, game=game)
        add_court_metrics([shot], frames, balls, court_map)
        before = {k: saved["game"]["metrics"].get(k) for k in COURT_METRICS}
        after = {k: shot.game["metrics"].get(k) for k in COURT_METRICS}
        shooter_speed, closing = shot_speeds(trajectories, frames, saved["game"])
        rows.append({"shot": saved["number"], "same_as_saved": before == after,
                     "changed": {k: (before[k], after[k]) for k in COURT_METRICS if before[k] != after[k]},
                     "shooter_speed_ft_s": shooter_speed, "defender_closing_ft_s": closing})
    return rows


def measure_job(directory: Path) -> dict:
    obs, frames, court_map, result = load_job(directory)
    stats: dict = {}
    trajectories = T.floor_trajectories(frames, court_map, obs["cuts"], stats=stats)
    raw = raw_steps(frames, court_map)
    smooth = smoothed_steps(trajectories)
    covered = sum(len(t.frames) for t in trajectories)
    return {
        "job": directory.name, "clip": clip_id(directory),
        "raw": {"steps": len(raw), "over_limit": float(np.mean(raw > SPEED_LIMIT)),
                "p50": float(np.median(raw)), "p99": float(np.percentile(raw, 99)), "max": float(raw.max())},
        "smoothed": {"steps": len(smooth), "over_limit": float(np.mean(smooth > SPEED_LIMIT)),
                     "p50": float(np.median(smooth)), "p99": float(np.percentile(smooth, 99)),
                     "max": float(smooth.max()), "trajectories": len(trajectories), "frames": covered,
                     **{k: stats[k] for k in ("measurements", "accepted", "gated", "airborne", "splits")}},
        "holdout": holdout(frames, court_map, obs["cuts"]),
        "shots": shot_check(frames, court_map, obs, result, trajectories),
    }


def calibration_consistency(directories: list[Path]) -> list[dict]:
    """Jobs with identical observations but separate court calibrations: how far their trajectories differ.

    Differences come only from the calibration (and camera-motion chain), so
    they bound what a user's landmark clicks do to speeds and positions.
    """
    groups = defaultdict(list)
    for d in directories:
        groups[hashlib.md5((d / "observations.json").read_bytes()).hexdigest()].append(d)
    rows = []
    for members in groups.values():
        if len(members) < 2:
            continue
        runs = []
        for d in members:
            obs, frames, court_map, _ = load_job(d)
            runs.append({(t.track_id, f): (p, s) for t in T.floor_trajectories(frames, court_map, obs["cuts"])
                         for f, p, s in zip(t.frames, t.position, t.speed)})
        for d, run in zip(members[1:], runs[1:]):
            shared = runs[0].keys() & run.keys()
            speed = np.array([abs(runs[0][k][1] - run[k][1]) for k in shared])
            where = np.array([np.linalg.norm(runs[0][k][0] - run[k][0]) for k in shared])
            rows.append({"reference": members[0].name, "job": d.name, "track_frames": len(shared),
                         "speed_diff_median": float(np.median(speed)), "speed_diff_p95": float(np.percentile(speed, 95)),
                         "position_diff_median_ft": float(np.median(where))})
    return rows


# --- synthetic stops and cuts -------------------------------------------------------------------

def _synthetic_camera(position=(-100., 40., 35.), look_at=(0., 20., 0.), focal=3000., width=1920, height=1080):
    position, forward = np.asarray(position), np.asarray(look_at) - np.asarray(position)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0., 0., 1.))
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    k = np.array([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]])
    return k @ np.c_[rotation[:, 0], rotation[:, 1], -rotation @ position]


def synthetic_court_map(frames: int, width: int = 1920, height: int = 1080):
    court_to_image = _synthetic_camera(width=width, height=height)
    ids = ["lane_base_left", "lane_base_right", "ft_left", "ft_right", "three_top", "half_right", "half_left"]
    image = apply(court_to_image, [template("nba").landmarks[key] for key in ids])
    calibration = fit([(key, (x / width, y / height)) for key, (x, y) in zip(ids, image)], width, height)
    mappings = follow_court(None, list(range(frames)), 0, calibration.image_to_court, template("nba"), {}, [],
                            width, height, frames, fixed=True)
    return CourtMap(calibration, mappings, 0, width, height), court_to_image


def synthetic_player(path, court_to_image, fps=30., noise_px=0., seed=0, track_id=1, height_px=220.,
                     width=1920, height=1080, lift_px=None):
    """Player frames for a floor path [(x, y) per frame]; lift_px[i] raises the whole body (a jump)."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, spot in enumerate(path):
        u, v = apply(court_to_image, [spot])[0]
        v -= 0. if lift_px is None else lift_px[i]
        nu, nv = rng.normal(0, noise_px, 2) if noise_px else (0., 0.)
        h = height_px
        ankle_y = (v + nv) / height

        def at(dx, dy):
            return ((u + nu + dx * h) / width, ankle_y + dy * h / height, .95)
        landmarks = {"left_ankle": at(-.04, 0.), "right_ankle": at(.04, 0.), "left_knee": at(-.04, -.25),
                     "right_knee": at(.04, -.25), "left_hip": at(-.05, -.5), "right_hip": at(.05, -.5),
                     "left_shoulder": at(-.07, -.78), "right_shoulder": at(.07, -.78), "nose": at(0., -.92)}
        box = ((u - .15 * h) / width, (v - .97 * h) / height, (u + .15 * h) / width, (v + .03 * h) / height)
        frames.append({"frame": i, "time_s": i / fps,
                       "players": [PoseFrame(i, i / fps, landmarks, track_id=track_id, box=box)]})
    return frames


def _speed_profile(true_path, fps, noise_px, seeds=20):
    court_map, court_to_image = synthetic_court_map(len(true_path))
    speeds, errors = [], []
    for seed in range(seeds):
        frames = synthetic_player(true_path, court_to_image, fps=fps, noise_px=noise_px, seed=seed)
        (t,) = T.floor_trajectories(frames, court_map)
        speed = np.full(len(true_path), np.nan)
        speed[t.frames] = t.speed
        speeds.append(speed)
        errors.append(np.max(np.linalg.norm(t.position - np.asarray(true_path)[t.frames], axis=1)))
    return np.nanmean(speeds, axis=0), float(np.max(errors))


def synthetic_report(fps: float = 30., noise_px: float = 2.) -> dict:
    """How much a sharp stop and a sharp cut are blurred in time by the smoother."""
    n, stop = 90, 45
    run = 20.  # ft/s
    stop_path = [(-10. + run * min(i, stop) / fps, 25.) for i in range(n)]
    out = {}
    for label, noise in (("noise-free", 0.), (f"{noise_px:g} px noise", noise_px)):
        speed, max_error = _speed_profile(stop_path, fps, noise, seeds=1 if noise == 0 else 20)
        # Time the smoothed speed takes to go from 90% to 10% of the running speed.
        above = np.flatnonzero(speed >= .9 * run)
        below = np.flatnonzero(speed <= .1 * run)
        first_above_end = above[above < stop].max()
        first_below = below[below > first_above_end].min()
        window = slice(stop - round(.5 * fps), stop)
        out[f"stop, {label}"] = {
            "blur_90_to_10_s": round(float(first_below - first_above_end) / fps, 3),
            "starts_slowing_before_stop_s": round(float(stop - first_above_end) / fps, 3),
            "mean_speed_last_0.5s_before_stop": round(float(np.mean(speed[window])), 2),
            "true_mean_speed_last_0.5s": run,
            "max_position_error_ft": round(max_error, 2)}
    # A 90 degree cut at 15 ft/s: speed stays 15, heading turns instantly.
    cut_speed = 15.
    cut_path, x, y = [], -10., 20.
    for i in range(n):
        cut_path.append((x, y))
        if i < stop:
            x += cut_speed / fps
        else:
            y += cut_speed / fps
    for label, noise in (("noise-free", 0.), (f"{noise_px:g} px noise", noise_px)):
        speed, max_error = _speed_profile(cut_path, fps, noise, seeds=1 if noise == 0 else 20)
        out[f"90-degree cut, {label}"] = {"min_speed_at_cut": round(float(np.nanmin(speed[stop - 10:stop + 10])), 2),
                                          "true_speed": cut_speed, "max_position_error_ft": round(max_error, 2)}
    return out


# --- sensitivity of the shot speeds to the smoother's settings ---------------------------------

VARIANTS = {
    "default": {},
    "raised foot 0.2 torso": {"AIR_RISE_TORSO": .2},
    "raised foot 0.35 torso": {"AIR_RISE_TORSO": .35},
    "no hysteresis": {"AIR_EDGE_TORSO": 1e9},
    "no raised-foot skipping": {"AIR_RISE_TORSO": 1e9, "AIR_EDGE_TORSO": 1e9},
    "ACCEL_PSD 75": {"q": 75.},
    "ACCEL_PSD 300": {"q": 300.},
    "ankle noise x0.5": {"scale": .5},
    "ankle noise x2": {"scale": 2.},
}


def sensitivity(directory: Path) -> dict:
    """Candidate shot speeds of every shot under each variant of the smoother settings."""
    obs, frames, court_map, result = load_job(directory)
    out = {}
    for name, change in VARIANTS.items():
        saved = {k: getattr(T, k) for k in change if k.isupper()}
        try:
            for key, value in change.items():
                if key.isupper():
                    setattr(T, key, value)
            scale = change.get("scale", 1.)
            sigma = {k: (u * scale, v * scale) if k == "ankles" else (u, v) for k, (u, v) in T.PIXEL_SIGMA.items()}
            trajectories = T.floor_trajectories(frames, court_map, obs["cuts"], q=change.get("q", T.ACCEL_PSD),
                                                sigma=sigma)
        finally:
            for key, value in saved.items():
                setattr(T, key, value)
        out[name] = [shot_speeds(trajectories, frames, s["game"]) for s in (result or {}).get("shots", [])
                     if s.get("game")]
    return out


# --- tuning ----------------------------------------------------------------------------------

def tune(directories: list[Path]) -> list[tuple]:
    """Likelihood per measurement over a grid of ACCEL_PSD and a scale on the ankle pixel noise."""
    loaded = [load_job(d) for d in directories]
    rows = []
    for q in (50., 100., 150., 250., 400.):
        for scale in (.5, .75, 1., 1.5, 2.):
            sigma = {k: (u * scale, v * scale) if k == "ankles" else (u, v) for k, (u, v) in T.PIXEL_SIGMA.items()}
            ll = n = 0
            for obs, frames, court_map, _ in loaded:
                stats = {}
                T.floor_trajectories(frames, court_map, obs["cuts"], q=q, sigma=sigma, stats=stats)
                ll += stats["log_likelihood"]
                n += stats["measurements"]
            rows.append((q, scale, ll / n))
            print(f"  ACCEL_PSD={q:g} ankle noise x{scale:g}: log-likelihood per measurement {ll / n:.3f}", flush=True)
    return rows


def distinct_clips(directories: list[Path]) -> list[Path]:
    """One job per distinct clip, so a clip analyzed several times does not dominate."""
    seen, distinct = set(), []
    for d in directories:
        obs_hash = hashlib.md5((d / "observations.json").read_bytes()).hexdigest()
        clip = clip_id(d) if (d / "input.mp4").exists() else obs_hash
        if clip not in seen and obs_hash not in seen:
            seen.update({clip, obs_hash})
            distinct.append(d)
    return distinct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", nargs="*", help="job ids or directories (default: every calibrated job)")
    parser.add_argument("--tune", action="store_true", help="grid-search the noise parameters by likelihood")
    parser.add_argument("--json", type=Path, help="also write the measurements to this file")
    args = parser.parse_args()
    base = ROOT / "data" / "jobs"
    directories = [Path(j) if Path(j).is_dir() else base / j for j in args.jobs] or sorted(
        d for d in base.iterdir() if (d / "court.json").exists() and (d / "observations.json").exists())
    if args.tune:
        distinct = distinct_clips(directories)
        print(f"Tuning on {', '.join(d.name for d in distinct)}")
        tune(distinct)
        return
    jobs = [measure_job(d) for d in directories]
    print(f"Steps faster than {SPEED_LIMIT:g} ft/s between consecutive analyzed frames of one track\n")
    print("| job | clip | raw steps | raw > limit | raw p50 / p99 / max ft/s | smoothed steps | smoothed > limit "
          "| smoothed p50 / p99 / max ft/s | hold-out error median / p90 ft |")
    print("|---|---|---|---|---|---|---|---|---|")
    for j in jobs:
        r, s, h = j["raw"], j["smoothed"], j["holdout"]
        print(f"| {j['job']} | {j['clip']} | {r['steps']} | {r['over_limit']:.2%} | {r['p50']:.1f} / {r['p99']:.1f} / "
              f"{r['max']:.0f} | {s['steps']} | {s['over_limit']:.2%} | {s['p50']:.1f} / {s['p99']:.1f} / "
              f"{s['max']:.1f} | {h['median_ft']:.2f} / {h['p90_ft']:.2f} |")
    print("\nSmoother bookkeeping")
    for j in jobs:
        s = j["smoothed"]
        print(f"  {j['job']}: {s['trajectories']} trajectories over {s['frames']} track-frames; of {s['measurements']} "
              f"floor points {s['accepted']} used, {s['gated']} down-weighted as outliers; {s['airborne']} raised-foot "
              f"points skipped; {s['splits']} splits; hold-out points beyond the gate {j['holdout']['beyond_gate']:.1%}")
    print("\nSame observations, separate calibrations: smoothed trajectory differences")
    consistency = calibration_consistency(directories)
    for row in consistency:
        print(f"  {row['job']} vs {row['reference']}: {row['track_frames']} track-frames, speed differs by "
              f"{row['speed_diff_median']:.2f} ft/s median ({row['speed_diff_p95']:.2f} p95), position by "
              f"{row['position_diff_median_ft']:.2f} ft median")
    print("\nCourt metrics recomputed from saved data vs result.json; candidate speeds over the "
          f"{SPEED_WINDOW_S:g} s before release (not shipped)")
    for j in jobs:
        for shot in j["shots"]:
            print(f"  {j['job']} shot {shot['shot']}: court metrics "
                  f"{'identical' if shot['same_as_saved'] else 'CHANGED ' + str(shot['changed'])}; "
                  f"shooter {shot['shooter_speed_ft_s']} ft/s, defender closing {shot['defender_closing_ft_s']} ft/s")
    print("\nCandidate shot speeds (shooter / defender closing, ft/s) under other reasonable smoother settings")
    sensitivities = {}
    for d in directories:
        sensitivities[d.name] = sensitivity(d)
        cells = [f"{name}: " + "; ".join(f"{s}/{c}" for s, c in values)
                 for name, values in sensitivities[d.name].items()]
        print(f"  {d.name}: " + " | ".join(cells))
    print("\nSynthetic stop from 20 ft/s and 90-degree cut at 15 ft/s (30 fps, broadcast-like camera)")
    synthetic = synthetic_report()
    for key, value in synthetic.items():
        print(f"  {key}: {value}")
    if args.json:
        args.json.write_text(json.dumps({"speed_limit_ft_s": SPEED_LIMIT, "jobs": jobs, "synthetic": synthetic,
                                         "calibration_consistency": consistency, "sensitivity": sensitivities},
                                        indent=2, default=float))


if __name__ == "__main__":
    main()
