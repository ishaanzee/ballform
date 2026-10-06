# labeled test set

Hand labels for scoring Ballform: does it find each shot, call make/miss right, name the shot type, and measure the distance. `ballform-eval` compares finished jobs against `labels.csv`.

## which clips

Aim for about 50 shots across 30–40 clips. The mix matters more than the count:

| clips | camera profile | what they test | distance truth |
| --- | --- | --- | --- |
| ~20 broadcast possessions, from full-game or condensed-game video | moving broadcast, court calibrated | detection, make/miss, shot type, feet | NBA.com play-by-play |
| ~10 of your own pickup or practice shots, fixed camera | stationary courtside or elevated, court calibrated | the same on non-broadcast footage | tape measure |
| ~8 form clips of your own shooting | stationary courtside, form mode | release detection and make/miss | none |
| 3–5 clips with no shot: passes, dribbling, a rebound scramble | any | false shots | none |

- **Do not use highlight reels.** They are nearly all makes, so they cannot show whether a miss is ever called a make. Get at least 15 misses in total.
- **Get rim attempts.** Layups, dunks, floaters and putbacks have never been checked on real footage. Include at least 8 of them.
- Add a few corner threes and a couple of possessions where both hoops are visible.
- **Trim before uploading.** Keep one continuous possession of 5–15 s that ends about 2 s after the ball reaches the rim, with no replays or cuts. Labels are times in the uploaded file, so label the trimmed clip.
- **Distance from play-by-play.** NBA.com play-by-play gives each shot's distance in feet ("26' 3PT Pull-Up Jump Shot"), measured to the basket like Ballform's `shot_distance_ft`. It is rounded to the foot, so errors under about 1 ft mean nothing. The two-or-three call there is exact.
- **Distance from a tape measure.** For your own footage, measure from the spot on the floor directly under the rim's centre (5 ft 3 in from the baseline on NBA, NCAA and high-school courts) to the shooter's feet. Chalk or tape marks at known distances before you film make this easy.

## how to label a clip

1. Run the trimmed clip through the app as you normally would: game mode, the right camera profile, and court calibration with 5 or more landmarks when you want distances. The job saves its settings to `data/jobs/<job id>/settings.json`, so it can be rerun the same way later.
2. Copy the job id (the folder name under `data/jobs/`).
3. **Label from the original clip, before looking at Ballform's output.** Otherwise you will anchor on its answer. Step through it frame by frame in a player that shows milliseconds, such as IINA, mpv or VLC.
4. Add one row per shot attempt to `labels.csv`:

| column | value |
| --- | --- |
| `job_id` | the job folder name |
| `release_s` | seconds into the clip when the ball leaves the fingertips; 0.1 s precision is plenty. Leave empty for a clip with no shot (one row per such clip). |
| `outcome` | `made`, `missed`, or `unclear` when you cannot tell (for example the video cuts away). A blocked shot is `missed`. |
| `shot_type` | `jump shot`, `floater`, `layup`, `dunk` or `tip`; empty if unsure |
| `distance_ft` | from play-by-play or a tape; leave empty rather than guess |
| `zone` | `paint`, `midrange`, `corner_three` or `above_break_three`; empty if unsure |
| `shooter` | who shot, e.g. `#11 NYK white`, so you can check the shooter by eye |
| `source` | where the distance came from: `nba_pbp`, `tape` |
| `notes` | anything odd: screen, heavy occlusion, camera zoom |

Label every attempt in the clip, including tips and putbacks. A rim rattle or rebound is part of the same attempt.

Example:

```csv
job_id,release_s,outcome,shot_type,distance_ft,zone,shooter,source,notes
0a1b2c3d4e5f,6.8,made,jump shot,25,above_break_three,#11 NYK,nba_pbp,
a1b2c3d4e5f6,3.2,missed,layup,,paint,#7 white,,contested at the rim
a1b2c3d4e5f6,4.1,made,tip,,paint,#23 dark,,putback
f0e1d2c3b4a5,,,,,,,,no shot: passes around the arc
```

## keep some clips back

Once thresholds are tuned against these clips, they stop being an honest test. Put about a third of the rows in `holdout.csv` (same columns) and only score it occasionally, without tuning against its per-shot errors.

`holdout2.csv` is a blind second holdout made on 2026-10-06, after `holdout.csv` had been scored once following a day of tuning: 88 rows (82 attempts, 6 no-shot clips) in 54 broadcast clips from ten 2025-26 games in neither other file (GSW@SAS 11-12, CHI@POR 11-19, MIA@DAL 12-03, PHX@GSW 12-20, BOS@IND 01-12, MIN@SAS 01-17, MEM@DEN 02-11, ORL@LAL 02-24, DEN@UTA 03-02, HOU@MIN 03-25), labeled before any Ballform output existed for them. Nothing has been tuned against it; score it rarely and never tune against its per-shot errors.

## running it

```bash
uv run --inexact ballform-eval                       # score the saved job results
uv run --inexact ballform-eval --rerun               # re-analyze every labeled clip with the current code
uv run --inexact ballform-eval --labels eval/holdout.csv
uv run --inexact ballform-eval --replay               # rescore the last rerun's measurements (no inference)
```

`--rerun` writes to `eval/runs/` and leaves the jobs untouched. It also saves each clip's frame-loop measurements (`scoring_inputs.pkl.gz`), and `--replay` rescores those with the current scoring code in seconds, in parallel, without running any model. Replay is valid for changes after inference (shots, outcomes, game and court metrics); rerun after changing detection, pose or tracking. Court landmarks in a job's `settings.json` are refit on every replay. Jobs without them use the automatic calibration in `court_landmarks.json`, made by `scripts/auto_court_landmarks.py` (run it for newly labeled clips; `--labels eval/holdout.csv` for the holdout), so distance and zone are scored without hand calibration; see `docs/court-calibration.md`. Each run prints a summary and per-shot lines, and writes `eval/report.json`. Copy that report to `eval/baseline-<date>.json` before changing code, so you can compare afterwards. Jobs made before `settings.json` existed are rerun without their rim or court calibration, and the script says so.

What the summary means:

- **detection:** recall is labeled shots found; precision is predictions that were real shots. Shots match when their release times are within 0.75 s.
- **outcome:** coverage is how often Ballform made a call instead of "unknown"; accuracy is scored only on those calls, and `likely made` counts as a make. `misses_called_made` is the number to watch.
- **shot_type:** accuracy scores the five types exactly (`layup or dunk` counts for either). `shot_vs_rim` scores only the coarse split: rim finishes (layup, dunk, tip) against shots (jump shot, floater). Keep labeling the fine type; the coarse split is what is worth reading until each type has enough examples.
- **distance_ft:** mean and median absolute error, and the bias (positive means Ballform measures too long), on shots where both a label and a measurement exist.
