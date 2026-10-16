# labeling toolkit

How to turn one NBA game into labeled 7 s clips for `eval/*.csv`. Read `eval/README.md`
once for what the columns mean. Run every command from the repo root. Each tool has `--help`.

```bash
L="uv run --inexact python scripts/labeling"
OCR="uv run --inexact --with pyobjc-framework-Vision --with pyobjc-framework-Quartz python scripts/labeling"
```

Measured on the pilot game (WAS@LAL 2026-03-30, 15 min video): find 3 s, download 52 s,
scorebug 38 s, match 46 s (cut scan; cached after), cut 10 s per clip, cuts 2 s, sheets 1-3 s.

## procedure

1. **Play-by-play.** `$L/pbp.py <event> > eval/staging/pbp/<event>.txt`. The first line gives the
   game tag for notes (`WAS@LAL 2026-03-30`, NBA abbreviations; ESPN's WSH/GS/NY/NO/SA/UTAH are
   already converted). Each shot row has a play number `#`, period, clock, score after the play
   (away-home), team, result, `dist`, `zone`, ESPN x/y, ESPN type and text. Rows starting `.` are
   nearby rebounds and fouls. `dist` is the stated feet; `*` marks a layup/dunk/tip with no stated
   feet, measured from the coordinates. `zone` `!` means ESPN's exact 2-or-3 call overrode the
   whole-foot coordinates on the arc.
2. **Find the video.** `$L/video.py find <event>`. Use the `BEST` id (NBA channel only). `NONE`: tell
   the coordinator and take another game.
3. **Download.** `$L/video.py download <yt id> eval/staging/video`. If it stalls, rerun it.
4. **Scorebug timeline.** `$OCR/scorebug.py eval/staging/video/<yt id>.mp4 --event <event>`. It prints
   one line per segment (a stretch with the bug on and the clock running continuously). Long gaps are
   replays, cutaways or footage without a bug.
5. **Match.** `$L/scorebug.py match eval/staging/scorebug/<yt id>.json <event> --covered-only > eval/staging/scorebug/<event>_match.txt`.
   For each play with footage: `time` (video seconds when the bug clock reads the ESPN clock), the
   cut-free span around it, whether the bug score agrees with ESPN, and a suggested `clip start`
   for a 7 s clip with the shot about 4 s in. `WARNING` means less than 7 s without a camera cut.
6. **Choose clips.** Aim for the number of clips the coordinator gave you. Prefer, in order: misses
   (we need far more misses than makes), rim attempts (layups, dunks, tips, putbacks, blocked
   shots), corner threes, both teams. Each clip: one continuous camera angle, 7.0 s, the shot 2-5 s
   in, ending at least 2 s after the ball reaches the rim. Skip replays (slow motion, or the bug
   missing) and clips with no free 7 s; 5-6 s is acceptable when the action fits. One or two clips
   with no shot (passes, dribbling, a scramble) per game are useful.
7. **Cut.** `$L/clips.py cut eval/staging/video/<yt id>.mp4 <start> 7.0` prints the new job id. The
   clip's time t is source time start + t.
8. **Check for cuts.** `$L/clips.py cuts <job>` must end `no camera cuts` (flash frames are fine).
   Otherwise look at the flagged frames, `rm -r data/jobs/<job>`, and cut again with a better start
   or a shorter duration.
9. **Look.** `$L/frames.py <job> --overview --out eval/draft-frames/<job>_overview.jpg` (a tile every
   0.25 s). Find every attempt in it. Then for each attempt:
   - release: `$L/frames.py <job> --at <t> --window 0.5 --out eval/draft-frames/<job>_release.jpg`
     (every frame; add `--crop x1,y1,x2,y2` in source pixels around the shooter to enlarge; the
     command prints the tile-to-source scale);
   - outcome: the same around the rim, `--step 2 --crop ...` around the rim, `..._rim.jpg`.
   Every tile is captioned `f<frame> <time>s`; those are the times to label (frame / 59.94). The
   release is the first frame where the ball has left the fingertips: note the last frame on the
   fingertips and the first free one. A make: the ball goes down through the net, the bug score
   changes, the shot clock resets. A miss: off the rim or glass and away, or blocked.
10. **Details.** `$L/pbp.py <event> --play <#>` prints distance, zone, shooter (`#23 LAL (LeBron
    James)`: add the jersey colour) and the exact start of the notes.
11. **Write rows** with `rows.py add` (never edit the CSV by hand), one call per attempt, into
    `eval/staging/<split>/<event>.csv`, where `<split>` is the one the coordinator gave you (dev ->
    `eval/labels.csv`, holdout3 -> `eval/holdout3.csv`, fixed -> `eval/fixed.csv`):
    ```bash
    $L/rows.py add eval/staging/dev/401810953.csv --job 1a2b3c4d5e6f --release 2.85 --outcome missed \
      --shot-type "jump shot" --distance 26 --zone above_break_three --shooter "#15 LAL gold (Austin Reaves)" \
      --source espn_pbp --notes "WAS@LAL 2026-03-30 ESPN event 401810953 Q1 6:45 '...' (Step Back Jump Shot; x=37 y=24), away-home 9-11. Trimmed from YouTube fED_h7XPWGY at 171.61s, duration 7.0s ..."
    ```
12. **Check.** `$L/rows.py check eval/staging/<split>/<event>.csv` must say OK.
13. **Clean up.** Once all clips are cut and checked: `rm eval/staging/video/<yt id>.mp4` (550 MB
    each; 30 agents share one disk). Delete jobs you abandoned. Leave the jobs you labeled.
14. **Commit** only after an independent second labeler has checked your rows:
    `$L/rows.py commit eval/staging/<split>/<event>.csv eval/<target>.csv -m "label six WAS@LAL 2026-03-30 clips"`.
    It waits for a lock, checks the rows, appends them, refuses job ids already in any `eval/*.csv`,
    commits only that CSV and prints the hash. Never `git add`/`git commit` label files by hand.

## conventions

- Label every attempt in a clip, by either team, including putbacks and tips that are not in the
  play-by-play. ESPN often splits one putback into miss + offensive rebound + make, or logs a tip
  you cannot see: label what the frames show and say so in the notes.
- `release_s`: when the ball leaves the fingertips, to 0.05 s.
- `outcome`: `made`, `missed` or `unclear`. A blocked shot is `missed`.
- `shot_type`: `jump shot`, `floater`, `layup`, `dunk` or `tip`. Leave it empty for hook shots (the
  existing convention) and when unsure. A catch-and-slam putback is a `dunk` even when ESPN says tip dunk.
- `distance_ft` comes from the play-by-play (`dist`), and only on the row of the play-by-play
  attempt; `zone` comes from the coordinates (`zone`); `source` is `espn_pbp` whenever
  `distance_ft` is set. Attempts not in the play-by-play get no distance and no source.
- `shooter`: `#<number> <TEAM> <jersey colour> (<name>)`.
- `notes` begin `AAA@BBB YYYY-MM-DD ESPN event <id> Q<period> <clock> '<pbp text>' (<ESPN type>; x=.. y=..), away-home <score>. Trimmed from YouTube <id> at <start>s, duration <d>s ...`,
  then the frame evidence for release and outcome with the contact sheet paths
  (`eval/draft-frames/<job>_release.jpg`), and end with `Confidence: release <high|medium|low>; outcome <...>.`
  The score is the one after the play, as `pbp.py` prints it.
- A clip with no shot is one row with only `job_id` and `notes`.
- Staging rows go to `eval/staging/<split>/<event_or_video_id>.csv` (`eval/staging/` is gitignored).
- Label from the frames before Ballform has seen the clip. Do not run Ballform on your clips.

## fixed-camera footage (non-NBA)

No play-by-play or scorebug: download with `video.py download`, find shots in
`frames.py --overview` sheets of the source cut into pieces, and cut with
`clips.py cut <source> <start> <dur> --keep-source-format --camera elevated` (or `courtside`).
Notes start `FIXED youtube:<11-char id> ...`; `distance_ft` only from a tape measure (`source` `tape`).

## gotchas

- ESPN coordinates: x 0-50 across the court with the rim at x=25, y feet toward half court with
  the rim at y=1 (fitted on 7139 shots: stated feet match the distance to (25, 1) with 0.38 ft mean
  error). Old notes with x=-134 y=211 are NBA-stats coordinates (tenths of a foot), not ESPN's.
- ESPN's clock is often 1-4 s off the broadcast clock, and the bug score can trail ESPN by a few
  seconds. `time` is a pointer: find the actual release in the frames.
- Bug segments are not camera-cut free: broadcasts cut to close-ups and replays while the bug stays
  on. `match` scans for cuts, but always run `clips.py cuts` on the finished clip.
- Flash photography makes single bright frames; `cuts` labels them `flash`. Mention a flash frame
  next to the release in the notes.
- Some stretches have no bug at all (the pilot video's first 148 s): their plays print `no footage`
  even though they are in the video. You may still use them by reading the arena clock above the
  backboard in the frames and the order of plays; say so in the notes.
- The clock stops after fouls and (late in periods) after makes, so several samples can show the
  play's clock; `time` is the first.
- The scorebug OCR needs the `--with pyobjc...` flags (`$OCR`); everything else runs in the venv. The
  first OCR run on a machine compiles Apple's models once (~30 s). OCR is cached next to the timeline.
- `dist` with `*` is from coordinates (rim attempts the text gives no feet for); use it, it is
  play-by-play data. Floaters and other shots with no stated feet get no distance.
