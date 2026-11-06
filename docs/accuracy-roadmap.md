# accuracy roadmap

What limits Ballform's accuracy, what changed on 2026-11-06, and what is most likely to help next. It comes from a diagnosis of every dev error with the code at `3031b36` (dev and fixed sets only; the holdouts were scored once afterwards, summary only). The per-error tables live in the night's scratch outputs; the counts below are the ones that matter for planning.

## where accuracy is lost

**Broadcast, 300 dev clips, 428 shots.**

- **Missed shots (55).** No single cause dominates. Blocks and strips that never reach the basket are 14–18 of them: dev recall is 57% on blocked shots against 90% on the rest, because an attempt is defined as a hand contact followed by the ball reaching the basket. A shooter or tipper with no usable pose is 9–10 (the detector usually boxes that player). The ball tracker locked onto a static false ball (crowd at the top edge, a referee, a stanchion) through the release in 6, while the detector scored the real ball at 0.53–0.87. Camera zoom or a fast pan breaks the image-space arc tests in 5. Rule thresholds account for 9–15. The detector truly missing a visible ball: 0–3.
- **False shots (48).** 40 come from the rim-attempt path. 18 are spurious ball perception, 13 of them an empty net boxed as the ball by the ball-in-basket class; that is 14 of the 25 false shots in the 16 fresh games. About 9 count one attempt twice or mistime it, 9 are a defender's touch (Ballform has no team concept), and 7 are a hand under a ball sitting on the rim.
- **Make/miss (36 unknown, 12 wrong).** Perception was adequate in 40 of 48 cases: the ball is seen on 88–100% of frames after it reaches the rim, and the rim on every frame in 41. The 2D decision layer is what fails. Three timing guards (follow-up, arrival, rattle) cause 20 of the 36 unknowns. Of the 6 misses called made, 2 are geometry margins, 2 are attribution (an unfound tip's crossing credited to the earlier shot), and 2 are ambiguous from the camera.
- **Shot type (61 wrong of 365).** Two boundaries hold 36: layup vs floater (18) and tip vs finish (18). 'Dunk' is never predicted.
- **Distance.** The large errors come from player tracking (a closing-out defender picked as the shooter, feet hidden behind a defender) and from finishes measured at their take-off. Against ESPN coordinates the calibration itself shows no systematic offset.

**Fixed cameras, 35 clips.** Perception is the bottleneck, the opposite of broadcast. The ball is seen on 68% of analyzed frames (40% on the 720p wall camera) against 84% on broadcast, the rim on 69% against 90%. All 4 missed shots and 7 of the 12 unknown outcomes come from the detector missing the ball or the rim. The detector is a third-party RF-DETR-M at 640×640, fine-tuned by its publisher on 465 images from 21 clips; a ball is about 11 px wide at that input.

**Evaluation.** The three blind holdouts pool to 233 shots: recall is known to about ±5 points pooled and ±8–10 per holdout. A paired before/after test needs about 400–1,250 shots to detect a 2-point change, 160–500 for 5 points. Replaying the pre-tuning code showed the 2026-10-15 tuning round kept only part of its dev gain on unseen games (about two thirds for recall, none for precision or make/miss), although leave-one-game-out had reported full transfer. Labels are accurate: 2 of 694 committed shot rows were ever corrected; shot type is the noisy field (3–5% ambiguous).

## what changed on 2026-11-06

Details and per-holdout numbers are in `docs/game-mode.md` (measured accuracy, 2026-11-06). In short:

- **A learned make/miss call** over the measurements the rules already make, where a rim was known at arrival (`app/outcome_model.py`, retrained with `scripts/train_outcome_model.py`). On the 16 fresh dev games, leave-one-game-out: 109 calls and 106 right with 2 misses called made, against the rules' 99, 92 and 4.
- **Two false-shot rules** on the rim-attempt path (a visible rise for contacts away from the rim; no lone ball-in-basket event while the rim is in view): dev false shots 48 → 36.
- **Finish rules**: "layup or dunk" instead of plain "layup", floaters landing within 5 ft are finishes, finishes with no landing seen are placed at the rim, off-court spots are withheld.
- **Rim**: short rim-track gaps bridged; a camera that follows play keeps the far hoop after a full-court pan; a fixed camera's steadily seen hoop fills the frames its track missed.
- **Opt-in**: `BALLFORM_STATIC_BALL_FILTER=1` (tracker locks on still false balls) and `BALLFORM_SHOT_VERIFIER=1` (learned shot check). Ball candidates are now saved per frame for later re-selection.
- **Evaluation**: Wilson intervals, a found-and-called-right metric, paired McNemar comparisons (`--compare`, `--compare-reports`) and `--summary-only` for holdouts.

Pooled over the three blind holdouts (233 shots, scored once): false shots 31 → 25, wrong make/miss calls 10 → 4 (misses called made 7 → 3), found-and-called-right 162 → 170, median distance error down on all three; shots found (190) and shot type (138) unchanged.

## what will work next, in order

1. **Fine-tune the detector on Ballform's own frames**, fixed cameras first, with hard negatives: empty nets, nets with a hand under them, rims. It is the largest lever on fixed cameras and the direct fix for the empty-net false shots on broadcast. It takes 2–5k box-labeled frames; pseudo-label them from confident tracks, trajectory interpolation and native-resolution re-detections, and have agents or people check contact sheets. The published model ships only as ONNX, so it means retraining RF-DETR-M (Apache-2.0) from COCO weights with the `rfdetr` package, on a rented GPU; keep the 640 input and the 11-output class schema so the MLX and Core ML paths stay drop-in. Native-resolution crops around the rim and the predicted ball (24–41% of missing ball frames recovered on fixed footage) are the cheaper first step.
2. **Make the static ball filter the default after one full rerun.** `BALLFORM_STATIC_BALL_FILTER=1` found all 6 tracker-lock misses and 2 more on the 56 dev clips rerun, losing none (dev recall 373 → 381, one more false shot and one more miss called made). Reruns are deterministic, so a full `--rerun` with the switch on decides it. Saved `ball_candidates` then let later replays re-select the ball offline.
3. **Play-by-play weak labels at scale.** ESPN's make/miss agrees with the hand labels on 347 of 348 dev shots and its mapped type on 97%. Time each shot from the scorebug's shot-clock reset, not ESPN's clock (release = reset − 1.64 s for jump shots, − 1.0 s for layups), keep isolated shots (no other attempt within 5 s), and drop games whose scorebug disagrees with ESPN. About 38 usable shots per game; 200 games give about 6,000. Use them for evaluation at ±1–2 points and as training data for the learned make/miss model, a rim-crop make/miss model and a type classifier. The NBA channel's uploads are highlight packages (82% of makes have footage, 17% of misses), so weight misses.
4. **A top-down pose pass** on unposed detector boxes near the ball at release. With an oracle wrist on the ball, 9 of the 10 pose-less misses are found; a ball-centred 2× crop through the current pose model found a wrist in 4 of 9.
5. **A rim-crop visual make/miss model** for the shots the learned model abstains on and for fixed cameras: a person can tell the outcome from a 4-rim-width crop for 1 s after arrival in 39 of 60 failure cases. It needs thousands of arrivals, which only weak labels supply.
6. **A team classifier.** 9 of 48 dev false shots are a defender's touch at the rim.
7. **Blocks.** 18 of 55 dev misses. They need an attempt definition that does not require the ball to reach the basket, and a learned verifier; 42 labeled blocks are too few to train one, and the detector's shot-block class fires on contested shots generally.
8. **Camera-compensated arcs** for shots taken during a zoom or fast pan (up to 4 dev misses), using the camera motion the court following already estimates.

## what will not help now

- **More broadcast dev labels for rule tuning.** The rules hold about one decision constant per tuned shot, and their dev gains transferred poorly.
- **Reading the scorebug at runtime.** The score changes 0.4–3.2 s after the ball goes through, after most trimmed clips end (seen for 27% of dev makes); a safe "no change means missed" rule would fix about 6 of 61 dev make/miss problems. It is a good offline labeler instead.
- **End-to-end video models or VLMs at runtime.** The best published NBA make/miss F1 is about 93, from a fine-tuned 7B video model; frontier VLMs score 14–30 F1 on extracting made/missed events. Cloud models would also break local-only processing. They are useful as labeling assistants.
- **A higher detector resolution on broadcast.** The ball's median radius is 17 px at 1080p, 720p broadcasts score as well as 1080p, and dropping 20% of ball detections at random costs about 1% of matched shots.

## what to label

- **Blind sets first.** Grow the pooled broadcast holdout to about 600 shots (holdout4 started on 2026-11-06 with games from the 2026-27 season). Score each new batch once, summary only, before anything is tuned on it: test, then train.
- **The product's own footage.** At least 150 blind fixed-camera or phone shots from at least 12 videos (fixed-holdout.csv started on 2026-11-06), the owner's own phone footage with tape-measured distances, and form-mode reps. 33 fixed shots from 4 videos behave like about 7 independent ones.
- **Detector boxes.** 2–5k frames, weighted toward fixed cameras, rim and release frames, and empty nets.
- **A written convention** for tip vs putback (hold time), catch-and-slam dunks and short floaters vs layups or jumpers, then adjudicate those rows; more labels under today's mixed convention will not move those boundaries.
- **Continuous footage** to measure false shots per minute: clips cut around shots hold 13 shots per minute, so their precision is optimistic.
