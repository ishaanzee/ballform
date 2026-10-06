# game mode: what it measures

The game pass uses multi-person pose detection, a basketball-trained detector, motion and jersey appearance to keep track of people. It chooses a shooter from recent hand/ball contact and release evidence, then chooses the nearest plausible opposing contest. Persistent IDs make the pre-release comparison less random, but crowded frames, similar jerseys, tiny players, officials, and occlusion can still break identity.

The shot-space score is deliberately transparent:

```text
65% separation score + 35% contest-clearance score
```

Separation is projected hip-to-hip distance in shooter torso lengths. Contest clearance is projected ball-to-defender-wrist distance in the same units. Separation change is reported as context, not secretly folded into the grade. It compares the release separation with the median over 0.3–0.7s before release. It relies on persistent player identities, so it needs both players tracked on at least half the frames through release. Each earlier frame is rescaled by the camera's zoom, estimated from every player seen in both frames. Frames zoomed by more than about 20% are skipped. One player crouching or turning no longer voids it, and neither does a camera pan. Each defender hand is measured on the frame nearest release, within 0.1s, where both that hand and the ball are visible. A hand briefly hidden on the release frame therefore no longer turns the contest into a range, and the confidence drops with the time offset. Only a hand unseen for that whole window leaves the score as a range with "other hand unknown". The score is a 0–100 review heuristic, not make probability, expected points, or a professional player grade. Missing evidence stays missing; it does not become a zero.

### which shots game mode finds

Each game shot is labelled jump shot, floater, layup, dunk, "layup or dunk" or tip, and its evidence says why (`app/shots.py`).

- **Jump shots** still need a ball arc, raised-hand ball contact at release and a flight well above the release shoulders. When the release itself is hidden, the basketball detector's jump-shot class plus a supported arc also counts. The release time is then the last visible hand contact, and the evidence says so.
- **Floaters** are arc shots where the detector's layup-dunk class is at least as confident as its jump-shot class around release.
- **Arc shots that reach the basket within 0.4 s of release are rim finishes**, not jump shots or floaters. From broadcast height a layup off the glass still draws a small arc, so the arc path finds it. On the labeled clips, jump shots took 1.05–1.38 s to reach the basket and floaters 0.50–0.65 s. Such a finish is a tip when it comes within 1 s of the previous shot reaching the basket, and otherwise "layup or dunk": the contacts around the rim are too noisy on broadcast footage to tell the two apart.
- **Layups, dunks, tips and putbacks** need a player's hand on the ball, then the ball reaching the basket within 1.2 s. "Reaching the basket" means a confident ball-in-basket detection, or the ball entering the rim's area when a rim exists. A dunk's last contact is at or above the basket. A tip is a raised-hand touch at the rim after the previous attempt got there. Without a rim or a basket detection, the detector's layup-dunk class plus the ball leaving the hands upward, above the head, gives "layup or dunk".
- **One attempt is one hand contact followed by the ball reaching the basket.** A ball rattling on the rim, or a rebound, has no new contact, so it belongs to the previous shot. A dribble has to leave the hands upward without bouncing. A pass that never reaches the basket is not a shot. A shot in flight claims its own arrival at the basket, so a hand touching it on the way (a contest, or a fan's hand the ball passes over in 2D) does not start a new attempt. A contact where the ball stays on one parabola through the "touch" is ignored for the same reason.
- **The shooter** for these attempts is the last player in unambiguous hand contact. It is checked against the decoded ball handler, and withheld when the two disagree (except for tips).
- **Ball-in-basket detections are never a make on their own.** Lone low-confidence detections also fired on an empty net. With a rim, the outcome uses the same rim-plane crossing and net-motion rules as jump shots. A ball-in-basket detection plus net motion can raise an otherwise-unknown rim attempt to "likely made".

**Scoring rim attempts.** Separation at the finish says little about a dunk, so layups, dunks and tips get a contest-only score: 100 × the contest-clearance component, measured at the last hand contact. Separation at the gather (the median over 0.3–0.7 s before that contact) is reported but not scored. These scores are labelled in the report and kept out of the mean shot-space score. Hidden-release jump shots use the normal score.

**Validation so far.** On the four one-jump-shot test clips, each still gives exactly its one jump shot, with the same release, shooter, defender, score and outcome. The new paths are covered by unit tests on synthetic tracks: layup, dunk, tip, putback, rim rattle, rebound, dribble, pass, mid-flight contest and hidden release. With the automatically detected rim, the rim-area path first found two false attempts on `2fcb`: a "dunk" and then a "tip". Both were the made jump shot falling past raised hands of fans behind the baseline. The flight and parabola rules above remove them. The labeled test set (`eval/`) now has 15 matched rim attempts and a few floaters from broadcast footage; on it, typing arc shots by flight time moved shot-vs-rim agreement from 10/26 to 21/26. That is still few examples per type, so treat the thresholds as lightly tuned. The parabola tolerance (2.5 ball radii) rests on only three real releases and two pass-overs.

### how make/miss is called

The outcome comes from the ball's path at the rim (`rim_outcome` in `app/scoring.py`); net motion only adds confidence. On broadcast angles a ball passing in front of or behind the rim, a rim-out and a back-rim bounce all cross the rim's plane inside the rim in the image, so a crossing alone is no longer a make:

- **Made** needs the ball centre to cross the rim plane downward within 1 s of the arc's apex (labeled makes took 0.08–0.78 s), at least a quarter of the rim width in from either edge (a ball through the hoop is a ball radius, 0.26 rim widths, inside it; labeled makes crossed at 0.34–0.69), and no bounce back above the rim in the next 0.6 s.
- **Missed** is a crossing outside the rim; a crossing that comes back above the rim (two confident detections over the rim box within 0.6 s); a ball that comes down to the rim without crossing it and rises two rim-box heights, back above the rim; or, after no crossing on arrival, a later drop outside the rim.
- **Unknown** is a crossing over the rim's edge that is not seen bouncing out, and a clean drop through the rim more than 1 s after the apex: that is a rattle-in, a rebound or a putback, and the putback is its own attempt. Before this, a miss that was put back took the putback's crossing as its make.

On the dev labels (`eval/labels.csv`, 76 matched shots) this moved misses called made from 13 to 0 and misses called missed from 10 to 25, with makes called made unchanged at 39 of 45; outcome coverage went from 82% to 84%. The six makes left unknown have the ball hidden at the rim or no rim found. The 0.25 edge margin is physically set but sits close to one labeled miss (crossing at 0.23), so treat it as lightly tuned.

Form mode reports 2D image-plane estimates such as release timing, launch angle, elbow angle, and upper-arm elevation. These are good for comparing your own reps from the same setup. They are not calibrated 3D biomechanics.

