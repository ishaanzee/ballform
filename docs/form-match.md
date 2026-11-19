# Free-throw player match

Form-match mode measures your free throws from a side-on clip, finds the NBA players whose free-throw form is closest, and says what to change to look more like one of them. It describes resemblance, not quality: shooting like a player is not evidence of shooting better.

## filming

- One shooter, free throws only, five or more in one clip.
- Side-on, level with the free-throw line, from the side opposite your shooting hand (from the left of a right-hander). Whole body in frame from before the ball comes up until the follow-through drops.
- 60 fps if the phone has it. Keep the camera still and level, with a wall, post or door frame somewhere behind you: their upright edges are how a tilted phone is noticed.

## what is measured

`app/form_signature.py` finds each free throw from the shooting arm alone: the wrist rises above the nose while the elbow opens from its most folded angle with the wrist above the shoulder (the set point) to its straightest (the release). No ball or rim is needed, so the same code measures a phone clip and a broadcast crop.

| measurement | meaning |
|---|---|
| elbow bend at the set point | elbow angle where the push starts |
| set point height / distance in front | wrist position relative to the nose, in torso lengths |
| elbow extension, release height, arm angle at release | the arm at its straightest |
| set point to release, waist to release | tempo |
| deepest knee bend, knee bend before release | legs and their timing |
| forward lean at release, follow-through hold | trunk and finish |

Angles are in the image plane. Lengths are in torso lengths (mid-shoulder to mid-hip), so a shooter's size and the zoom do not matter.

## camera tilt

Forward lean and arm angle are measured against the vertical, so a phone tilted 3° shifts both by 3°. The shooter cannot be the reference, because how upright the shooter stands is one of the measurements. `app/camera_tilt.py` reads the tilt from the scene instead: long straight edges within 20° of the image's vertical, in eight frames spread over the clip, leaving out the shooter and everything below the hips (floor lines running away from the camera look upright without being so). A camera pointing up or down makes upright edges lean together or apart across the frame, so their lean is fitted as a line across the frame and read at the centre. The poses are turned by that angle before anything is measured.

No tilt is reported, and nothing is corrected, unless the edges settle it: at least half of them must agree within 1.5°, from three or more places across the frame, on both sides of the centre or near it. The results then say that tilt could not be checked.

Checked on ten fixed-camera clips and one phone clip by rotating whole frames 3° and 6° each way: the reading moved by the applied angle to within 0.5° on all nine clips that gave a reading, and two clips of the same camera agreed within 0.2°. Steep overhead views and broadcast game footage gave no reading. The true tilt of those cameras is not known, so this shows that a change in tilt is measured, not that the zero is exact. The templates are not corrected: broadcast cameras are assumed level.

## the player templates

`app/form_db.json` holds, for each player, the median of each measurement over his free throws, his shot-to-shot spread, and a median skeleton through the release. It contains numbers only; no footage is stored.

The templates come from broadcast free throws filmed by the main sideline camera, which sees the shooter in profile. `scripts/formdb/build.py scan` finds them in a video of one player's free throws: a pose model looks three times a second for a standing person with a raised arm who stood still for the second before, then MediaPipe follows that person on an upscaled crop. Views from behind or at an angle (shoulder width over 0.35 torso lengths) are dropped, and every kept rep is checked by eye on a contact sheet (`build.py sheet`).

There are two sets of templates, one per camera side. The sideline camera sees a shooter from his right at one basket and from his left at the other, and the same free throw measures differently from the two. Your clip is compared only with templates from the same side.

Almost all templates have the shooting arm on the far side of the body from the camera: 27 players, against 4 on the near side. At the basket where the camera is on a right-hander's shooting-hand side, it sees him partly from behind (shoulder width 0.37 to 0.51 torso lengths), and those free throws are dropped as not side-on. That is why you film from the side opposite your shooting hand.

A player needs three checked free throws on a side to get a template. Many players have none: their home broadcasts show free throws from behind or from too far away, and highlight videos often cut in after the release.

## matching

Each measurement is weighted by the share of its variance that lies between players rather than between one player's own free throws. On the far-arm side, knee bend (0.88), forward lean (0.81), set point distance in front of the face (0.76) and knee timing (0.73) separate players best; elbow extension at release (0.44) varies more within a player than between players and counts least. The distance to a player is the weighted root-mean-square difference in between-player standard deviations, and similarity is `100 · exp(−distance² / 2)`.

As a check, each template free throw is left out of its own player's template and matched against all players. On the far-arm side the right player comes first for 112 of 156 free throws and in the first three for 133, where chance is 6; on the near-arm side, with four players, 12 of 15.

Feedback lists the largest weighted differences from the chosen player, leaving out any difference inside that player's own shot-to-shot spread or below 5°, 0.1 torso lengths or 0.08 s.

## comparing poses

The results panel plays your median skeleton and the chosen player's through the release, side by side or overlaid on the same feet. Skeletons are drawn relative to the feet in each frame, so a panning or shaking camera does not slide the body; the match adds a note when the feet moved more than 0.3 torso lengths in the frame before a release. Above the review video, "Draw the player's pose over my video" draws that player's skeleton on your footage at each free throw: following your feet frame by frame, scaled to your torso length, facing your way, turned by your camera's tilt. The player's limbs are in his proportions, not yours, so joints line up only where the two forms and builds agree.

## limits

- The templates are broadcast footage: an elevated camera 20 to 30 degrees off a true side view. A phone at chest height directly side-on sees slightly different angles. This offset has not been measured.
- Camera tilt is corrected only when the scene has upright edges to read it from, and to about half a degree. A lens that bends straight lines (an action camera) is not allowed for.
- 2D only. Elbow flare, hand placement and wrist action are not seen from the side.
- Broadcast video is 25 to 60 fps, so timing is known to a few hundredths of a second.
- A player with few reps, or reps from one game, has a less certain template. Templates have 3 to 14 free throws.
- A free throw the pose model reads badly is dropped when the elbow opens faster than an arm can (under 0.08 s); other misreadings can remain in a template.
