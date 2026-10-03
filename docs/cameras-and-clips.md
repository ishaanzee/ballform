# cameras and clips

## clip tips

For form work, keep the shooting arm, ball, feet, rim, and net visible. A side view is best for release and arc; a rear-oblique view is better for alignment. 60 fps and 1080p are a good target. A phone three to six feet high is usually enough. Do not digitally zoom halfway through the possession.

For game footage, short continuous half-court possessions are the sweet spot. NBA skycam and elevated pickup clips are supported, but replays, cuts, graphics, extreme zooms, and a rim that disappears behind the broadcast edit are still hard problems. A five-on-five clip is fine even when the UI says 1-on-1; the analyzer still tries to identify the ball carrier and the primary contesting defender.

Moving-broadcast game footage needs no court marking. The basketball detector labels players, referees and everyone else, and each detected player or referee box can vouch for only one pose. A spectator or duplicate pose overlapping a real player therefore cannot pass as a player. A playing-area polygon is optional and only offered for fixed cameras (elevated or courtside). If you draw one, keep it convex and cover the playable area, not the benches. A foot within about a body width of the edge still counts, so a player standing on the line is kept. Moving-camera analyses ignore a polygon: it is fixed in image coordinates, so it cannot follow a pan, and on test clips it cut a player standing on the sideline in roughly 10% of player detections. People standing just off the court can still pass as players when the detector calls them players, for example a bench player in warmups on the baseline.

## camera choices

There are three camera profiles. `Stationary courtside` is the default.

`Stationary courtside` is the cleanest choice when you want make/miss.

The rim no longer needs marking. In game mode, the basketball detector's rim class finds a snug box around the hoop ring on every frame, whatever the camera profile. The frames are linked into one hoop per camera segment, one-off false detections are dropped, and jitter is smoothed with a local line fit. It restarts after camera cuts. On four test clips it found the rim on every game frame. It called all four shots made, which close-ups of the ball passing through the net confirm; three of them previously had no outcome at all. Form mode samples about 15 frames and uses the median box. Dragging a box in the preview still overrides the detection: it is fixed on stationary cameras and tracked from that frame on a moving one. Reports say whether the rim was `detected` or `marked` under `diagnostics.rim_source`. Two limits: if both hoops are visible, the one detected more strongly is used, so mark the rim if the wrong hoop is chosen; and the form-mode sampling has not yet been tested on a stationary single-shooter clip.

`Pickup / elevated wide view` is for a fixed, wide, elevated camera. It tracks players and the ball, and reports make/miss when the rim is detected or marked.

`Moving broadcast + tracked rim` is the profile for NBA and other broadcast footage, meaning a continuous pan or moderate zoom. The detected rim follows the hoop through pans, zooms and cuts. It can score the rim crossing, use net motion as supporting evidence, and place the green make pulse over the moving hoop. If you mark the rim instead, a local CSRT tracker follows your box forward and backward from that frame; a hard cut, a lost track or an implausible jump then ends outcome scoring rather than producing a confident-looking lie.

The make classifier wants a visible downward crossing through the rim. If the ball vanishes at the hoop, it can call a **likely make** only when the descending path projects through the rim and localized net motion arrives afterward. Net movement by itself never turns an airball into a make. Green animation means the analyzer found a verified or likely make; it is not a broadcast replay graphic.

