"""Follow a calibrated floor homography through camera pans and zooms.

Frame-to-frame motion comes from corner features on the floor only: the mask
is the court (plus apron) under the current mapping, minus every player box
(extended below the feet to cover reflections on a glossy floor) and minus
static broadcast graphics. Features are tracked from a keyframe with
Lucas-Kanade flow, checked forward-backward, and fitted with RANSAC; a new
keyframe is taken when too few survive. Chaining runs forward and backward
from the marked frame, as rim tracking does, and stops at cuts and on loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.court import (Calibration, Camera, Template, apply, camera_from_homography, floor_polygon,
                       project_lines, template)

WORK_WIDTH = 960
RANSAC_PX = 1.5            # in working-resolution pixels
FB_ERROR_PX = 1.0
RELIABLE_INLIERS, RELIABLE_RATIO = 40, .6
LOST_INLIERS = 15
REKEY_INLIERS = 80
APRON_FT = 6.              # floor beyond the lines (painted apron) that is still usable
MAX_STEP_FT = 6.           # image-centre movement on the floor between analyzed frames
BACKWARD_CHUNK = 90
LK = {"winSize": (21, 21), "maxLevel": 3,
      "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}


@dataclass
class FrameMapping:
    frame: int
    image_to_court: np.ndarray | None
    status: str  # anchor | tracked | fixed | weak | lost | cut
    inliers: int | None = None
    inlier_ratio: float | None = None
    keyframe: int | None = None

    @property
    def reliable(self) -> bool:
        return self.image_to_court is not None and self.status in {"anchor", "tracked", "fixed"}

    def to_dict(self) -> dict:
        return {"frame": self.frame, "status": self.status, "reliable": self.reliable,
                "inliers": self.inliers,
                "inlier_ratio": None if self.inlier_ratio is None else round(self.inlier_ratio, 3),
                "keyframe": self.keyframe,
                "image_to_court": None if self.image_to_court is None
                else [[float(v) for v in row] for row in self.image_to_court]}


@dataclass
class CourtMap:
    """A calibration and the per-frame floor mappings that follow it."""
    calibration: Calibration
    frames: dict[int, FrameMapping]
    anchor_frame: int
    width: int
    height: int

    @property
    def court(self) -> Template:
        return template(self.calibration.standard)

    def reliable(self, frame: int) -> FrameMapping | None:
        mapping = self.frames.get(frame)
        return mapping if mapping is not None and mapping.reliable else None

    def to_court(self, frame: int, pixel) -> tuple[float, float] | None:
        """Floor point in court feet for an image pixel on a reliably mapped frame."""
        mapping = self.reliable(frame)
        if mapping is None:
            return None
        point = apply(mapping.image_to_court, [pixel])[0]
        return (float(point[0]), float(point[1])) if np.all(np.isfinite(point)) else None

    def to_anchor_image(self, frame: int, pixel) -> tuple[float, float] | None:
        """The pixel with camera motion removed: where it falls in the marked frame's image.

        For a panning and zooming camera the frame-to-frame motion is the same
        for every depth, so this also steadies points above the floor.
        """
        mapping = self.reliable(frame)
        if mapping is None:
            return None
        point = apply(self.calibration.court_to_image @ mapping.image_to_court, [pixel])[0]
        return (float(point[0]), float(point[1])) if np.all(np.isfinite(point)) else None

    def camera(self, frame: int) -> Camera | None:
        mapping = self.reliable(frame)
        return None if mapping is None else camera_from_homography(
            np.linalg.inv(mapping.image_to_court), self.width, self.height)

    def limitation(self) -> str:
        summary = self.summary()
        text = (f"Court calibration ({summary['standard'].upper()}, {summary['points']} landmarks, "
                f"{summary['clicked_error_px']['rms']:.1f} px RMS error on the clicked points) is reliable on "
                f"{summary['reliable_frames']} of {summary['frames']} analyzed frames")
        if summary["reliable_frames"] < summary["frames"]:
            text += (". Floor measurements are withheld on the rest: the mapping stops at camera cuts and where too "
                     "few floor features could be tracked")
        return text + ". Check the court lines drawn in the review video before relying on distances in feet."

    def summary(self) -> dict:
        statuses: dict[str, int] = {}
        for mapping in self.frames.values():
            statuses[mapping.status] = statuses.get(mapping.status, 0) + 1
        reliable = sorted(f for f, m in self.frames.items() if m.reliable)
        camera = camera_from_homography(self.calibration.court_to_image, self.width, self.height)
        return {
            **self.calibration.summary(), "anchor_frame": self.anchor_frame,
            "frames": len(self.frames), "reliable_frames": len(reliable),
            "first_reliable_frame": reliable[0] if reliable else None,
            "last_reliable_frame": reliable[-1] if reliable else None,
            "frame_status_counts": statuses,
            "camera_at_anchor": None if camera is None else {
                "focal_px": round(camera.focal_px, 1),
                "position_ft": [round(float(v), 1) for v in camera.center]},
        }


def person_boxes(frame_data: dict) -> list[tuple[float, float, float, float]]:
    """Normalized boxes of every tracked person on one analyzed frame (poses, unposed and possession boxes)."""
    boxes = []
    for pose in frame_data.get("players", []):
        box = getattr(pose, "box", None)
        if box is None:
            points = [p for p in pose.landmarks.values() if p and p[2] >= .2]
            if not points:
                continue
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            height = max(ys) - min(ys)
            box = (min(xs), min(ys) - .25 * height, max(xs), max(ys) + .05 * height)
        boxes.append(tuple(box))
    for item in list(frame_data.get("unposed", [])) + list(frame_data.get("possession", [])):
        boxes.append(tuple(item[1]))
    return boxes


def segment_bounds(anchor: int, cuts: list[int], total: int) -> tuple[int, int]:
    """[start, end) of the camera shot containing the anchor frame."""
    start = max((c for c in cuts if c <= anchor), default=0)
    end = min((c for c in cuts if c > anchor), default=total)
    return start, end


class _Masker:
    def __init__(self, court: Template, boxes: dict[int, list], width: int, height: int,
                 scale: float, static: np.ndarray | None):
        self.court, self.boxes, self.scale, self.static = court, boxes, scale, static
        self.size = (round(width * scale), round(height * scale))
        self.width, self.height = width, height

    def _people(self, frame: int) -> list[tuple[float, float, float, float]]:
        w, h = self.size
        out = []
        for x1, y1, x2, y2 in self.boxes.get(frame, []):
            bw, bh = x2 - x1, y2 - y1
            # Pad the sides and extend below the feet: the floor mirrors the player.
            out.append(((x1 - .15 * bw) * w, (y1 - .05 * bh) * h, (x2 + .15 * bw) * w, (y2 + .5 * bh) * h))
        return out

    def mask(self, frame: int, image_to_court_scaled: np.ndarray) -> np.ndarray:
        w, h = self.size
        mask = np.zeros((h, w), np.uint8)
        polygon = floor_polygon(np.linalg.inv(image_to_court_scaled), self.court, APRON_FT)
        if polygon is None:
            return mask
        polygon = np.clip(polygon, -4 * w, 4 * w)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
        for x1, y1, x2, y2 in self._people(frame):
            cv2.rectangle(mask, (int(x1), int(y1)), (int(np.ceil(x2)), int(np.ceil(y2))), 0, -1)
        if self.static is not None:
            mask[self.static] = 0
        return mask

    def outside_people(self, frame: int, points: np.ndarray) -> np.ndarray:
        keep = np.ones(len(points), bool)
        for x1, y1, x2, y2 in self._people(frame):
            keep &= ~((points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2))
        return keep


def _features(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    points = cv2.goodFeaturesToTrack(gray, maxCorners=800, qualityLevel=.005, minDistance=7,
                                     blockSize=7, mask=mask)
    return np.empty((0, 2), np.float32) if points is None else points.reshape(-1, 2)


def _follow(sequence, anchor_frame: int, anchor_gray: np.ndarray, anchor_h: np.ndarray,
            masker: _Masker) -> dict[int, FrameMapping]:
    """Chain the scaled image->court homography along one direction of an ordered frame sequence."""
    out: dict[int, FrameMapping] = {}
    key_frame, key_h = anchor_frame, anchor_h
    key_pts = _features(anchor_gray, masker.mask(anchor_frame, anchor_h))
    current = key_pts.copy()
    previous, last_h = anchor_gray, anchor_h
    centre = np.array([[masker.size[0] / 2, masker.size[1] / 2]])
    lost = False
    for frame, gray in sequence:
        if lost:
            out[frame] = FrameMapping(frame, None, "lost")
            continue
        status = None
        if len(current) >= LOST_INLIERS:
            moved, ok, _ = cv2.calcOpticalFlowPyrLK(previous, gray, current.reshape(-1, 1, 2), None, **LK)
            back, ok_back, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None, **LK)
            moved, back = moved.reshape(-1, 2), back.reshape(-1, 2)
            keep = ((ok.ravel() == 1) & (ok_back.ravel() == 1)
                    & (np.linalg.norm(back - current, axis=1) < FB_ERROR_PX)
                    & (moved[:, 0] >= 0) & (moved[:, 1] >= 0)
                    & (moved[:, 0] < masker.size[0]) & (moved[:, 1] < masker.size[1]))
            keep &= masker.outside_people(frame, moved)
            key_pts, current = key_pts[keep], moved[keep]
        if len(current) < LOST_INLIERS:
            status = "lost"
        else:
            motion, inliers = cv2.findHomography(key_pts, current, cv2.RANSAC, RANSAC_PX)
            if motion is None or not np.all(np.isfinite(motion)):
                status = "lost"
            else:
                inliers = inliers.ravel().astype(bool)
                h = key_h @ np.linalg.inv(motion)
                step = float(np.linalg.norm(apply(h, centre) - apply(last_h, centre)))
                count, ratio = int(inliers.sum()), float(inliers.mean())
                if count < LOST_INLIERS or not step <= MAX_STEP_FT:
                    status = "lost"
        if status == "lost":
            out[frame] = FrameMapping(frame, None, "lost", keyframe=key_frame)
            lost = True
            continue
        reliable = count >= RELIABLE_INLIERS and ratio >= RELIABLE_RATIO
        out[frame] = FrameMapping(frame, h, "tracked" if reliable else "weak", count, ratio, key_frame)
        key_pts, current = key_pts[inliers], current[inliers]
        if count < REKEY_INLIERS:
            key_frame, key_h = frame, h
            key_pts = _features(gray, masker.mask(frame, h))
            current = key_pts.copy()
        previous, last_h = gray, h
    return out


def _gray(frame: np.ndarray, scale: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else gray


def _static_graphics(capture, start: int, end: int, scale: float) -> np.ndarray | None:
    """Textured pixels that stay identical while the scene moves: score bugs, logos and watermarks.

    Only textured pixels matter (flat regions yield no features), and a moving
    scene blurs out of the mean image while a static graphic stays sharp.
    """
    samples = []
    for frame in np.linspace(start, end - 1, num=min(10, max(2, end - start)), dtype=int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = capture.read()
        if ok:
            samples.append(_gray(image, scale).astype(np.float32))
    if len(samples) < 4:
        return None
    stack = np.stack(samples)
    mean = stack.mean(axis=0)
    textured = cv2.magnitude(cv2.Sobel(mean, cv2.CV_32F, 1, 0), cv2.Sobel(mean, cv2.CV_32F, 0, 1)) > 40
    static = textured & (stack.std(axis=0) < 2.5)
    # A camera that never moved leaves every textured pixel static; the mask would then erase the floor.
    if not textured.any() or static.sum() > .6 * textured.sum():
        return None
    return cv2.dilate(static.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


def follow_court(input_path: Path, frames: list[int], anchor_frame: int, anchor_h: np.ndarray,
                 court: Template, boxes: dict[int, list], cuts: list[int], width: int, height: int,
                 total: int, fixed: bool = False) -> dict[int, FrameMapping]:
    """Image->court homography (full-resolution pixels -> feet) for each analyzed frame.

    Fixed cameras keep the marked mapping. Frames in another camera shot than
    the anchor are marked "cut"; after a tracking failure a direction is "lost".
    """
    start, end = segment_bounds(anchor_frame, cuts, total)
    result = {f: FrameMapping(f, None, "cut") for f in frames if not start <= f < end}
    inside = sorted(f for f in frames if start <= f < end)
    if fixed:
        result.update({f: FrameMapping(f, anchor_h, "anchor" if f == anchor_frame else "fixed") for f in inside})
        return result
    scale = min(1., WORK_WIDTH / width)
    to_scaled = np.diag([scale, scale, 1.])
    anchor_scaled = anchor_h @ np.linalg.inv(to_scaled)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        result.update({f: FrameMapping(f, None, "lost") for f in inside})
        return result
    try:
        masker = _Masker(court, boxes, width, height, scale, _static_graphics(capture, start, end, scale))
        capture.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame)
        ok, image = capture.read()
        if not ok:
            result.update({f: FrameMapping(f, None, "lost") for f in inside})
            return result
        anchor_gray = _gray(image, scale)
        wanted_after = [f for f in inside if f > anchor_frame]
        wanted_before = [f for f in inside if f < anchor_frame]

        def forward():
            wanted, position = set(wanted_after), anchor_frame + 1
            last = max(wanted_after, default=anchor_frame)
            while position <= last:
                ok, image = capture.read()
                if not ok:
                    return
                if position in wanted:
                    yield position, _gray(image, scale)
                position += 1

        def backward():
            wanted = sorted(wanted_before, reverse=True)
            while wanted:
                chunk_end = wanted[0]
                chunk_start = max(start, chunk_end - BACKWARD_CHUNK + 1)
                capture.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
                grays = {}
                for position in range(chunk_start, chunk_end + 1):
                    ok, image = capture.read()
                    if not ok:
                        break
                    if position in wanted:
                        grays[position] = _gray(image, scale)
                chunk = [f for f in wanted if f >= chunk_start]
                for f in chunk:
                    if f not in grays:
                        return
                    yield f, grays[f]
                wanted = wanted[len(chunk):]

        mappings = _follow(forward(), anchor_frame, anchor_gray, anchor_scaled, masker)
        mappings.update(_follow(backward(), anchor_frame, anchor_gray, anchor_scaled, masker))
    finally:
        capture.release()
    if anchor_frame in frames:
        mappings[anchor_frame] = FrameMapping(anchor_frame, anchor_scaled, "anchor", keyframe=anchor_frame)
    for f in inside:
        mapping = mappings.get(f) or FrameMapping(f, None, "lost")
        if mapping.image_to_court is not None:
            mapping.image_to_court = mapping.image_to_court @ to_scaled
        result[f] = mapping
    return result


def anchor_frame(landmarks: dict, fps: float, total: int) -> int:
    """The source frame the landmarks were marked on (from its frame number or video time)."""
    frame = landmarks["frame"] if landmarks.get("frame") is not None else round(landmarks["time_s"] * fps)
    return min(max(0, int(frame)), max(0, total - 1))


def build_court_map(input_path: Path, calibration: Calibration, anchor: int, player_frames: list[dict],
                    cuts: list[int], width: int, height: int, total: int, fixed: bool) -> CourtMap:
    """Follow the calibration over the analyzed frames, masking the players seen on each."""
    boxes = {frame["frame"]: person_boxes(frame) for frame in player_frames}
    frames = follow_court(input_path, [frame["frame"] for frame in player_frames], anchor,
                          calibration.image_to_court, template(calibration.standard), boxes, cuts,
                          width, height, total, fixed=fixed)
    return CourtMap(calibration, frames, anchor, width, height)


def court_json(court_map: CourtMap) -> dict:
    return {"summary": court_map.summary(),
            "frames": [court_map.frames[f].to_dict() for f in sorted(court_map.frames)]}


def draw_court(frame: np.ndarray, court_map: CourtMap, frame_no: int) -> None:
    """Calibrated court lines, drawn only where the floor mapping is reliable."""
    mapping = court_map.reliable(frame_no)
    if mapping is None:
        return
    lines = project_lines(np.linalg.inv(mapping.image_to_court), court_map.court, frame.shape[1], frame.shape[0])
    cv2.polylines(frame, lines, False, (255, 210, 80), 1, cv2.LINE_AA)
