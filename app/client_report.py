"""Customer-facing report: one standalone, printable HTML page per job.

The results page is a review tool and shows model names, timings and raw
evidence.  This page is what a player, parent or coach receives: plain-language
metrics, a release snapshot per shot and the limitations, with nothing that
needs the local server.  Print it to PDF from the browser and send it with
the annotated video.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import statistics
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
THUMB_WIDTH = 640

# Plain-language label and unit for each metric shown to customers.  Metrics
# not listed here (model internals such as defender selection margin) are left
# out of the customer report.
FORM_METRICS = {
    "elbow_angle_at_release_deg": ("Elbow angle at release", "°"),
    "set_point_elbow_angle_deg": ("Elbow angle at set point", "°"),
    "upper_arm_elevation_deg": ("Upper-arm lift", "°"),
    "wrist_over_elbow_pct_shoulder_width": ("Wrist over elbow offset", "% of shoulder width"),
    "release_height_body_ratio": ("Release height", "× body height"),
    "follow_through_extension_deg": ("Follow-through extension", "°"),
    "release_angle_2d_deg": ("Launch angle (camera view)", "°"),
}
# Game metrics in feet (court calibrated) take priority over the torso-length
# versions of the same measurement.
GAME_METRICS = [
    ("shot_distance_ft", None, "Shot distance"),
    ("separation_ft", "separation_torso", "Space from defender"),
    ("contest_clearance_ft", "contest_clearance_torso", "Defender hand to ball"),
    ("visible_hand_clearance_ft", "visible_hand_clearance_torso", "Visible defender hand to ball"),
    (None, "separation_change_torso", "Space gained before the shot"),
    (None, "gather_separation_torso", "Space at the gather"),
]
ZONES = {
    "paint": "Paint",
    "midrange": "Midrange",
    "corner_three": "Corner three",
    "above_break_three": "Above-the-break three",
}
OUTCOMES = {"made": "Made", "likely made": "Likely made", "missed": "Missed", "unknown": "Not called"}


def _esc(value) -> str:
    return html.escape(str(value))


def _num(value) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def _with_unit(value, unit: str) -> str:
    if unit in {"°", "%"}:
        return f"{_num(value)}{unit}"
    return f"{_num(value)} {unit}"


def make_miss_available(result: dict) -> bool:
    """Make/miss needs a rim.  Reports from before automatic rim detection have
    no rim_source and keep their recorded outcomes."""
    diagnostics = result.get("diagnostics") or {}
    return "rim_source" not in diagnostics or diagnostics["rim_source"] is not None


def release_thumbnails(video: Path, times_s: list[float], width: int = THUMB_WIDTH) -> list[str | None]:
    """JPEG data URIs of the annotated video at each time, or None where the frame can't be read."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return [None] * len(times_s)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    thumbs: list[str | None] = []
    try:
        for time_s in times_s:
            frame_index = max(0, round(float(time_s) * fps))
            if count:
                frame_index = min(frame_index, count - 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                thumbs.append(None)
                continue
            height = round(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            thumbs.append(f"data:image/jpeg;base64,{base64.b64encode(jpeg.tobytes()).decode()}" if ok else None)
    finally:
        capture.release()
    return thumbs


def form_consistency(shots: list[dict]) -> list[tuple[str, str, str, str]]:
    """(label, average, lowest, highest) for each form metric measured on two or more shots."""
    rows = []
    for key, (label, unit) in FORM_METRICS.items():
        values = [s["metrics"][key] for s in shots
                  if isinstance((s.get("metrics") or {}).get(key), (int, float))]
        if len(values) < 2:
            continue
        rows.append((label, _with_unit(float(statistics.fmean(values)), unit),
                     _with_unit(float(min(values)), unit), _with_unit(float(max(values)), unit)))
    return rows


def _game_metric_rows(game: dict) -> list[tuple[str, str]]:
    metrics = game.get("metrics") or {}
    rows = []
    for feet_key, torso_key, label in GAME_METRICS:
        if feet_key and metrics.get(feet_key) is not None:
            rows.append((label, _with_unit(metrics[feet_key], "ft")))
        elif torso_key and metrics.get(torso_key) is not None:
            rows.append((label, _with_unit(metrics[torso_key], "torso lengths")))
    if metrics.get("shot_zone"):
        rows.insert(1 if rows and rows[0][0] == "Shot distance" else 0,
                    ("Zone", ZONES.get(metrics["shot_zone"], str(metrics["shot_zone"]).replace("_", " "))))
    return rows


def _score_text(game: dict | None) -> str:
    if not game:
        return "Not scored"
    if game.get("score") is not None:
        return f"{_num(game['score'])} / 100"
    if game.get("score_range"):
        return f"{_num(game['score_range']['lower'])}–{_num(game['score_range']['upper'])} / 100"
    return "Not scored"


def _outcome(shot: dict, rim: bool) -> tuple[str, str]:
    if not rim:
        return "Not measured", "unknown"
    outcome = shot.get("outcome", "unknown")
    css = "missed" if "miss" in outcome else "unknown" if outcome == "unknown" else "made"
    return OUTCOMES.get(outcome, outcome.capitalize()), css


def _shot_card(shot: dict, thumb: str | None, game_mode: bool, rim: bool) -> str:
    outcome, css = _outcome(shot, rim)
    title = f"Shot {_esc(shot['number'])}"
    if shot.get("shot_type"):
        title += f" · {_esc(shot['shot_type'])}"
    rows: list[tuple[str, str]] = []
    if game_mode:
        game = shot.get("game")
        score_label = "Contest score (rim attempt)" if game and game.get("score_basis") else "Shot-space score"
        rows.append((score_label, _score_text(game)))
        if game:
            rows += _game_metric_rows(game)
    for key, (label, unit) in FORM_METRICS.items():
        value = (shot.get("metrics") or {}).get(key)
        if value is not None:
            rows.append((label, _with_unit(value, unit)))
    metrics = "".join(f"<div><dt>{_esc(k)}</dt><dd>{_esc(v)}</dd></div>" for k, v in rows)
    image = (f'<img src="{thumb}" alt="Annotated frame at the release of shot {_esc(shot["number"])}">'
             if thumb else "")
    cues = "".join(f"<li>{_esc(c)}</li>" for c in shot.get("cues") or [])
    return (f'<article class="shot"><header><h3>{title}</h3><span class="time">{_num(shot["release_s"])} s</span>'
            f'<span class="pill {css}">{_esc(outcome)}</span></header>{image}'
            f'{f"<dl>{metrics}</dl>" if metrics else ""}{f"<ul class=cues>{cues}</ul>" if cues else ""}</article>')


def _game_table(shots: list[dict], rim: bool) -> str:
    body = ""
    for shot in shots:
        game = shot.get("game") or {}
        metrics = game.get("metrics") or {}
        zone = ZONES.get(metrics.get("shot_zone"), "—") if metrics.get("shot_zone") else "—"
        distance = _with_unit(metrics["shot_distance_ft"], "ft") if metrics.get("shot_distance_ft") is not None else "—"
        body += (f"<tr><td>{_esc(shot['number'])}</td><td>{_num(shot['release_s'])} s</td>"
                 f"<td>{_esc(shot.get('shot_type') or 'shot')}</td><td>{_esc(_outcome(shot, rim)[0])}</td>"
                 f"<td>{_esc(_score_text(shot.get('game')))}</td><td>{_esc(zone)}</td><td>{_esc(distance)}</td></tr>")
    return ("<table><thead><tr><th>#</th><th>Time</th><th>Type</th><th>Result</th><th>Shot-space</th>"
            f"<th>Zone</th><th>Distance</th></tr></thead><tbody>{body}</tbody></table>")


def render_client_report(result: dict, thumbnails: list[str | None], *, analyzed_on: datetime,
                         prepared_for: str = "", video_name: str = "") -> str:
    shots = result.get("shots") or []
    game_mode = result.get("mode") == "one_on_one"
    rim = make_miss_available(result)
    made = sum(1 for s in shots if s.get("outcome") in {"made", "likely made"})
    # A shot with no make/miss call must not read as a miss.
    called = sum(1 for s in shots if s.get("outcome", "unknown") != "unknown")
    title = "Game shot report" if game_mode else "Shooting form review"

    tiles = [("Shots found", str(len(shots))),
             ("Made", f"{made} / {called}" if rim and called else "Not measured")]
    if game_mode:
        mean = (result.get("game_summary") or {}).get("mean_score")
        tiles.append(("Average shot-space", f"{_num(mean)} / 100" if mean is not None else "—"))
        tiles.append(("Court measured in feet", "Yes" if result.get("court_calibration") else "No"))
    else:
        tiles.append(("Shooting hand", str(result.get("handedness") or "right").capitalize()))
        tiles.append(("Clip length", f"{_num((result.get('video') or {}).get('duration_s', 0))} s"))
    summary = "".join(f"<div><small>{_esc(k)}</small><strong>{_esc(v)}</strong></div>" for k, v in tiles)

    sections = []
    if game_mode and shots:
        sections.append(f"<section><h2>Every shot</h2>{_game_table(shots, rim)}</section>")
    consistency = [] if game_mode else form_consistency(shots)
    if consistency:
        rows = "".join(f"<tr><td>{_esc(a)}</td><td>{_esc(b)}</td><td>{_esc(c)}</td><td>{_esc(d)}</td></tr>"
                       for a, b, c, d in consistency)
        sections.append("<section><h2>Consistency across shots</h2><p class=note>A tight range from lowest to "
                        "highest means a repeatable motion. Compare these with the next session filmed from the "
                        "same spot.</p><table><thead><tr><th>Measurement</th><th>Average</th><th>Lowest</th>"
                        f"<th>Highest</th></tr></thead><tbody>{rows}</tbody></table></section>")
    if shots:
        cards = "".join(_shot_card(s, t, game_mode, rim) for s, t in zip(shots, thumbnails))
        sections.append(f"<section><h2>Shot by shot</h2>{cards}</section>")
    else:
        sections.append("<section><h2>Shot by shot</h2><p class=note>No complete shot was found in this clip. "
                        "Keep the ball, shooting arm and basket in frame from the gather through the landing.</p>"
                        "</section>")

    reading = []
    if game_mode:
        reading.append("Shot-space score is 0–100: higher means the shooter had more room. It is a review aid, "
                       "not a make probability or a player rating.")
    if not rim:
        reading.append("Make/miss was not measured because the basket was not visible to the analyzer.")
    reading += result.get("limitations") or []
    limits = "".join(f"<li>{_esc(x)}</li>" for x in reading)
    video_line = (f"<p class=note>The annotated video (<b>{_esc(video_name)}</b>) comes with this report. "
                  "Each snapshot above is the release frame from that video.</p>" if video_name else "")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ballform {_esc(title.lower())}</title>
<style>
:root{{--ink:#171a19;--paper:#fbfaf6;--orange:#fa5a24;--mint:#b9f6d8;--line:#d6d2c8;--muted:#62655f}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 Manrope,system-ui,-apple-system,sans-serif}}
main{{max-width:860px;margin:auto;padding:40px 16px 60px}}
.brand{{font:700 12px ui-monospace,Menlo,monospace;letter-spacing:.12em}}
.brand span{{display:inline-grid;place-items:center;background:var(--orange);color:#fff;width:26px;height:26px;border-radius:50%;margin-right:8px;letter-spacing:0}}
h1{{font-size:clamp(32px,6vw,46px);letter-spacing:-.04em;line-height:1;margin:26px 0 12px}}
h2{{font-size:20px;margin:36px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--ink)}}
h3{{margin:0;font-size:17px}}
.meta{{display:flex;flex-wrap:wrap;gap:6px 28px;color:var(--muted);font-size:14px}}
.meta b{{color:var(--ink)}}
[contenteditable]{{outline:1px dashed var(--line);outline-offset:3px;min-width:8em;display:inline-block}}
.summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--ink);margin:28px 0 0}}
.summary div{{padding:16px;border-right:1px solid var(--ink)}}
.summary div:last-child{{border-right:0}}
.summary small,dt{{display:block;font:11px ui-monospace,Menlo,monospace;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
.summary strong{{display:block;font-size:22px;margin-top:4px}}
.notes{{margin-top:24px;padding:14px 16px;border-left:3px solid var(--orange);background:#fff}}
.notes div[contenteditable]{{display:block;min-height:3em;outline:none}}
.notes div[contenteditable]:empty::before{{content:"Add your notes for the player or coach here before printing.";color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
th{{font:11px ui-monospace,Menlo,monospace;text-transform:uppercase;color:var(--muted)}}
.note{{color:var(--muted);font-size:14px}}
.shot{{border:1px solid var(--ink);margin:0 0 18px;background:#fff;break-inside:avoid}}
.shot header{{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--ink)}}
.time{{color:var(--muted);font-size:13px;margin-right:auto}}
.pill{{font:700 11px ui-monospace,Menlo,monospace;padding:5px 10px;border-radius:20px;background:var(--mint);text-transform:uppercase}}
.pill.missed{{background:#ffd4c5}}.pill.unknown{{background:#e2e0da}}
.shot img{{display:block;width:100%;height:auto;border-bottom:1px solid var(--line)}}
dl{{display:grid;grid-template-columns:repeat(3,1fr);margin:0}}
dl div{{padding:12px 16px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}}
dd{{margin:3px 0 0;font-weight:700;font-size:16px}}
.cues{{margin:0;padding:12px 16px 14px 34px;color:var(--muted);font-size:14px}}
.limits{{color:var(--muted);font-size:13px;padding-left:18px}}
footer{{margin-top:40px;color:var(--muted);font:11px ui-monospace,Menlo,monospace;letter-spacing:.06em}}
.print{{position:fixed;right:16px;bottom:16px;border:1px solid var(--ink);background:var(--orange);color:#fff;padding:12px 18px;font:700 12px ui-monospace,Menlo,monospace;cursor:pointer}}
@media(max-width:640px){{.summary{{grid-template-columns:1fr 1fr}}.summary div:nth-child(2){{border-right:0}}.summary div:nth-child(-n+2){{border-bottom:1px solid var(--ink)}}dl{{grid-template-columns:1fr 1fr}}table{{font-size:12px}}th,td{{padding:6px 4px}}}}
@page{{margin:14mm}}
@media print{{body{{background:#fff;font-size:12px}}main{{padding:0 2px;max-width:none}}.print,.notes.empty{{display:none}}
[contenteditable]{{outline:none}}h2{{break-after:avoid}}.shot img{{max-height:70mm;object-fit:contain;background:#111}}
.summary,.pill{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style>
</head>
<body>
<main>
<div class="brand"><span>BF</span>BALLFORM</div>
<h1>{_esc(title)}</h1>
<div class="meta"><span>Prepared for: <b contenteditable="true" spellcheck="false">{_esc(prepared_for)}</b></span>
<span>Analyzed: <b>{analyzed_on:%B} {analyzed_on.day}, {analyzed_on.year}</b></span></div>
<div class="summary">{summary}</div>
<div class="notes"><small class="brand">NOTES</small><div contenteditable="true"></div></div>
{"".join(sections)}
{video_line}
<section><h2>How to read this</h2><ul class="limits">{limits}</ul></section>
<footer>BALLFORM · VIDEO REVIEW · 2D CAMERA MEASUREMENTS</footer>
</main>
<button class="print" type="button" onclick="window.print()">Save as PDF</button>
<script>
// Hide an empty notes box on paper; the placeholder only helps on screen.
addEventListener("beforeprint",()=>{{const n=document.querySelector(".notes");n.classList.toggle("empty",!n.querySelector("div").textContent.trim());}});
</script>
</body>
</html>
"""


def build_client_report(job_dir: Path, prepared_for: str = "") -> str:
    result_path = job_dir / "result.json"
    result = json.loads(result_path.read_text())
    video = job_dir / (result.get("annotated_video") or "annotated.mp4")
    shots = result.get("shots") or []
    thumbs = (release_thumbnails(video, [s["release_s"] for s in shots])
              if video.exists() else [None] * len(shots))
    return render_client_report(
        result, thumbs,
        analyzed_on=datetime.fromtimestamp(result_path.stat().st_mtime),
        prepared_for=prepared_for,
        video_name=f"ballform-{job_dir.name}.mp4" if video.exists() else "",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the customer report for a finished job.")
    parser.add_argument("job", help="job id under data/jobs, or a job directory")
    parser.add_argument("--for", dest="prepared_for", default="", help="player, parent or team name")
    parser.add_argument("-o", "--output", type=Path, help="default: <job>/client-report.html")
    args = parser.parse_args()
    job_dir = Path(args.job)
    if not job_dir.is_dir():
        job_dir = ROOT / "data" / "jobs" / args.job
    if not (job_dir / "result.json").exists():
        parser.error(f"no result.json in {job_dir}")
    output = args.output or job_dir / "client-report.html"
    output.write_text(build_client_report(job_dir, args.prepared_for))
    print(output)


if __name__ == "__main__":
    main()
