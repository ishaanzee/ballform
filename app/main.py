from __future__ import annotations

import json
import math
import os
import secrets
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import ROOT, analyze_video
from app.vision import CAMERAS, validate_court

app = FastAPI(title="Ballform", version="0.1.0")
JOBS_DIR = ROOT / "data" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
STATE: dict[str, dict] = {}
LOCK = threading.Lock()
ANALYSIS_LOCK = threading.Lock()
ACCESS_TOKEN = os.environ.get("BALLFORM_ACCESS_TOKEN", "")
MAX_BYTES = 750 * 1024 * 1024
ALLOWED = {".mp4", ".mov", ".m4v", ".avi", ".webm"}


def _update(job_id: str, **values) -> None:
    with LOCK:
        STATE.setdefault(job_id, {}).update(values)


def _run(job_id: str, input_path: Path, rim: tuple[float, float, float, float] | None,
         mode: str = "form", handedness: str = "right", camera: str = "auto", court=None) -> None:
    try:
        _update(job_id, status="waiting", message="Waiting for the local analyzer")
        with ANALYSIS_LOCK:
            _update(job_id, status="running", message="Starting analysis")
            result = analyze_video(input_path, input_path.parent, rim,
                                   lambda p, m: _update(job_id, progress=round(p, 3), message=m),
                                   mode=mode, handedness=handedness, camera=camera, court=court)
        _update(job_id, status="complete", progress=1.0, message="Complete", result=result)
    except Exception as exc:
        _update(job_id, status="failed", message=str(exc), error=type(exc).__name__)


@app.middleware("http")
async def require_lan_token(request: Request, call_next):
    """Protect job data when the server is deliberately exposed to the LAN."""
    if ACCESS_TOKEN and request.url.path.startswith("/api/"):
        supplied = request.query_params.get("token") or request.headers.get("x-ballform-token")
        if not supplied or not secrets.compare_digest(supplied, ACCESS_TOKEN):
            return JSONResponse({"detail": "Invalid or missing Ballform pairing token."}, status_code=401)
    return await call_next(request)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


@app.post("/api/jobs", status_code=202)
async def create_job(background: BackgroundTasks, video: UploadFile = File(...), rim: str | None = Form(None),
                     mode: str = Form("form"), handedness: str = Form("right"),
                     camera: str = Form("auto"), court: str | None = Form(None)) -> dict:
    if mode not in {"form", "one_on_one"}:
        raise HTTPException(422, "Mode must be form or one_on_one.")
    if handedness not in {"right", "left"}:
        raise HTTPException(422, "Handedness must be right or left.")
    if camera not in CAMERAS:
        raise HTTPException(422, "Camera must be auto, broadcast, elevated, moving or courtside.")
    try:
        court_polygon = validate_court(json.loads(court)) if court else None
    except (ValueError, TypeError):
        raise HTTPException(422, "Court must be 3–8 normalized [x,y] points around a convex playing area.") from None
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"Use one of: {', '.join(sorted(ALLOWED))}")
    rim_box = None
    if rim:
        try:
            values = tuple(float(v) for v in json.loads(rim))
            if (len(values) != 4 or any(not math.isfinite(v) or v < 0 or v > 1 for v in values)
                    or values[2] <= .005 or values[3] <= .005
                    or values[0] + values[2] > 1 or values[1] + values[3] > 1):
                raise ValueError
            rim_box = values
        except (ValueError, TypeError, json.JSONDecodeError):
            raise HTTPException(422, "Rim must be a normalized [x, y, width, height] box.") from None
    if camera == "moving" and rim_box is None:
        raise HTTPException(422, "Moving camera analysis requires a rim box marked on the first frame.")
    job_id = uuid.uuid4().hex[:12]
    directory = JOBS_DIR / job_id
    directory.mkdir()
    input_path = directory / f"input{suffix}"
    size = 0
    with input_path.open("wb") as target:
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                target.close()
                input_path.unlink(missing_ok=True)
                directory.rmdir()
                raise HTTPException(413, "Video exceeds the 750 MB local upload limit.")
            target.write(chunk)
    _update(job_id, status="queued", progress=0.0, message="Queued")
    background.add_task(_run, job_id, input_path, rim_box, mode, handedness, camera, court_polygon)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if not job_id.isalnum():
        raise HTTPException(404, "Job not found")
    if job_id not in STATE:
        result_path = JOBS_DIR / job_id / "result.json"
        if result_path.exists():
            return {"status": "complete", "progress": 1, "result": json.loads(result_path.read_text())}
        raise HTTPException(404, "Job not found")
    return STATE[job_id]


@app.get("/api/jobs/{job_id}/video")
def get_video(job_id: str) -> FileResponse:
    if not job_id.isalnum():
        raise HTTPException(404)
    path = JOBS_DIR / job_id / "annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Annotated video is not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"ballform-{job_id}.mp4")


app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")
