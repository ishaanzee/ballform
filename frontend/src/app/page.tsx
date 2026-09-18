"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent,
} from "react";
import {
  API_CONFIGURED,
  API_ORIGIN,
  apiUrl,
  getJob,
  uploadVideo,
} from "@/lib/api";
import type {
  AnalysisMode,
  AnalysisResult,
  Handedness,
  RimBox,
  ShotResult,
} from "@/lib/types";

const labels: Record<string, string> = {
  elbow_angle_at_release_deg: "Elbow at release",
  set_point_elbow_angle_deg: "Set-point elbow",
  upper_arm_elevation_deg: "Upper-arm elevation",
  wrist_over_elbow_pct_shoulder_width: "Wrist / elbow offset",
  release_height_body_ratio: "Release height ratio",
  follow_through_extension_deg: "Follow-through",
  release_angle_2d_deg: "2D launch angle",
  separation_torso: "Projected separation at release",
  contest_clearance_torso: "Defender hand / release clearance",
  separation_change_torso: "Separation change before release",
  separation: "Release separation",
  contest_clearance: "Contest clearance",
};
function label(key: string) {
  return labels[key] || key.replaceAll("_", " ");
}
function metric(key: string, value: unknown) {
  if (value == null) return "Unavailable";
  const display =
    typeof value === "number" ? Number(value.toFixed(2)) : String(value);
  return `${display}${key.includes("angle") || key.includes("_deg") ? "°" : key.includes("pct") ? "%" : key.includes("torso") ? " torso lengths" : key.endsWith("_s") ? " s" : ""}`;
}
function Metrics({ values }: { values?: Record<string, unknown> }) {
  return (
    <div className="metrics">
      {Object.entries(values || {}).map(([key, value]) => (
        <div className="metric" key={key}>
          <small>{label(key)}</small>
          <strong>{metric(key, value)}</strong>
        </div>
      ))}
    </div>
  );
}
function GameReview({ game }: { game: ShotResult["game"] }) {
  if (!game)
    return (
      <p className="cue">Game measurements unavailable for this release.</p>
    );
  return (
    <section className="game-review" aria-label="Shot-space analysis">
      <div className="score-row">
        <div>
          <small>Shot-space score</small>
          <strong>
            {game.score == null ? (
              "Not scored"
            ) : (
              <>
                {game.score}
                <span> / 100</span>
              </>
            )}
          </strong>
        </div>
        <p>
          {game.confidence == null
            ? "Evidence quality unavailable"
            : `${Math.round(game.confidence * 100)} / 100 evidence quality`}
          <br />
          <small>Heuristic · higher means more measured space</small>
        </p>
      </div>
      <Metrics values={game.metrics} />
      {!!Object.keys(game.components || {}).length && (
        <details>
          <summary>Score components</summary>
          <div className="components">
            {Object.entries(game.components || {}).map(([key, value]) => {
              const component =
                typeof value === "object" && value !== null
                  ? (value as { score?: number | null; weight?: number })
                  : { score: value };
              return (
                <div className="component" key={key}>
                  <span>{label(key)}</span>
                  <strong>
                    {component.score == null
                      ? "Unavailable"
                      : `${component.score} / 100`}
                  </strong>
                  {component.weight != null && (
                    <small>Weight {Math.round(component.weight * 100)}%</small>
                  )}
                </div>
              );
            })}
          </div>
        </details>
      )}
      <p className="evidence">{game.evidence?.join(" · ")}</p>
      {!!game.limitations?.length && (
        <ul className="game-limitations">
          {game.limitations.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default function Home() {
  const [mode, setMode] = useState<AnalysisMode>("form");
  const [handedness, setHandedness] = useState<Handedness>("right");
  const [token, setToken] = useState("");
  const [ready, setReady] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [aspect, setAspect] = useState(16 / 9);
  const [scrub, setScrub] = useState(25);
  const [rim, setRim] = useState<RimBox | null>(null);
  const [over, setOver] = useState(false);
  const [phase, setPhase] = useState<
    "choose" | "mark" | "progress" | "results"
  >("choose");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [report, setReport] = useState<AnalysisResult | null>(null);
  const preview = useRef<HTMLVideoElement>(null);
  const annotated = useRef<HTMLVideoElement>(null);
  const dragStart = useRef<{ x: number; y: number } | null>(null);
  const controller = useRef<AbortController | null>(null);
  const uploading = useRef(false);
  const storageKey = `ballform-job:${API_ORIGIN}`;
  const tokenKey = `ballform-token:${API_ORIGIN}`;

  useEffect(() => {
    const url = new URL(window.location.href);
    let savedToken = url.searchParams.get("token") || "";
    try {
      savedToken ||= sessionStorage.getItem(tokenKey) || "";
      sessionStorage.setItem(tokenKey, savedToken);
    } catch {
      /* Storage is optional. */
    }
    if (url.searchParams.has("token")) {
      url.searchParams.delete("token");
      window.history.replaceState(null, "", url);
    }
    setToken(savedToken);
    setReady(true);
    if (API_CONFIGURED) {
      try {
        // Both the job and its credential stay in this tab's session storage.
        const saved = JSON.parse(
          sessionStorage.getItem(storageKey) || "null",
        ) as { id: string; token: string } | null;
        if (
          saved &&
          saved.token === savedToken &&
          /^[a-zA-Z0-9_-]+$/.test(saved.id)
        ) {
          setJobId(saved.id);
          setPhase("progress");
          setMessage("Reconnecting to analysis…");
        }
      } catch {
        /* A stale browser record does not block a new upload. */
      }
    }
    return () => controller.current?.abort();
  }, [storageKey, tokenKey]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const forgetJob = useCallback(() => {
    try {
      sessionStorage.removeItem(storageKey);
    } catch {
      /* optional */
    }
  }, [storageKey]);
  useEffect(() => {
    if (!jobId || phase !== "progress" || !ready) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const state = await getJob(jobId!, token, abort.signal);
        if (abort.signal.aborted) return;
        setProgress(35 + Math.max(0, Math.min(1, state.progress || 0)) * 65);
        setMessage(state.message || "Analyzing video…");
        if (state.status === "complete" && state.result) {
          setError("");
          setReport(state.result);
          setPhase("results");
          forgetJob();
          return;
        }
        if (state.status === "failed") {
          forgetJob();
          setJobId(null);
          setError(
            state.message || "Analysis failed. Please try another clip.",
          );
          return;
        }
        timer = setTimeout(poll, 1000);
      } catch (cause) {
        if (!abort.signal.aborted)
          setError(
            cause instanceof Error
              ? cause.message
              : "Unable to reconnect to the analysis service.",
          );
      }
    }
    void poll();
    return () => {
      abort.abort();
      clearTimeout(timer);
    };
  }, [jobId, phase, ready, token, forgetJob, retryCount]);

  function loadFile(chosen?: File) {
    if (!chosen) return;
    if (chosen.size > 750 * 1024 * 1024) {
      setError("Choose a video smaller than 750 MB.");
      return;
    }
    if (!/\.(mp4|mov|m4v|webm)$/i.test(chosen.name)) {
      setError("Choose an MP4, MOV, M4V, or WEBM video.");
      return;
    }
    setError("");
    setFile(chosen);
    setRim(null);
    setScrub(25);
    setPhase("mark");
  }
  function point(event: PointerEvent<HTMLDivElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      y: Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
    };
  }
  async function analyze() {
    if (!file || !API_CONFIGURED || uploading.current) return;
    uploading.current = true;
    const abort = new AbortController();
    controller.current?.abort();
    controller.current = abort;
    setPhase("progress");
    setError("");
    setProgress(0);
    setMessage("Preparing upload…");
    setJobId(null);
    const form = new FormData();
    form.append("video", file);
    form.append("mode", mode);
    form.append("handedness", handedness);
    if (rim && rim[2] >= 0.01 && rim[3] >= 0.01)
      form.append("rim", JSON.stringify(rim));
    try {
      const result = await uploadVideo(
        form,
        token,
        (percent: number) => {
          setProgress(percent * 0.35);
          setMessage(`Uploading video… ${Math.round(percent)}%`);
        },
        abort.signal,
      );
      if (abort.signal.aborted) return;
      try {
        sessionStorage.setItem(
          storageKey,
          JSON.stringify({ id: result.job_id, token }),
        );
      } catch {
        /* Analysis works without storage. */
      }
      setMessage("Upload complete. Waiting for analysis…");
      setJobId(result.job_id);
    } catch (cause) {
      if (!abort.signal.aborted)
        setError(cause instanceof Error ? cause.message : "Upload failed.");
    } finally {
      uploading.current = false;
    }
  }
  function reset() {
    controller.current?.abort();
    forgetJob();
    setJobId(null);
    setReport(null);
    setFile(null);
    setRim(null);
    setError("");
    setPhase("choose");
  }
  function download() {
    if (!report) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `ballform-${report.mode || "form"}-report.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const game = report?.mode === "one_on_one";
  return (
    <>
      <header>
        <a className="brand" href="/">
          <span>BF</span> BALLFORM
        </a>
        <div className="local">● VIDEO ANALYSIS LAB</div>
      </header>
      <main>
        <section className="hero">
          <p className="eyebrow">COMPUTER VISION SHOOTING REVIEW</p>
          <h1>
            See the shot
            <br />
            <em>behind the shot.</em>
          </h1>
          <p className="lede">
            Review your mechanics or the space you create in 1-on-1. Follow each
            release, inspect the evidence, and take specific questions back to
            the court.
          </p>
        </section>
        {!API_CONFIGURED && (
          <aside className="setup-note">
            <h2>Connect your analysis service</h2>
            <p>
              Set <code>NEXT_PUBLIC_API_URL</code> to your public HTTPS Python
              backend URL in Vercel, then redeploy. The backend runs the
              computer vision models; videos upload directly to it.
            </p>
          </aside>
        )}
        {phase !== "results" && (
          <section className="workspace">
            {phase !== "progress" && (
              <>
                <div className="step">
                  <b>01</b>
                  <span>CHOOSE CLIP</span>
                </div>
                <div className="settings">
                  <label>
                    Analysis mode
                    <select
                      value={mode}
                      onChange={(e) => setMode(e.target.value as AnalysisMode)}
                      aria-describedby="modeHelp"
                    >
                      <option value="form">Shooting form</option>
                      <option value="one_on_one">1-on-1 game</option>
                    </select>
                  </label>
                  <label>
                    Shooting hand
                    <select
                      value={handedness}
                      onChange={(e) =>
                        setHandedness(e.target.value as Handedness)
                      }
                    >
                      <option value="right">Right hand</option>
                      <option value="left">Left hand</option>
                    </select>
                  </label>
                  <p id="modeHelp">
                    {mode === "one_on_one"
                      ? "Use a steady side or slightly angled view. Keep both players, their hands and feet, the ball, and the basket visible. Overlap and camera movement reduce measurement confidence."
                      : "Keep the shooting arm, ball, feet, and basket visible. Use a steady camera for a repeatable mechanics review."}
                  </p>
                  <label className="token-field">
                    Backend access token (if required)
                    <input
                      type="password"
                      autoComplete="off"
                      value={token}
                      onChange={(e) => {
                        setToken(e.target.value);
                        try {
                          sessionStorage.setItem(tokenKey, e.target.value);
                        } catch {
                          /* optional */
                        }
                      }}
                    />
                    <small>
                      Stored only for this browser tab&apos;s session.
                    </small>
                  </label>
                </div>
              </>
            )}
            {error && (
              <p className="error inline-error" role="alert">
                {error}
              </p>
            )}
            {phase === "choose" && (
              <>
                <label
                  className={`drop${over ? " over" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setOver(true);
                  }}
                  onDragLeave={() => setOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setOver(false);
                    loadFile(e.dataTransfer.files[0]);
                  }}
                >
                  <input
                    className="accessible-file"
                    type="file"
                    accept="video/mp4,video/quicktime,video/x-m4v,video/webm"
                    onChange={(e) => loadFile(e.target.files?.[0])}
                  />
                  <span className="ball">●</span>
                  <strong>Drop a basketball video here</strong>
                  <small>
                    or tap to choose from Photos · MP4, MOV, M4V, WEBM · up to
                    750 MB
                  </small>
                </label>
                <label className="record">
                  <input
                    className="accessible-file"
                    type="file"
                    accept="video/*"
                    capture="environment"
                    onChange={(e) => loadFile(e.target.files?.[0])}
                  />
                  <span>PHONE</span>
                  <strong>Record a new clip</strong>
                  <small>Use your rear camera</small>
                </label>
              </>
            )}
            {phase === "mark" && (
              <>
                <div className="step rim-step">
                  <b>02</b>
                  <span>MARK THE RIM</span>
                  <small>Drag a snug box around the orange rim.</small>
                </div>
                <div className="preview-wrap">
                  <div className="video-frame" style={{ aspectRatio: aspect }}>
                    <video
                      ref={preview}
                      src={previewUrl}
                      muted
                      playsInline
                      onError={() =>
                        setError(
                          "This browser could not preview the clip. Try an MP4 encoded with H.264.",
                        )
                      }
                      onLoadedMetadata={(e) => {
                        const video = e.currentTarget;
                        setAspect(
                          video.videoWidth / video.videoHeight || 16 / 9,
                        );
                        if (Number.isFinite(video.duration))
                          video.currentTime = Math.max(
                            0,
                            Math.min(
                              video.duration * 0.25,
                              video.duration - 0.05,
                            ),
                          );
                      }}
                    />
                    <div
                      className="rim-overlay"
                      aria-label="Drag to mark the rim in the video"
                      onPointerDown={(e) => {
                        dragStart.current = point(e);
                        setRim(null);
                        e.currentTarget.setPointerCapture(e.pointerId);
                      }}
                      onPointerMove={(e) => {
                        if (!dragStart.current) return;
                        const p = point(e),
                          start = dragStart.current;
                        setRim([
                          Math.min(p.x, start.x),
                          Math.min(p.y, start.y),
                          Math.abs(p.x - start.x),
                          Math.abs(p.y - start.y),
                        ]);
                      }}
                      onPointerUp={() => {
                        dragStart.current = null;
                      }}
                      onPointerCancel={() => {
                        dragStart.current = null;
                      }}
                    >
                      {rim && (
                        <div
                          className="rim-box"
                          style={{
                            left: `${rim[0] * 100}%`,
                            top: `${rim[1] * 100}%`,
                            width: `${rim[2] * 100}%`,
                            height: `${rim[3] * 100}%`,
                          }}
                        >
                          <span>RIM</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="scrub">
                    <input
                      aria-label="Preview frame"
                      type="range"
                      min="0"
                      max="100"
                      value={scrub}
                      onChange={(e) => {
                        setScrub(Number(e.target.value));
                        if (
                          preview.current &&
                          Number.isFinite(preview.current.duration)
                        )
                          preview.current.currentTime = Math.min(
                            preview.current.duration - 0.01,
                            (preview.current.duration *
                              Number(e.target.value)) /
                              100,
                          );
                      }}
                    />
                    <span>Move to a clear rim frame</span>
                  </div>
                </div>
                <p className="rim-help">
                  {file?.name} · Rim marking is optional. Use a steady clip: the
                  box stays fixed throughout the video.
                </p>
                <div className="actions">
                  <button className="ghost" onClick={reset}>
                    Change clip
                  </button>
                  <button className="ghost" onClick={() => setRim(null)}>
                    Clear box
                  </button>
                  <button
                    disabled={!API_CONFIGURED || !ready}
                    onClick={() => void analyze()}
                  >
                    Analyze {mode === "one_on_one" ? "1-on-1" : "form"}{" "}
                    <span>→</span>
                  </button>
                </div>
              </>
            )}
            {phase === "progress" && (
              <div className="progress-panel">
                <div className="step">
                  <b>03</b>
                  <span>ANALYZING</span>
                </div>
                <div
                  className="meter"
                  role="progressbar"
                  aria-label="Analysis progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(progress)}
                >
                  <i style={{ width: `${Math.max(2, progress)}%` }} />
                </div>
                <p role="status" aria-live="polite">
                  {message}
                </p>
                {error && (
                  <div className="progress-actions">
                    {jobId && (
                      <button
                        className="ghost"
                        onClick={() => {
                          setError("");
                          setMessage("Reconnecting to analysis…");
                          setRetryCount((value) => value + 1);
                        }}
                      >
                        Reconnect
                      </button>
                    )}
                    <button
                      className="ghost"
                      onClick={() => {
                        forgetJob();
                        setJobId(null);
                        setError("");
                        setPhase(file ? "mark" : "choose");
                      }}
                    >
                      Return to clip
                    </button>
                  </div>
                )}
                <small>
                  Video processing runs on your connected analysis service.
                  Large clips can take several minutes.
                </small>
              </div>
            )}
          </section>
        )}
        {phase === "results" && report && (
          <section className="results">
            <div className="result-head">
              <div>
                <p className="eyebrow">
                  {game ? "1-ON-1 GAME" : "SHOOTING FORM"} ·{" "}
                  {(report.handedness || handedness).toUpperCase()} HAND
                </p>
                <h2>Your shot report</h2>
              </div>
              <div className="report-actions">
                <button className="ghost" onClick={download}>
                  Download report
                </button>
                <button className="ghost" onClick={reset}>
                  Analyze another
                </button>
              </div>
            </div>
            <div className="summary">
              <div>
                <small>Releases found</small>
                <strong>{report.shots.length}</strong>
              </div>
              <div>
                <small>Made / likely made</small>
                <strong>
                  {
                    report.shots.filter(
                      (s) =>
                        s.outcome === "made" || s.outcome === "likely made",
                    ).length
                  }
                </strong>
              </div>
              <div>
                <small>Camera view</small>
                <strong>{report.camera_view || "Unknown"}</strong>
              </div>
              <div>
                <small>{game ? "Scored releases" : "Pose coverage"}</small>
                <strong>
                  {game
                    ? `${report.shots.filter((s) => s.game?.score != null).length} / ${report.shots.length}`
                    : `${report.diagnostics?.pose_frames ?? 0} frames`}
                </strong>
              </div>
            </div>
            {game && (
              <p className="report-note">
                Shot-space score is a transparent 0–100 heuristic, not make
                probability or a validated player grade. Distances are projected
                in the image and normalized to the shooter&apos;s torso length;
                they are not feet or meters. Compare clips only with similar
                camera angles.
                {report.game_summary?.method?.formula
                  ? ` Score: ${report.game_summary.method.formula}.`
                  : ""}
              </p>
            )}
            <video
              id="annotated"
              ref={annotated}
              src={apiUrl(`/api/jobs/${jobId}/video`, token)}
              controls
              playsInline
              onError={() =>
                setError(
                  "The annotated video could not load. Check your backend connection and access token.",
                )
              }
            />
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
            {report.shots.map((shot) => (
              <article className="shot" key={shot.number}>
                <div className="shot-top">
                  <h3>Shot {shot.number}</h3>
                  <button
                    className="ghost seek"
                    aria-label={`Review shot ${shot.number} release`}
                    onClick={() => {
                      const video = annotated.current;
                      if (video) {
                        video.currentTime = Math.max(0, shot.release_s - 0.5);
                        video.pause();
                        video.scrollIntoView({
                          behavior: "smooth",
                          block: "center",
                        });
                      }
                    }}
                  >
                    Review {shot.release_s}s
                  </button>
                  <span
                    className={`pill ${shot.outcome.includes("miss") ? "missed" : shot.outcome === "unknown" ? "unknown" : ""}`}
                  >
                    {shot.outcome} · {Math.round(shot.outcome_confidence * 100)}
                    %
                  </span>
                </div>
                {game && <GameReview game={shot.game} />}
                {!!Object.keys(shot.metrics || {}).length && (
                  <>
                    <h4 className="section-label">Shooting mechanics</h4>
                    <Metrics values={shot.metrics} />
                  </>
                )}
                {!!shot.cues?.length && (
                  <p className="cue">{shot.cues.join(" ")}</p>
                )}
                <div className="evidence">
                  Evidence: {shot.evidence?.join(" · ")}
                </div>
              </article>
            ))}
            {!report.shots.length && (
              <article className="shot">
                <p className="cue">
                  No complete shot arc was detected. Try a clip where the ball,
                  shooting wrist, and basket stay visible from gather through
                  landing.
                </p>
              </article>
            )}
            <div className="limitations">
              <h3>Read this correctly</h3>
              <ul>
                {report.limitations.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </div>
          </section>
        )}
      </main>
      <footer>
        <span>BALLFORM / VISION LAB</span>
        <span>VIDEO EVIDENCE · 2D CAMERA PLANE</span>
      </footer>
    </>
  );
}
