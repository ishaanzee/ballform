import type { JobState } from "./types";

export const API_ORIGIN = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/+$/, "");
export const API_CONFIGURED = Boolean(API_ORIGIN);

export function apiUrl(path: string, token = ""): string {
  if (!API_CONFIGURED) throw new Error("Set NEXT_PUBLIC_API_URL to the analysis service address and redeploy.");
  const url = new URL(`${API_ORIGIN}${path}`);
  if (!["https:", "http:"].includes(url.protocol)) throw new Error("The analysis service must use HTTP or HTTPS.");
  if (typeof window !== "undefined" && window.location.protocol === "https:" && url.protocol !== "https:") {
    throw new Error("The deployed app needs an HTTPS analysis service. HTTP uploads are blocked by browsers.");
  }
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

function errorDetail(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map(item => item?.msg || "Invalid request").join("; ");
  }
  return fallback;
}

export async function getJob(id: string, token = "", signal?: AbortSignal): Promise<JobState> {
  const response = await fetch(apiUrl(`/api/jobs/${encodeURIComponent(id)}`), {
    headers: token ? { "x-ballform-token": token } : {},
    cache: "no-store",
    signal,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(errorDetail(body, `Could not read analysis (${response.status}).`));
  return body as JobState;
}

export function uploadVideo(form: FormData, token: string, onProgress: (percent: number) => void,
                            signal?: AbortSignal): Promise<{ job_id: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const abort = () => { xhr.abort(); reject(new DOMException("Upload cancelled", "AbortError")); };
    if (signal?.aborted) { abort(); return; }
    xhr.open("POST", apiUrl("/api/jobs"));
    if (token) xhr.setRequestHeader("x-ballform-token", token);
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) onProgress(Math.round(event.loaded / event.total * 100));
    };
    xhr.onload = () => {
      let body: { job_id?: string } = {};
      try { body = JSON.parse(xhr.responseText); } catch { /* report HTTP failure below */ }
      if (xhr.status >= 200 && xhr.status < 300 && typeof body.job_id === "string") resolve({ job_id: body.job_id });
      else reject(new Error(errorDetail(body, `Upload failed (${xhr.status}).`)));
    };
    xhr.onerror = () => reject(new Error("Cannot reach the analysis service. Check its address, connection, and allowed frontend origin."));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    xhr.onloadend = () => signal?.removeEventListener("abort", abort);
    signal?.addEventListener("abort", abort, { once: true });
    xhr.send(form);
  });
}
