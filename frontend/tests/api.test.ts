import assert from "node:assert/strict";
import { test } from "node:test";

process.env.NEXT_PUBLIC_API_URL = "https://worker.example/";
const { apiUrl, getJob, uploadVideo } = await import("../src/lib/api");

test("video URLs use the worker and encode the private token", () => {
  const url = new URL(apiUrl("/api/jobs/abc/video", "a+b&c"));
  assert.equal(url.origin, "https://worker.example");
  assert.equal(url.pathname, "/api/jobs/abc/video");
  assert.equal(url.searchParams.get("token"), "a+b&c");
});

test("polling sends authorization in a header, disables caching and supports cancellation", async () => {
  const originalFetch = globalThis.fetch;
  const controller = new AbortController();
  globalThis.fetch = async (input, init) => {
    assert.equal(String(input), "https://worker.example/api/jobs/abc");
    assert.equal((init?.headers as Record<string, string>)["x-ballform-token"], "secret");
    assert.equal(init?.cache, "no-store");
    assert.equal(init?.signal, controller.signal);
    return Response.json({ status: "running", progress: .5 });
  };
  try {
    assert.equal((await getJob("abc", "secret", controller.signal)).progress, .5);
  } finally { globalThis.fetch = originalFetch; }
});

test("authentication errors remain actionable", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ detail: "Invalid or missing Ballform access token." }, { status: 401 });
  try {
    await assert.rejects(getJob("abc", "bad"), /Invalid or missing Ballform access token/);
  } finally { globalThis.fetch = originalFetch; }
});

test("uploads go directly to the worker with real transfer progress", async () => {
  const originalXHR = globalThis.XMLHttpRequest;
  let sent: FormData | undefined;
  class MockXHR {
    status = 202;
    responseText = '{"job_id":"job123"}';
    upload: { onprogress?: (event: {lengthComputable: boolean; loaded: number; total: number}) => void } = {};
    onload?: () => void;
    onloadend?: () => void;
    open(method: string, url: string) {
      assert.equal(method, "POST");
      assert.equal(url, "https://worker.example/api/jobs");
    }
    setRequestHeader(key: string, value: string) {
      assert.equal(key, "x-ballform-token");
      assert.equal(value, "secret");
    }
    send(form: FormData) {
      sent = form;
      this.upload.onprogress?.({lengthComputable: true, loaded: 25, total: 100});
      this.onload?.();
      this.onloadend?.();
    }
  }
  globalThis.XMLHttpRequest = MockXHR as unknown as typeof XMLHttpRequest;
  try {
    const form = new FormData();
    form.append("mode", "one_on_one");
    const progress: number[] = [];
    const result = await uploadVideo(form, "secret", value => progress.push(value));
    assert.equal(result.job_id, "job123");
    assert.equal(sent, form);
    assert.deepEqual(progress, [25]);
  } finally { globalThis.XMLHttpRequest = originalXHR; }
});
