/**
 * PodCodex API client.
 *
 * Exposes the shared `json()` fetch wrapper plus a barrel re-export of every
 * feature module (health, shows, transcribe, ...). Feature modules import
 * `json` from this file, so all API traffic flows through one place.
 */

// In Tauri production builds the frontend is served from tauri:// and needs
// an absolute URL to reach FastAPI. In dev (Vite) the proxy handles it.
import { isTauri } from "@/platform";

export const BASE =
  isTauri() && import.meta.env.PROD ? "http://127.0.0.1:18811" : "";

// Custom CSRF header forces a CORS preflight that rejects cross-origin pages.
// Mirrors `CSRF_HEADER`/`CSRF_VALUE` in src/podcodex/api/app.py.
export const CSRF_HEADER = "X-PodCodex";
export const CSRF_VALUE = "1";

// Loopback auth token. Mirrors `TOKEN_HEADER`/`TOKEN_QUERY_PARAM` in
// src/podcodex/core/api_token.py (sync-checked by
// tests/test_frontend_constants_sync.py). In Tauri the Rust shell reads the
// token file and hands it over via invoke; in dev the Vite proxy injects the
// header server-side (see vite.config.ts), so the browser holds no token.
const TOKEN_HEADER = "X-PodCodex-Token";
const TOKEN_QUERY_PARAM = "token";

let API_TOKEN = "";

const tokenListeners = new Set<() => void>();

/** Subscribe to the token arriving *after* startup (see `initApiToken`).
 *  URLs built by `withToken` embed the token at render time, so anything
 *  rendered during the empty window has to be rebuilt once it lands. */
export function onApiTokenAcquired(cb: () => void): () => void {
  tokenListeners.add(cb);
  return () => tokenListeners.delete(cb);
}

function setApiToken(next: string): void {
  const acquired = !API_TOKEN && !!next;
  API_TOKEN = next;
  if (acquired) for (const cb of tokenListeners) cb();
}

async function readApiToken(): Promise<string> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    return await invoke<string>("get_api_token");
  } catch {
    return "";
  }
}

// First-launch race: the PyInstaller sidecar takes 10-30s to extract and
// only then writes the token file, so the first read can come back empty.
// Give up eventually — if the sidecar never starts, the app is dead anyway.
const TOKEN_RETRY_LIMIT = 60;
let tokenRetryTimer: ReturnType<typeof setTimeout> | null = null;
// Module-level, not a parameter: `rawFetch` re-enters `initApiToken` on every
// 401, which would otherwise restart the backoff at attempt 0 forever and
// make the cap decorative.
let tokenRetryAttempt = 0;

function scheduleTokenRetry(): void {
  if (tokenRetryTimer || tokenRetryAttempt >= TOKEN_RETRY_LIMIT) return;
  const delay = Math.min(500 * 2 ** tokenRetryAttempt, 3000);
  tokenRetryAttempt += 1;
  tokenRetryTimer = setTimeout(() => {
    tokenRetryTimer = null;
    void readApiToken().then((token) => {
      setApiToken(token);
      if (token) tokenRetryAttempt = 0;
      else scheduleTokenRetry();
    });
  }, delay);
}

/** Resolve the loopback auth token before the app renders (see main.tsx).
 *  One immediate attempt keeps the common path instant; an empty result
 *  means the sidecar hasn't written the file yet, so keep polling in the
 *  background rather than running the whole session unauthenticated. */
export async function initApiToken(): Promise<void> {
  if (!isTauri()) return;
  setApiToken(await readApiToken());
  if (!API_TOKEN) scheduleTokenRetry();
}

/** Append the auth token to a URL destined for src/href/WebSocket use.
 *  No-op in dev, where the Vite proxy injects the header instead. */
export const withToken = (url: string): string =>
  API_TOKEN
    ? `${url}${url.includes("?") ? "&" : "?"}${TOKEN_QUERY_PARAM}=${encodeURIComponent(API_TOKEN)}`
    : url;

/** Thrown for non-2xx responses. Carries the status and parsed JSON body
 *  (raw text when not JSON) so callers can read structured details (e.g.
 *  FastAPI's `detail`) without re-parsing the message string. */
export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, body: unknown, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/** Thrown when the request never reached the server: connection refused,
 *  socket reset, DNS. `fetch` rejects with a bare `TypeError` for these
 *  ("Failed to fetch" in Chromium, "Load failed" in WebKit) and gives no
 *  machine-readable code, so it is tagged at the one place that knows the
 *  rejection came from `fetch` rather than from our own code. */
export class ConnectionError extends Error {
  constructor(cause: unknown) {
    super(cause instanceof Error ? cause.message : String(cause));
    this.name = "ConnectionError";
    this.cause = cause;
  }
}

/** True when the backend has not answered at all — as opposed to answering
 *  with a status we did not like.
 *
 *  This is what lets a query tell "the backend has not finished booting"
 *  apart from "the backend said no": the first deserves a patient retry,
 *  the second an error state right away. Matched on our own tag rather than
 *  on `TypeError`, because a genuine bug inside a queryFn (reading a
 *  property of undefined) is also a TypeError, and would otherwise be
 *  retried for three minutes instead of surfacing. */
export function isConnectionError(error: unknown): boolean {
  return error instanceof ConnectionError;
}

/**
 * How patient a query is while the backend is still coming up.
 *
 * One policy, applied by the QueryClient defaults in `main.tsx`, so every
 * query agrees. It lives here rather than beside a particular query because
 * it is about the connection, not about any endpoint — and because callers
 * spreading their own copy is what let `/api/health` retry a genuine 500
 * sixty times.
 *
 * The delay ramp matters more than the count: a cold first launch takes
 * seconds, so backing off to a few seconds and staying there beats
 * exponential growth that overshoots.
 */
export const CONNECT_RETRY = {
  limit: 4,
  delay: (attempt: number) => Math.min(500 + attempt * 500, 3000),
} as const;

/**
 * The exception to `CONNECT_RETRY`: queries that must survive a slow first
 * launch rather than give up and be refetched later.
 *
 * There are exactly two. `/api/health` drives the boot banner and the
 * "Backend not reachable" screen, and its first success is what refetches
 * everything else. The pipeline-defaults hydration is the other: giving up
 * there leaves the store on built-ins, and the user's next settings edit
 * writes those over their real values (see `stores/pipelineConfigStore`).
 *
 * Connection failures only — a status the server actually returned is a
 * real failure and must surface, not hide behind three minutes of retries.
 */
export const BOOT_PATIENT_RETRY = {
  retry: (failureCount: number, error: unknown) =>
    isConnectionError(error) && failureCount < 60,
  retryDelay: (attempt: number) => CONNECT_RETRY.delay(attempt),
} as const;

/** `fetch`, with a transport failure re-thrown as a `ConnectionError`.
 *  Everything the server answered — any status — comes back normally. */
async function fetchOrThrow(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (error) {
    throw new ConnectionError(error);
  }
}

/** Fetch with the CSRF header already set. Use for non-JSON responses or
 *  FormData uploads where `json()` doesn't fit. Throws ApiError on `!res.ok`. */
export async function rawFetch(url: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  headers.set(CSRF_HEADER, CSRF_VALUE);
  headers.set(TOKEN_HEADER, API_TOKEN);
  let res = await fetchOrThrow(`${BASE}${url}`, { ...init, headers });
  if (res.status === 401) {
    // First-boot race in Tauri: the token file may not have existed when
    // the app initialized. Retry only when re-reading actually yields a
    // different token, so mutations are never blindly re-sent on a 401.
    const previous = API_TOKEN;
    await initApiToken();
    if (API_TOKEN && API_TOKEN !== previous) {
      headers.set(TOKEN_HEADER, API_TOKEN);
      res = await fetchOrThrow(`${BASE}${url}`, { ...init, headers });
    }
  }
  if (!res.ok) {
    const text = await res.text();
    let body: unknown = text;
    try {
      body = JSON.parse(text);
    } catch { /* keep raw text */ }
    // Prefer FastAPI's `detail` string so surfaced errors read as plain
    // sentences, not status codes and JSON.
    const detail = (body as { detail?: unknown } | null)?.detail;
    const message = typeof detail === "string" ? detail : `${res.status}: ${text}`;
    throw new ApiError(res.status, body, message);
  }
  return res;
}

export async function json<T>(url: string, init?: RequestInit): Promise<T> {
  return (await rawFetch(url, init)).json();
}

export * from "./health";
export * from "./shows";
export * from "./transcribe";
export * from "./correct";
export * from "./translate";
export * from "./synthesize";
export * from "./search";
export * from "./episodes";
export * from "./filesystem";
export * from "./integrations";
export * from "./mcpPrompts";
export * from "./botAccess";
export * from "./config";
export * from "./bundle";
export * from "./gpu";
export * from "./keys";
export * from "./providerProfiles";
