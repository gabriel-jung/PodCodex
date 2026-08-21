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
import { ConnectionError } from "./connection";

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

export {
  BOOT_PATIENT_RETRY,
  CONNECT_RETRY,
  ConnectionError,
  isConnectionError,
} from "./connection";
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
