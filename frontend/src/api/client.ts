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

/** Fetch with the CSRF header already set. Use for non-JSON responses or
 *  FormData uploads where `json()` doesn't fit. Throws on `!res.ok`. */
export async function rawFetch(url: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  headers.set(CSRF_HEADER, CSRF_VALUE);
  const res = await fetch(`${BASE}${url}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
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
