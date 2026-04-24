/**
 * PodCodex API client.
 *
 * Exposes the shared `json()` fetch wrapper plus a barrel re-export of every
 * feature module (health, shows, transcribe, ...). Feature modules import
 * `json` from this file, so all API traffic flows through one place.
 */

// In Tauri production builds the frontend is served from tauri:// and needs
// an absolute URL to reach FastAPI. In dev (Vite) the proxy handles it.
export const BASE =
  (window as any).__TAURI__ && import.meta.env.PROD
    ? "http://127.0.0.1:18811"
    : "";

export async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
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
