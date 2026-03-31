/** Shared fetch wrapper for the PodCodex API. */

// In Tauri production builds, the frontend is served from tauri:// protocol
// so we need an absolute URL to reach the FastAPI backend.
// In dev (Vite), the proxy handles it.
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
