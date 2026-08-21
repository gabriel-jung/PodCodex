/** Connection failure tagging and the retry policies built on it.
 *
 * A leaf module on purpose. `client.ts` is a barrel (`export * from
 * "./health"` and friends), and in ESM a barrel's dependencies evaluate
 * before its own body, so anything the barrel re-exports that reads a
 * `client.ts` value *at module scope* gets a temporal-dead-zone error.
 * `health.ts` does exactly that with `BOOT_PATIENT_RETRY`. Rollup reorders
 * and hides it in the production build; Vite's per-module ESM in dev does
 * not, so it showed up only under `make dev-no-tauri` as a white page.
 *
 * Keeping these here, importing nothing, makes the whole class impossible:
 * modules inside the barrel cycle can read them at init safely. `client.ts`
 * re-exports all four, so every existing import path still works.
 */

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
