import type {
  AboutResponse,
  ExtrasResponse,
  HealthResponse,
  ModelsResponse,
  OllamaCheckResponse,
  TaskResponse,
} from "./types";
import { BOOT_PATIENT_RETRY, json } from "./client";
import { queryKeys } from "./queryKeys";

export const getHealth = () => json<HealthResponse>("/api/health");

/**
 * Shared /api/health query options: use these everywhere instead of
 * re-declaring the query.
 *
 * This is the one query that waits a slow launch out. Everything else gives
 * up after a few connection retries (`CONNECT_RETRY`) and is refetched when
 * this one first succeeds, so the whole app does not run eighteen separate
 * retry ladders against the same closed port. React Query shares one fetch
 * per key: a stricter observer mounting first would abort the boot fetch for
 * the whole app, which is why the schedule lives here and not at call sites.
 *
 * Only connection failures are retried this long. A status the server
 * actually returned is a real failure and falls through to the default, so
 * the "Backend not reachable" screen still appears instead of hiding behind
 * three minutes of pointless retries.
 */
export const healthQueryOptions = {
  queryKey: queryKeys.health(),
  queryFn: getHealth,
  staleTime: 30_000,
  ...BOOT_PATIENT_RETRY,
} as const;

export const getAbout = () => json<AboutResponse>("/api/system/about");

export const getExtras = () => json<ExtrasResponse>("/api/system/extras");

export const checkOllama = () =>
  json<OllamaCheckResponse>("/api/system/ollama/check");

export const getActiveTask = (audioPath: string) =>
  json<{
    task_id: string;
    status: string;
    progress: number;
    message: string;
    steps?: string[];
    log?: string[];
    result?: Record<string, unknown>;
    error?: string;
  } | null>(
    `/api/tasks/active?audio_path=${encodeURIComponent(audioPath)}`,
  );

export const installExtra = (extra: string) =>
  json<TaskResponse>("/api/system/install-extra", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ extra }),
  });

export const removeExtra = (extra: string) =>
  json<TaskResponse>("/api/system/remove-extra", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ extra }),
  });

export const getTaskStatus = (taskId: string) =>
  json<{
    task_id: string;
    status: string;
    progress: number;
    message: string;
    steps?: string[];
    log?: string[];
    result?: Record<string, unknown>;
    error?: string;
  } | null>(`/api/tasks/${encodeURIComponent(taskId)}`);

export const cancelTask = (taskId: string) =>
  json<{ status: string; task_id: string }>(`/api/tasks/${encodeURIComponent(taskId)}/cancel`, {
    method: "POST",
  });

// ── Models ────────────────────────────────

export const getModels = () => json<ModelsResponse>("/api/models");

export const deleteModel = (modelId: string) =>
  json<{ status: string; model_id: string }>(`/api/models/${encodeURIComponent(modelId)}`, {
    method: "DELETE",
  });
