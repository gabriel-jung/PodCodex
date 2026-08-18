import type {
  AboutResponse,
  ExtrasResponse,
  HealthResponse,
  ModelsResponse,
  OllamaCheckResponse,
  TaskResponse,
} from "./types";
import { json } from "./client";
import { queryKeys } from "./queryKeys";

export const getHealth = () => json<HealthResponse>("/api/health");

/**
 * Shared /api/health query options: use these everywhere instead of
 * re-declaring the query.
 *
 * The PyInstaller sidecar can take 10-30 s to extract and boot on the first
 * launch of a session, so every observer must agree on the generous retry
 * schedule. React Query shares one fetch per key: a stricter observer
 * mounting first would abort the boot fetch for the whole app.
 */
export const healthQueryOptions = {
  queryKey: queryKeys.health(),
  queryFn: getHealth,
  staleTime: 30_000,
  retry: 60,
  retryDelay: (attempt: number) => Math.min(500 + attempt * 500, 3000),
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
