import type { ExtrasResponse, HealthResponse, ModelsResponse, TaskResponse } from "./types";
import { json } from "./client";

export const getHealth = () => json<HealthResponse>("/api/health");

export const getExtras = () => json<ExtrasResponse>("/api/system/extras");

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
