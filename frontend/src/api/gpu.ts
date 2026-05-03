import { json } from "./client";
import type { TaskResponse } from "./types";

export interface GPUStatus {
  mode: "bundle" | "dev";
  current_backend: "cpu" | "gpu" | "missing";
  gpu_detected: boolean;
  gpu_name: string | null;
  vram_mb: number | null;
  /** cuda-libs archive version (e.g. "cu128-v1") of the installed bundle. */
  installed_version: string | null;
  /** Server-core version reported by the installed binary's --version. */
  installed_server_version: string | null;
  /** Running app version — used to detect server-core staleness vs the app. */
  app_version: string;
  activated: boolean;
  install_dir: string;
  /** Whether this OS can install the GPU backend (Windows-only today). */
  platform_supported: boolean;
  /** True when an install exists but its server-core version trails the app —
   *  the Tauri shell has silently fallen back to the bundled CPU sidecar. */
  needs_update: boolean;
}

export const getGPUStatus = () => json<GPUStatus>("/api/gpu/status");

export const downloadGPUBackend = (manifest_url?: string) =>
  json<TaskResponse>("/api/gpu/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ manifest_url: manifest_url ?? null }),
  });

export const activateGPUBackend = () =>
  json<{ activated: boolean; restart_required: boolean }>(
    "/api/gpu/activate",
    { method: "POST" },
  );

export const deactivateGPUBackend = () =>
  json<{ activated: boolean; restart_required: boolean }>(
    "/api/gpu/deactivate",
    { method: "POST" },
  );

export const uninstallGPUBackend = () =>
  json<{ installed: boolean }>("/api/gpu/uninstall", { method: "POST" });

// ── Device override ─────────────────────────────────────────────────────

export type DeviceOverride = "auto" | "cpu" | "cuda";

export interface DeviceInfo {
  /** Active env override in the running process (null when "auto"). */
  override: "cpu" | "cuda" | null;
  /** Effective CUDA availability after override. */
  available: boolean;
  /** Resolved device passed to faster-whisper / pyannote: "cpu" | "cuda". */
  device: string;
  /** CTranslate2 compute_type chosen for the GPU class. */
  compute_type: string;
  /** torch dtype string (without the "torch." prefix). */
  torch_dtype?: string;
  /** "8.0", "6.1", … — only present when a CUDA device is detected. */
  compute_capability?: string;
  gpu_name?: string;
  arch_list?: string[];
  /** Populated when device_info() caught a resolve error. */
  error?: string;
  /** What is saved on disk for next launch. */
  persisted_override: DeviceOverride;
}

export const getDeviceInfo = () => json<DeviceInfo>("/api/system/device");

export const setDeviceOverride = (override: DeviceOverride) =>
  json<DeviceInfo & { restart_required: boolean }>(
    "/api/system/device",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ override }),
    },
  );
