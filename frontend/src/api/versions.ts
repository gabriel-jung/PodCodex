/**
 * Shared version API factory for steps that store segments + versions
 * (transcribe, correct, translate). Each step has the same five endpoints
 * — the only variation is per-step extra query params (lang for translate, etc.).
 *
 * Either audioPath or output_dir (passed via extra) must be set. Backend
 * resolves the episode root from whichever is provided; output-dir-only is
 * the path for episodes without audio (e.g. YouTube subtitle imports).
 */

import type { Segment, TaskResponse, VersionEntry } from "./types";
import { json } from "./client";

type Extra = Record<string, string | undefined> | undefined;
type AudioRef = string | null | undefined;

function build(path: string, audioPath: AudioRef, extra: Extra) {
  const params = new URLSearchParams();
  if (audioPath) params.set("audio_path", audioPath);
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      if (v !== undefined && v !== "") params.set(k, v);
    }
  }
  return `${path}?${params}`;
}

export function createVersionApi(step: string) {
  const seg = (audioPath: AudioRef, extra?: Extra) =>
    build(`/api/${step}/segments`, audioPath, extra);
  const ver = (audioPath: AudioRef, extra?: Extra) =>
    build(`/api/${step}/versions`, audioPath, extra);
  const verId = (audioPath: AudioRef, id: string, extra?: Extra) =>
    build(`/api/${step}/versions/${encodeURIComponent(id)}`, audioPath, extra);

  return {
    getSegments: (audioPath: AudioRef, extra?: Extra) =>
      json<Segment[]>(seg(audioPath, extra)),

    getSegmentsPreview: (audioPath: AudioRef, limit: number, extra?: Extra) =>
      json<Segment[]>(seg(audioPath, { ...extra, limit: String(limit) })),

    saveSegments: (audioPath: AudioRef, segments: Segment[], extra?: Extra) =>
      json<{ status: string; count: number }>(seg(audioPath, extra), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(segments),
      }),

    getVersions: (audioPath: AudioRef, extra?: Extra) =>
      json<VersionEntry[]>(ver(audioPath, extra)),

    loadVersion: (audioPath: AudioRef, id: string, extra?: Extra) =>
      json<Segment[]>(verId(audioPath, id, extra)),

    deleteVersion: (audioPath: AudioRef, id: string, extra?: Extra) =>
      json<{ status: string }>(verId(audioPath, id, extra), { method: "DELETE" }),
  };
}

/** Extended factory for LLM pipeline steps (correct, translate) that also
 *  have start, manual-prompts, and apply-manual endpoints. */
export function createLLMPipelineApi(step: string) {
  const post = <T>(path: string, body: unknown) =>
    json<T>(`/api/${step}/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

  return {
    ...createVersionApi(step),

    start: (req: Record<string, unknown>) =>
      post<TaskResponse>("start", req),

    getManualPrompts: (params: Record<string, unknown>) =>
      post<{ batch_index: number; prompt: string; segment_count: number }[]>(
        "manual-prompts", params,
      ),

    applyManual: (params: Record<string, unknown>) =>
      post<{ status: string; count: number }>("apply-manual", params),
  };
}
