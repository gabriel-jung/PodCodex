/**
 * Shared version API factory for steps that store segments + versions
 * (transcribe, correct, translate). Each step has the same five endpoints
 * — the only variation is per-step extra query params (lang for translate, etc.).
 *
 * audioPath can be a real audio file path or an output_dir for episodes
 * without audio (e.g. YouTube subtitle imports). The build function sends
 * output_dir when audio_path is absent.
 */

import type { Segment, TaskResponse, VersionEntry } from "./types";
import { json } from "./client";

type Extra = Record<string, string | undefined> | undefined;

function build(path: string, audioPath: string, extra: Extra) {
  const params = new URLSearchParams();
  params.set("audio_path", audioPath);
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      if (v !== undefined && v !== "") params.set(k, v);
    }
  }
  return `${path}?${params}`;
}

export function createVersionApi(step: string) {
  const seg = (audioPath: string, extra?: Extra) =>
    build(`/api/${step}/segments`, audioPath, extra);
  const ver = (audioPath: string, extra?: Extra) =>
    build(`/api/${step}/versions`, audioPath, extra);
  const verId = (audioPath: string, id: string, extra?: Extra) =>
    build(`/api/${step}/versions/${encodeURIComponent(id)}`, audioPath, extra);

  return {
    getSegments: (audioPath: string, extra?: Extra) =>
      json<Segment[]>(seg(audioPath, extra)),

    getSegmentsPreview: (audioPath: string, limit: number, extra?: Extra) =>
      json<Segment[]>(seg(audioPath, { ...extra, limit: String(limit) })),

    saveSegments: (audioPath: string, segments: Segment[], extra?: Extra) =>
      json<{ status: string; count: number }>(seg(audioPath, extra), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(segments),
      }),

    getVersions: (audioPath: string, extra?: Extra) =>
      json<VersionEntry[]>(ver(audioPath, extra)),

    loadVersion: (audioPath: string, id: string, extra?: Extra) =>
      json<Segment[]>(verId(audioPath, id, extra)),

    deleteVersion: (audioPath: string, id: string, extra?: Extra) =>
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
