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

/**
 * Query string identifying an episode to the backend: `audio_path` when the
 * episode has audio, `output_dir` when it does not (subtitle-only imports),
 * plus any endpoint-specific keys. The one builder every per-episode call
 * goes through, so the output_dir-only case cannot be forgotten by a new
 * endpoint copying the wrong template.
 */
export function episodeParams(
  audioPath: AudioRef,
  outputDir?: string | null,
  extra?: Extra,
): URLSearchParams {
  const params = new URLSearchParams();
  if (audioPath) params.set("audio_path", audioPath);
  if (outputDir) params.set("output_dir", outputDir);
  if (extra) {
    for (const [k, v] of Object.entries(extra)) {
      if (v !== undefined && v !== "") params.set(k, v);
    }
  }
  return params;
}

function build(path: string, audioPath: AudioRef, extra: Extra) {
  return `${path}?${episodeParams(audioPath, undefined, extra)}`;
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

/** Request body types of one LLM step's endpoints, from generated-types.ts,
 *  so a renamed or newly required backend field fails the typecheck instead
 *  of surfacing as a runtime 422. */
interface LLMStepRequests {
  start: object;
  manualPrompts: object;
  applyManual: object;
  applyBatches: object;
}

/** Extended factory for LLM pipeline steps (correct, translate) that also
 *  have start, manual-prompts, and apply-manual endpoints. */
export function createLLMPipelineApi<R extends LLMStepRequests>(step: string) {
  const post = <T>(path: string, body: unknown) =>
    json<T>(`/api/${step}/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

  return {
    ...createVersionApi(step),

    start: (req: R["start"]) =>
      post<TaskResponse>("start", req),

    getManualPrompts: (params: R["manualPrompts"]) =>
      post<{ batch_index: number; prompt: string; segment_count: number }[]>(
        "manual-prompts", params,
      ),

    applyManual: (params: R["applyManual"]) =>
      post<{ status: string; count: number }>("apply-manual", params),

    applyBatches: (params: R["applyBatches"]) =>
      post<{ status: string; count: number; rejected: number }>(
        "apply-batches", params,
      ),
  };
}
