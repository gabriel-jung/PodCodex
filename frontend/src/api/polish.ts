import type { PolishRequest, Segment, TaskResponse, VersionEntry } from "./types";
import { json } from "./base";

export const getPolishSegments = (audioPath: string) =>
  json<Segment[]>(`/api/polish/segments?audio_path=${encodeURIComponent(audioPath)}`);

export const savePolishSegments = (audioPath: string, segments: Segment[]) =>
  json<{ status: string; count: number }>(`/api/polish/segments?audio_path=${encodeURIComponent(audioPath)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segments),
  });

export const getPolishVersions = (audioPath: string) =>
  json<VersionEntry[]>(`/api/polish/versions?audio_path=${encodeURIComponent(audioPath)}`);

export const loadPolishVersion = (audioPath: string, versionId: string) =>
  json<Segment[]>(`/api/polish/versions/${encodeURIComponent(versionId)}?audio_path=${encodeURIComponent(audioPath)}`);

export const startPolish = (req: PolishRequest) =>
  json<TaskResponse>("/api/polish/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const skipPolish = (audioPath: string) =>
  json<{ status: string; count: number }>("/api/polish/skip", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ audio_path: audioPath }),
  });

export const getPolishManualPrompts = (params: {
  audio_path: string;
  context?: string;
  source_lang?: string;
  batch_minutes?: number;
  engine?: string;
}) =>
  json<{ batch_index: number; prompt: string; segment_count: number }[]>(
    "/api/polish/manual-prompts",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) },
  );

export const applyPolishManual = (params: { audio_path: string; corrections: unknown[] }) =>
  json<{ status: string; count: number }>("/api/polish/apply-manual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
