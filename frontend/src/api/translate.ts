import type { Segment, TaskResponse, TranslateRequest, VersionEntry } from "./types";
import { json } from "./base";

export const getTranslateSegments = (audioPath: string, lang: string) =>
  json<Segment[]>(`/api/translate/segments?audio_path=${encodeURIComponent(audioPath)}&lang=${encodeURIComponent(lang)}`);

export const saveTranslateSegments = (audioPath: string, lang: string, segments: Segment[]) =>
  json<{ status: string; count: number }>(`/api/translate/segments?audio_path=${encodeURIComponent(audioPath)}&lang=${encodeURIComponent(lang)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segments),
  });

export const getTranslateVersions = (audioPath: string, lang: string) =>
  json<VersionEntry[]>(`/api/translate/versions?audio_path=${encodeURIComponent(audioPath)}&lang=${encodeURIComponent(lang)}`);

export const loadTranslateVersion = (audioPath: string, lang: string, versionId: string) =>
  json<Segment[]>(`/api/translate/versions/${encodeURIComponent(versionId)}?audio_path=${encodeURIComponent(audioPath)}&lang=${encodeURIComponent(lang)}`);

export const deleteTranslateVersion = (audioPath: string, lang: string, versionId: string) =>
  json<{ status: string }>(`/api/translate/versions/${encodeURIComponent(versionId)}?audio_path=${encodeURIComponent(audioPath)}&lang=${encodeURIComponent(lang)}`, { method: "DELETE" });

export const getTranslateLanguages = (audioPath: string) =>
  json<string[]>(`/api/translate/languages?audio_path=${encodeURIComponent(audioPath)}`);

export const startTranslate = (req: TranslateRequest) =>
  json<TaskResponse>("/api/translate/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const getTranslateManualPrompts = (params: {
  audio_path: string;
  context?: string;
  source_lang?: string;
  target_lang?: string;
  batch_minutes?: number;
}) =>
  json<{ batch_index: number; prompt: string; segment_count: number }[]>(
    "/api/translate/manual-prompts",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) },
  );

export const applyTranslateManual = (params: { audio_path: string; lang: string; corrections: unknown[] }) =>
  json<{ status: string; count: number }>("/api/translate/apply-manual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
