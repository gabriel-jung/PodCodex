import type { Segment, TaskResponse, TranslateRequest } from "./types";
import { createLLMPipelineApi } from "./versions";

const api = createLLMPipelineApi("translate");

export const getTranslateSegments = (audioPath: string, lang: string) =>
  api.getSegments(audioPath, { lang });
export const saveTranslateSegments = (audioPath: string, lang: string, segments: Segment[]) =>
  api.saveSegments(audioPath, segments, { lang });
export const getTranslateVersions = (audioPath: string, lang: string) =>
  api.getVersions(audioPath, { lang });
export const loadTranslateVersion = (audioPath: string, lang: string, versionId: string) =>
  api.loadVersion(audioPath, versionId, { lang });
export const deleteTranslateVersion = (audioPath: string, lang: string, versionId: string) =>
  api.deleteVersion(audioPath, versionId, { lang });

export const startTranslate = (req: TranslateRequest) =>
  api.start(req as unknown as Record<string, unknown>) as Promise<TaskResponse>;

export const getTranslateManualPrompts = (params: {
  audio_path?: string;
  output_dir?: string;
  context?: string;
  source_lang?: string;
  target_lang?: string;
  batch_minutes?: number;
  source_version_id?: string;
}) => api.getManualPrompts(params);

export const applyTranslateManual = (params: { audio_path?: string; output_dir?: string; lang: string; corrections: unknown[] }) =>
  api.applyManual(params);
