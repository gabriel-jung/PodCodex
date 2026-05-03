import type { Segment, TaskResponse, TranslateRequest } from "./types";
import { createLLMPipelineApi } from "./versions";

type AudioRef = string | null | undefined;

const api = createLLMPipelineApi("translate");

export const getTranslateSegments = (audioPath: AudioRef, lang: string, outputDir?: string) =>
  api.getSegments(audioPath, { lang, output_dir: outputDir });
export const saveTranslateSegments = (audioPath: AudioRef, lang: string, segments: Segment[], outputDir?: string) =>
  api.saveSegments(audioPath, segments, { lang, output_dir: outputDir });
export const getTranslateVersions = (audioPath: AudioRef, lang: string, outputDir?: string) =>
  api.getVersions(audioPath, { lang, output_dir: outputDir });
export const loadTranslateVersion = (audioPath: AudioRef, lang: string, versionId: string, outputDir?: string) =>
  api.loadVersion(audioPath, versionId, { lang, output_dir: outputDir });
export const deleteTranslateVersion = (audioPath: AudioRef, lang: string, versionId: string, outputDir?: string) =>
  api.deleteVersion(audioPath, versionId, { lang, output_dir: outputDir });

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
