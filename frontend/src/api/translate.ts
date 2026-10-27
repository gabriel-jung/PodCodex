import type { TranslateManualPromptsRequest, TranslateApplyManualRequest, TranslateApplyBatchesRequest, Segment, TaskResponse, TranslateRequest } from "./types";
import { createLLMPipelineApi } from "./versions";

type AudioRef = string | null | undefined;

const api = createLLMPipelineApi<{
  start: TranslateRequest;
  manualPrompts: TranslateManualPromptsRequest;
  applyManual: TranslateApplyManualRequest;
  applyBatches: TranslateApplyBatchesRequest;
}>("translate");

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

export const startTranslate = (req: TranslateRequest): Promise<TaskResponse> => api.start(req);

export const getTranslateManualPrompts = (params: TranslateManualPromptsRequest) => api.getManualPrompts(params);

export const applyTranslateManual = (params: TranslateApplyManualRequest) => api.applyManual(params);

export const applyTranslateBatches = (params: TranslateApplyBatchesRequest) => api.applyBatches(params);
