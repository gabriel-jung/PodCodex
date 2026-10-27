import type { CorrectManualPromptsRequest, CorrectApplyManualRequest, CorrectApplyBatchesRequest, CorrectRequest, Segment, TaskResponse } from "./types";
import { createLLMPipelineApi } from "./versions";

type AudioRef = string | null | undefined;

const api = createLLMPipelineApi<{
  start: CorrectRequest;
  manualPrompts: CorrectManualPromptsRequest;
  applyManual: CorrectApplyManualRequest;
  applyBatches: CorrectApplyBatchesRequest;
}>("correct");

export const getCorrectSegments = (audioPath: AudioRef, outputDir?: string) =>
  api.getSegments(audioPath, { output_dir: outputDir });
export const getCorrectSegmentsPreview = (audioPath: AudioRef, limit: number, outputDir?: string) =>
  api.getSegmentsPreview(audioPath, limit, { output_dir: outputDir });
export const saveCorrectSegments = (audioPath: AudioRef, segments: Segment[], outputDir?: string) =>
  api.saveSegments(audioPath, segments, { output_dir: outputDir });
export const getCorrectVersions = (audioPath: AudioRef, outputDir?: string) =>
  api.getVersions(audioPath, { output_dir: outputDir });
export const loadCorrectVersion = (audioPath: AudioRef, versionId: string, outputDir?: string) =>
  api.loadVersion(audioPath, versionId, { output_dir: outputDir });
export const deleteCorrectVersion = (audioPath: AudioRef, versionId: string, outputDir?: string) =>
  api.deleteVersion(audioPath, versionId, { output_dir: outputDir });

export const startCorrect = (req: CorrectRequest): Promise<TaskResponse> => api.start(req);

export const getCorrectManualPrompts = (params: CorrectManualPromptsRequest) => api.getManualPrompts(params);

export const applyCorrectManual = (params: CorrectApplyManualRequest) => api.applyManual(params);

export const applyCorrectBatches = (params: CorrectApplyBatchesRequest) => api.applyBatches(params);
