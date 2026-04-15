import type { CorrectRequest, Segment, TaskResponse } from "./types";
import { createLLMPipelineApi } from "./versions";

const api = createLLMPipelineApi("correct");

export const getCorrectSegments = (audioPath: string) => api.getSegments(audioPath);
export const getCorrectSegmentsPreview = (audioPath: string, limit: number) =>
  api.getSegmentsPreview(audioPath, limit);
export const saveCorrectSegments = (audioPath: string, segments: Segment[]) =>
  api.saveSegments(audioPath, segments);
export const getCorrectVersions = (audioPath: string) => api.getVersions(audioPath);
export const loadCorrectVersion = (audioPath: string, versionId: string) =>
  api.loadVersion(audioPath, versionId);
export const deleteCorrectVersion = (audioPath: string, versionId: string) =>
  api.deleteVersion(audioPath, versionId);

export const startCorrect = (req: CorrectRequest) =>
  api.start(req as unknown as Record<string, unknown>) as Promise<TaskResponse>;

export const getCorrectManualPrompts = (params: {
  audio_path?: string;
  output_dir?: string;
  context?: string;
  source_lang?: string;
  batch_minutes?: number;
  source_version_id?: string;
}) => api.getManualPrompts(params);

export const applyCorrectManual = (params: { audio_path?: string; output_dir?: string; corrections: unknown[] }) =>
  api.applyManual(params);
