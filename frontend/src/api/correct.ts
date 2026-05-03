import type { CorrectRequest, Segment, TaskResponse } from "./types";
import { createLLMPipelineApi } from "./versions";

type AudioRef = string | null | undefined;

const api = createLLMPipelineApi("correct");

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
