import type { PolishRequest, Segment, TaskResponse } from "./types";
import { json } from "./client";
import { createLLMPipelineApi } from "./versions";

const api = createLLMPipelineApi("polish");

export const getPolishSegments = (audioPath: string) => api.getSegments(audioPath);
export const savePolishSegments = (audioPath: string, segments: Segment[]) =>
  api.saveSegments(audioPath, segments);
export const getPolishVersions = (audioPath: string) => api.getVersions(audioPath);
export const loadPolishVersion = (audioPath: string, versionId: string) =>
  api.loadVersion(audioPath, versionId);
export const deletePolishVersion = (audioPath: string, versionId: string) =>
  api.deleteVersion(audioPath, versionId);

export const startPolish = (req: PolishRequest) =>
  api.start(req as unknown as Record<string, unknown>) as Promise<TaskResponse>;

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
}) => api.getManualPrompts(params);

export const applyPolishManual = (params: { audio_path: string; corrections: unknown[] }) =>
  api.applyManual(params);
