import type { Segment, TaskResponse, TranscribeRequest } from "./types";
import { json, rawFetch } from "./client";
import { createVersionApi } from "./versions";

type AudioRef = string | null | undefined;

const api = createVersionApi("transcribe");

export const getSegments = (audioPath: AudioRef, outputDir?: string) =>
  api.getSegments(audioPath, { output_dir: outputDir });
export const getSegmentsPreview = (audioPath: AudioRef, limit: number, outputDir?: string) =>
  api.getSegmentsPreview(audioPath, limit, { output_dir: outputDir });
export const saveSegments = (audioPath: AudioRef, segments: Segment[], outputDir?: string) =>
  api.saveSegments(audioPath, segments, { output_dir: outputDir });
export const getTranscribeVersions = (audioPath: AudioRef, outputDir?: string) =>
  api.getVersions(audioPath, { output_dir: outputDir });
export const loadTranscribeVersion = (audioPath: AudioRef, versionId: string, outputDir?: string) =>
  api.loadVersion(audioPath, versionId, { output_dir: outputDir });
export const deleteTranscribeVersion = (audioPath: AudioRef, versionId: string, outputDir?: string) =>
  api.deleteVersion(audioPath, versionId, { output_dir: outputDir });

function qs(audioPath: AudioRef, outputDir?: string) {
  const p = new URLSearchParams();
  if (audioPath) p.set("audio_path", audioPath);
  if (outputDir) p.set("output_dir", outputDir);
  return p.toString();
}

export const getSpeakerMap = (audioPath: AudioRef, outputDir?: string) =>
  json<Record<string, string>>(`/api/transcribe/speaker-map?${qs(audioPath, outputDir)}`);

export const saveSpeakerMap = (audioPath: AudioRef, mapping: Record<string, string>, outputDir?: string) =>
  json<{ status: string }>(`/api/transcribe/speaker-map?${qs(audioPath, outputDir)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(mapping),
  });

export const importTranscript = (audioPath: AudioRef, filePath: string, outputDir?: string) =>
  json<{ status: string; count: number }>(`/api/transcribe/import?${qs(audioPath, outputDir)}&file_path=${encodeURIComponent(filePath)}`, {
    method: "POST",
  });

export const startTranscribe = (req: TranscribeRequest) =>
  json<TaskResponse>("/api/transcribe/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export async function uploadTranscript(audioPath: AudioRef, file: File, outputDir?: string): Promise<{ status: string; count: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await rawFetch(`/api/transcribe/upload?${qs(audioPath, outputDir)}`, {
    method: "POST",
    body: form,
  });
  return res.json();
}
