import type { Segment, TaskResponse, TranscribeRequest, VersionEntry } from "./types";
import { BASE, json } from "./base";

function qs(audioPath: string, outputDir?: string) {
  const p = new URLSearchParams({ audio_path: audioPath });
  if (outputDir) p.set("output_dir", outputDir);
  return p.toString();
}

export const getSegments = (audioPath: string, outputDir?: string) =>
  json<Segment[]>(`/api/transcribe/segments?${qs(audioPath, outputDir)}`);

export const saveSegments = (audioPath: string, segments: Segment[], outputDir?: string) =>
  json<{ status: string; count: number }>(`/api/transcribe/segments?${qs(audioPath, outputDir)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segments),
  });

export const getTranscribeVersions = (audioPath: string, outputDir?: string) =>
  json<VersionEntry[]>(`/api/transcribe/versions?${qs(audioPath, outputDir)}`);

export const loadTranscribeVersion = (audioPath: string, versionId: string, outputDir?: string) =>
  json<Segment[]>(`/api/transcribe/versions/${encodeURIComponent(versionId)}?${qs(audioPath, outputDir)}`);

export const deleteTranscribeVersion = (audioPath: string, versionId: string, outputDir?: string) =>
  json<{ status: string }>(`/api/transcribe/versions/${encodeURIComponent(versionId)}?${qs(audioPath, outputDir)}`, { method: "DELETE" });

export const getSpeakerMap = (audioPath: string) =>
  json<Record<string, string>>(`/api/transcribe/speaker-map?audio_path=${encodeURIComponent(audioPath)}`);

export const saveSpeakerMap = (audioPath: string, mapping: Record<string, string>) =>
  json<{ status: string }>(`/api/transcribe/speaker-map?audio_path=${encodeURIComponent(audioPath)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(mapping),
  });

export const importTranscript = (audioPath: string, filePath: string, outputDir?: string) =>
  json<{ status: string; count: number }>(`/api/transcribe/import?${qs(audioPath, outputDir)}&file_path=${encodeURIComponent(filePath)}`, {
    method: "POST",
  });

export const startTranscribe = (req: TranscribeRequest) =>
  json<TaskResponse>("/api/transcribe/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export async function uploadTranscript(audioPath: string, file: File, outputDir?: string): Promise<{ status: string; count: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/transcribe/upload?${qs(audioPath, outputDir)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}
