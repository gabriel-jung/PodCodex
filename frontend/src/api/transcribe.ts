import type { Segment, TaskResponse, TranscribeRequest, VersionInfo } from "./types";
import { BASE, json } from "./base";

export const getSegments = (audioPath: string) =>
  json<Segment[]>(`/api/transcribe/segments?audio_path=${encodeURIComponent(audioPath)}`);

export const getSegmentsRaw = (audioPath: string) =>
  json<Segment[]>(`/api/transcribe/segments/raw?audio_path=${encodeURIComponent(audioPath)}`);

export const saveSegments = (audioPath: string, segments: Segment[]) =>
  json<{ status: string; count: number }>(`/api/transcribe/segments?audio_path=${encodeURIComponent(audioPath)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(segments),
  });

export const getTranscribeVersionInfo = (audioPath: string) =>
  json<VersionInfo>(`/api/transcribe/version-info?audio_path=${encodeURIComponent(audioPath)}`);

export const getSpeakerMap = (audioPath: string) =>
  json<Record<string, string>>(`/api/transcribe/speaker-map?audio_path=${encodeURIComponent(audioPath)}`);

export const saveSpeakerMap = (audioPath: string, mapping: Record<string, string>) =>
  json<{ status: string }>(`/api/transcribe/speaker-map?audio_path=${encodeURIComponent(audioPath)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(mapping),
  });

export const startTranscribe = (req: TranscribeRequest) =>
  json<TaskResponse>("/api/transcribe/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export async function uploadTranscript(audioPath: string, file: File): Promise<{ status: string; count: number }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/transcribe/upload?audio_path=${encodeURIComponent(audioPath)}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}
