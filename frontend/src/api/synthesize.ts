import type {
  AssembleRequest,
  ExtractVoicesRequest,
  GenerateRequest,
  GeneratedSegment,
  SynthesisStatus,
  TaskResponse,
  VoiceSample,
} from "./types";
import { BASE, json } from "./client";

export const getSynthesisStatus = (audioPath: string) =>
  json<SynthesisStatus>(`/api/synthesize/status?audio_path=${encodeURIComponent(audioPath)}`);

export const startExtractVoices = (req: ExtractVoicesRequest) =>
  json<TaskResponse>("/api/synthesize/extract-voices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const getVoiceSamples = (audioPath: string) =>
  json<Record<string, VoiceSample[]>>(`/api/synthesize/voice-samples?audio_path=${encodeURIComponent(audioPath)}`);

export async function uploadVoiceSample(audioPath: string, speaker: string, file: File): Promise<VoiceSample & { speaker: string }> {
  const form = new FormData();
  form.append("audio_path", audioPath);
  form.append("speaker", speaker);
  form.append("file", file);
  const res = await fetch(`${BASE}/api/synthesize/upload-sample`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

export const extractSelectedSamples = (audioPath: string, selections: { speaker: string; start: number; end: number; text: string }[]) =>
  json<{ status: string; speakers: number; total_samples: number; samples: Record<string, VoiceSample[]> }>(
    "/api/synthesize/extract-selected",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio_path: audioPath, selections }),
    },
  );

export const startGenerateTTS = (req: GenerateRequest) =>
  json<TaskResponse>("/api/synthesize/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const getGeneratedSegments = (audioPath: string) =>
  json<GeneratedSegment[]>(`/api/synthesize/generated-segments?audio_path=${encodeURIComponent(audioPath)}`);

export const assembleEpisode = (req: AssembleRequest) =>
  json<{ path: string; duration: number }>("/api/synthesize/assemble", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
