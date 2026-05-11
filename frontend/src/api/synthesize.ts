import type {
  AssembleRequest,
  ExtractVoicesRequest,
  GenerateRequest,
  GeneratedSegment,
  SynthesisStatus,
  TaskResponse,
  VersionEntry,
  VoiceSample,
} from "./types";
import { json, rawFetch } from "./client";

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
  const res = await rawFetch(`/api/synthesize/upload-sample`, {
    method: "POST",
    body: form,
  });
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
  json<{ path: string; duration: number; version_id: string }>("/api/synthesize/assemble", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

function synthVersionsQuery(audioPath: string | null | undefined, outputDir?: string | null): string {
  const params = new URLSearchParams();
  if (audioPath) params.set("audio_path", audioPath);
  if (outputDir) params.set("output_dir", outputDir);
  return params.toString();
}

export const getSynthesizeVersions = (audioPath: string | null | undefined, outputDir?: string | null) =>
  json<VersionEntry[]>(`/api/synthesize/versions?${synthVersionsQuery(audioPath, outputDir)}`);

export const getSynthesizeVersionPath = (audioPath: string | null | undefined, versionId: string, outputDir?: string | null) =>
  json<{ path: string; duration: number; version_id: string }>(
    `/api/synthesize/versions/${encodeURIComponent(versionId)}?${synthVersionsQuery(audioPath, outputDir)}`,
  );

export const deleteSynthesizeVersion = (audioPath: string | null | undefined, versionId: string, outputDir?: string | null) =>
  json<{ status: string; version_id: string }>(
    `/api/synthesize/versions/${encodeURIComponent(versionId)}?${synthVersionsQuery(audioPath, outputDir)}`,
    { method: "DELETE" },
  );
