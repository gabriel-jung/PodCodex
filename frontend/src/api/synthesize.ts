import type {
  AssembleRequest,
  GenerateRequest,
  GeneratedSegment,
  SynthesisStatus,
  TaskResponse,
  VersionEntry,
  VoiceSample,
} from "./types";
import { json, rawFetch } from "./client";

function synthVersionsQuery(audioPath: string | null | undefined, outputDir?: string | null): string {
  const params = new URLSearchParams();
  if (audioPath) params.set("audio_path", audioPath);
  if (outputDir) params.set("output_dir", outputDir);
  return params.toString();
}

export const getSynthesisStatus = (audioPath: string | null | undefined, outputDir?: string | null) =>
  json<SynthesisStatus>(`/api/synthesize/status?${synthVersionsQuery(audioPath, outputDir)}`);

export const getVoiceSamples = (
  audioPath: string | null | undefined,
  outputDir?: string | null,
  sourceVersionId?: string | null,
) => {
  const params = new URLSearchParams(synthVersionsQuery(audioPath, outputDir));
  if (sourceVersionId) params.set("source_version_id", sourceVersionId);
  return json<Record<string, VoiceSample[]>>(`/api/synthesize/voice-samples?${params}`);
};

export async function uploadVoiceSample(
  audioPath: string | null | undefined,
  speaker: string,
  file: File,
  outputDir?: string | null,
): Promise<VoiceSample & { speaker: string }> {
  const form = new FormData();
  if (audioPath) form.append("audio_path", audioPath);
  if (outputDir) form.append("output_dir", outputDir);
  form.append("speaker", speaker);
  form.append("file", file);
  const res = await rawFetch(`/api/synthesize/upload-sample`, {
    method: "POST",
    body: form,
  });
  return res.json();
}

export const extractSelectedSamples = (
  audioPath: string | null,
  selections: { speaker: string; start: number; end: number; text: string }[],
  outputDir?: string | null,
) =>
  json<{ status: string; speakers: number; total_samples: number; samples: Record<string, VoiceSample[]> }>(
    "/api/synthesize/extract-selected",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ audio_path: audioPath, output_dir: outputDir, selections }),
    },
  );

export const startGenerateTTS = (req: GenerateRequest) =>
  json<TaskResponse>("/api/synthesize/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

export const getGeneratedSegments = (
  audioPath: string | null | undefined,
  outputDir?: string | null,
  sourceVersionId?: string | null,
) => {
  const params = new URLSearchParams(synthVersionsQuery(audioPath, outputDir));
  if (sourceVersionId) params.set("source_version_id", sourceVersionId);
  return json<GeneratedSegment[]>(`/api/synthesize/generated-segments?${params}`);
};

export const assembleEpisode = (req: AssembleRequest) =>
  json<{ path: string; duration: number; version_id: string }>("/api/synthesize/assemble", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

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
