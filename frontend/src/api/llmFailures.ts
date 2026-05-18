/** Client for the per-batch LLM outcome records (llm_failures.json).
 *  An auto correct/translate run records every batch here; a rejected batch
 *  silently kept its original text, so the user can inspect and fix it. */

import { json } from "./client";

export interface LlmBatchRecord {
  batch: number;
  status: "ok" | "rejected";
  reason: string;
  expected: number;
  got: number;
  raw: string;
  input: { index: number; text: string }[];
}

export interface LlmFailures {
  timestamp: string;
  model: string;
  mode: string;
  total_batches: number;
  rejected: number;
  batches: LlmBatchRecord[];
}

type AudioRef = string | null | undefined;

function qs(audioPath: AudioRef, extra: Record<string, string | undefined>): string {
  const p = new URLSearchParams();
  if (audioPath) p.set("audio_path", audioPath);
  for (const [k, v] of Object.entries(extra)) {
    if (v) p.set(k, v);
  }
  return p.toString();
}

export const getCorrectFailures = (audioPath: AudioRef, outputDir?: string) =>
  json<LlmFailures | null>(`/api/correct/llm-failures?${qs(audioPath, { output_dir: outputDir })}`);

export const dismissCorrectFailures = (audioPath: AudioRef, outputDir?: string) =>
  json<{ cleared: boolean }>(
    `/api/correct/llm-failures?${qs(audioPath, { output_dir: outputDir })}`,
    { method: "DELETE" },
  );

export const getTranslateFailures = (audioPath: AudioRef, lang: string, outputDir?: string) =>
  json<LlmFailures | null>(
    `/api/translate/llm-failures?${qs(audioPath, { output_dir: outputDir, lang })}`,
  );

export const dismissTranslateFailures = (audioPath: AudioRef, lang: string, outputDir?: string) =>
  json<{ cleared: boolean }>(
    `/api/translate/llm-failures?${qs(audioPath, { output_dir: outputDir, lang })}`,
    { method: "DELETE" },
  );
