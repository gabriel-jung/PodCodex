/** Persisted pipeline configuration — shared between episode panels and batch modal. */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { LLMConfig } from "@/components/common/LLMControls";

export interface TranscribeConfig {
  modelSize: string;
  batchSize: number;
  diarize: boolean;
  hfToken: string;
  numSpeakers: string;
}

// ── Pipeline presets ─────────────────────────────────────

export interface PipelinePreset {
  label: string;
  desc: string;
  whisperModel: string;
  embedModel: string;
}

export const PIPELINE_PRESETS: Record<string, PipelinePreset> = {
  heavy: { label: "Heavy", desc: "8 GB+ GPU", whisperModel: "large-v3", embedModel: "bge-m3" },
  medium: { label: "Medium", desc: "4 GB+ GPU", whisperModel: "large-v3-turbo", embedModel: "bge-m3" },
  light: { label: "Light", desc: "CPU ok", whisperModel: "small", embedModel: "bge-m3" },
};

// ── Pipeline config state ────────────────────────────────

export interface PipelineConfigState {
  // Transcribe
  transcribe: TranscribeConfig;
  setTranscribe: (patch: Partial<TranscribeConfig>) => void;

  // LLM (polish + translate shared)
  llm: LLMConfig;
  setLLM: (patch: Partial<LLMConfig>) => void;

  // Polish-specific
  engine: string;
  setEngine: (engine: string) => void;

  // Translate-specific
  targetLang: string;
  setTargetLang: (lang: string) => void;

  // Preset (simple mode)
  preset: string;
  setPreset: (preset: string) => void;
}

export const usePipelineConfigStore = create<PipelineConfigState>()(
  persist(
    (set) => ({
      transcribe: {
        modelSize: "large-v3-turbo",
        batchSize: 16,
        diarize: true,
        hfToken: "",
        numSpeakers: "",
      },
      setTranscribe: (patch) =>
        set((s) => ({ transcribe: { ...s.transcribe, ...patch } })),

      llm: {
        mode: "ollama",
        provider: "openai",
        model: "",
        context: "",
        sourceLang: "French",
        batchMinutes: 15,
        apiBaseUrl: "",
        apiKey: "",
      },
      setLLM: (patch) => set((s) => ({ llm: { ...s.llm, ...patch } })),

      engine: "Whisper",
      setEngine: (engine) => set({ engine }),

      targetLang: "English",
      setTargetLang: (targetLang) => set({ targetLang }),

      preset: "medium",
      setPreset: (preset) => set({ preset }),
    }),
    {
      name: "podcodex-pipeline-config",
      // Don't persist secrets
      partialize: (s) => ({
        transcribe: { ...s.transcribe, hfToken: "" },
        llm: { ...s.llm, apiKey: "" },
        engine: s.engine,
        targetLang: s.targetLang,
        preset: s.preset,
      }),
    },
  ),
);
