/** Persisted pipeline configuration — shared between episode panels and batch modal. */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { LLMConfig } from "@/components/common/LLMControls";

export interface TranscribeConfig {
  modelSize: string;
  batchSize: number;
  diarize: boolean;
  clean: boolean;
  hfToken: string;
  numSpeakers: string;
}

// ── Per-step presets ────────────────────────────────────

export const TRANSCRIBE_PRESETS = {
  cpu: { label: "CPU", desc: "No GPU needed", modelSize: "base", diarize: false },
  gpu: { label: "GPU", desc: "Fast & accurate", modelSize: "large-v3-turbo", diarize: false },
  "gpu-speakers": { label: "GPU + Speakers", desc: "Detect who's talking", modelSize: "large-v3-turbo", diarize: true },
} as const;

export const LLM_PRESETS = {
  local: { label: "Local", desc: "Ollama (free, needs GPU)", mode: "ollama" as const },
  cloud: { label: "Cloud", desc: "API (fast, needs key)", mode: "api" as const },
  manual: { label: "Manual", desc: "Copy prompts yourself", mode: "manual" as const },
} as const;

export const INDEX_PRESETS = {
  fast: { label: "Fast", desc: "Lightweight, CPU ok", model: "e5-small" },
  balanced: { label: "Balanced", desc: "Best all-round", model: "bge-m3" },
  gpu: { label: "GPU", desc: "Best quality, slow on CPU", model: "pplx" },
} as const;

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

  // Index
  indexModel: string;
  setIndexModel: (model: string) => void;

  // Per-step presets
  transcribePreset: string;
  setTranscribePreset: (preset: string) => void;
  llmPreset: string;
  setLLMPreset: (preset: string) => void;
  indexPreset: string;
  setIndexPreset: (preset: string) => void;
}

export const usePipelineConfigStore = create<PipelineConfigState>()(
  persist(
    (set) => ({
      transcribe: {
        modelSize: "large-v3-turbo",
        batchSize: 16,
        diarize: true,
        clean: false,
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

      indexModel: "bge-m3",
      setIndexModel: (indexModel) => set({ indexModel }),

      transcribePreset: "gpu",
      setTranscribePreset: (transcribePreset) => set({ transcribePreset }),
      llmPreset: "local",
      setLLMPreset: (llmPreset) => set({ llmPreset }),
      indexPreset: "balanced",
      setIndexPreset: (indexPreset) => set({ indexPreset }),
    }),
    {
      name: "podcodex-pipeline-config",
      // Don't persist secrets
      partialize: (s) => ({
        transcribe: { ...s.transcribe, hfToken: "" },
        llm: { ...s.llm, apiKey: "" },
        engine: s.engine,
        targetLang: s.targetLang,
        indexModel: s.indexModel,
        transcribePreset: s.transcribePreset,
        llmPreset: s.llmPreset,
        indexPreset: s.indexPreset,
      }),
    },
  ),
);
