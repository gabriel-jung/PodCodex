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
  language: string;
}

// ── Per-step presets ────────────────────────────────────

export const CPU_MODELS = new Set(["base", "small"]);

export const TRANSCRIBE_PRESETS = {
  cpu: { label: "CPU", desc: "Lightweight, no GPU", modelSize: "base", diarize: false },
  gpu: { label: "GPU", desc: "Fast & accurate", modelSize: "large-v3-turbo", diarize: false },
  "gpu-speakers": { label: "GPU + Speakers", desc: "Detect who's talking", modelSize: "large-v3-turbo", diarize: true },
} as const;

export const LLM_PRESETS = {
  manual: { label: "Manual", desc: "Paste into any LLM chatbot", mode: "manual" as const },
  local: { label: "Local", desc: "Run via Ollama, GPU required", mode: "ollama" as const },
  cloud: { label: "Cloud", desc: "Use any LLM with API key", mode: "api" as const },
} as const;

export const INDEX_PRESETS = {
  fast: { label: "Fast", desc: "Very fast to run", model: "e5-small" },
  balanced: { label: "Balanced", desc: "Default, good search quality", model: "bge-m3" },
  gpu: { label: "GPU", desc: "Context-aware, slow on CPU", model: "pplx" },
} as const;

// ── Pipeline config state ────────────────────────────────

export interface PipelineConfigState {
  // Transcribe
  transcribe: TranscribeConfig;
  setTranscribe: (patch: Partial<TranscribeConfig>) => void;

  // LLM (correct + translate shared)
  llm: LLMConfig;
  setLLM: (patch: Partial<LLMConfig>) => void;

  // Correct-specific: transcript source override ("" = auto-detect from provenance)
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
        language: "",
      },
      setTranscribe: (patch) =>
        set((s) => ({
          transcribe: { ...s.transcribe, ...patch },
          // Only reset preset when model/diarize change (what presets control)
          transcribePreset: ("modelSize" in patch || "diarize" in patch) ? "" : s.transcribePreset,
        })),

      llm: {
        mode: "manual",
        provider: "openai",
        model: "",
        context: "",
        sourceLang: "French",
        batchMinutes: 15,
        apiBaseUrl: "",
        apiKey: "",
      },
      setLLM: (patch) => set((s) => ({ llm: { ...s.llm, ...patch }, llmPreset: "" })),

      engine: "",
      setEngine: (engine) => set({ engine }),

      targetLang: "English",
      setTargetLang: (targetLang) => set({ targetLang }),

      indexModel: "bge-m3",
      setIndexModel: (indexModel) => set({ indexModel, indexPreset: "" }),

      transcribePreset: "gpu",
      setTranscribePreset: (transcribePreset) => set({ transcribePreset }),
      llmPreset: "manual",
      setLLMPreset: (llmPreset) => set({ llmPreset }),
      indexPreset: "balanced",
      setIndexPreset: (indexPreset) => set({ indexPreset }),
    }),
    {
      name: "podcodex-pipeline-config",
      version: 1,
      migrate(persisted: unknown, fromVersion: number) {
        const s = persisted as Record<string, unknown>;
        if (fromVersion < 1) {
          const tc = s.transcribe as Record<string, unknown> | undefined;
          if (tc && tc.clean === undefined) tc.clean = false;
          // Ensure new preset fields exist
          if (!s.transcribePreset) s.transcribePreset = "";
          if (!s.llmPreset) s.llmPreset = "";
          if (!s.indexPreset) s.indexPreset = "";
          if (!s.indexModel) s.indexModel = "bge-m3";
        }
        return s as PipelineConfigState;
      },
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
