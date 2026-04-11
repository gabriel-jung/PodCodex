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

export const CPU_LABELS: Record<string, string> = { base: "Fastest", small: "Slightly more accurate" };
export const GPU_LABELS: Record<string, string> = { "large-v3": "Slightly more accurate, 2-3x slower", "large-v3-turbo": "Fast, near-best quality", medium: "Lighter, still good" };
export const CPU_MODELS = new Set(Object.keys(CPU_LABELS));
export const GPU_MODELS = new Set(Object.keys(GPU_LABELS));

export const TRANSCRIBE_PRESETS = {
  cpu: { label: "CPU", desc: "Lightweight, no GPU", modelSize: "base" },
  gpu: { label: "GPU", desc: "More accurate, requires GPU", modelSize: "large-v3-turbo" },
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

  // Per-step presets — `apply*Preset` atomically updates the underlying
  // config fields AND the preset key, avoiding the "reset then restore" dance.
  transcribePreset: string;
  applyTranscribePreset: (key: keyof typeof TRANSCRIBE_PRESETS) => void;
  llmPreset: string;
  applyLLMPreset: (key: keyof typeof LLM_PRESETS) => void;
  indexPreset: string;
  applyIndexPreset: (key: keyof typeof INDEX_PRESETS) => void;
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
          // Only reset preset when the model changes (what presets control)
          transcribePreset: "modelSize" in patch ? "" : s.transcribePreset,
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
      applyTranscribePreset: (key) =>
        set((s) => {
          const p = TRANSCRIBE_PRESETS[key];
          return p
            ? { transcribe: { ...s.transcribe, modelSize: p.modelSize }, transcribePreset: key }
            : s;
        }),
      llmPreset: "manual",
      applyLLMPreset: (key) =>
        set((s) => {
          const p = LLM_PRESETS[key];
          return p ? { llm: { ...s.llm, mode: p.mode }, llmPreset: key } : s;
        }),
      indexPreset: "balanced",
      applyIndexPreset: (key) =>
        set(() => {
          const p = INDEX_PRESETS[key];
          return p ? { indexModel: p.model, indexPreset: key } : {};
        }),
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
