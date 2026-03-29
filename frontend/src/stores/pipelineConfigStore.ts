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
        batchSize: 10,
        apiBaseUrl: "",
        apiKey: "",
      },
      setLLM: (patch) => set((s) => ({ llm: { ...s.llm, ...patch } })),

      engine: "Whisper",
      setEngine: (engine) => set({ engine }),

      targetLang: "English",
      setTargetLang: (targetLang) => set({ targetLang }),
    }),
    {
      name: "podcodex-pipeline-config",
      // Don't persist secrets
      partialize: (s) => ({
        transcribe: { ...s.transcribe, hfToken: "" },
        llm: { ...s.llm, apiKey: "" },
        engine: s.engine,
        targetLang: s.targetLang,
      }),
    },
  ),
);
