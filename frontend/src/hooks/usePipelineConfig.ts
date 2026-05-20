/** Convenience hooks that bundle pipeline config selectors from the store. */

import { useMemo } from "react";
import { usePipelineConfigStore } from "@/stores";

export function usePipelineConfig() {
  const tc = usePipelineConfigStore((s) => s.transcribe);
  const setTc = usePipelineConfigStore((s) => s.setTranscribe);
  const llm = usePipelineConfigStore((s) => s.llm);
  const setLLM = usePipelineConfigStore((s) => s.setLLM);
  const engine = usePipelineConfigStore((s) => s.engine);
  const setEngine = usePipelineConfigStore((s) => s.setEngine);
  const targetLang = usePipelineConfigStore((s) => s.targetLang);
  const setTargetLang = usePipelineConfigStore((s) => s.setTargetLang);

  return { tc, setTc, llm, setLLM, engine, setEngine, targetLang, setTargetLang };
}

/** Stable defaults object for step-status comparison (used by episode
 *  queries). Reads the app defaults; the backend merges per-show config
 *  on top, so the working copy must not be sent here. */
export function usePipelineDefaults() {
  const app = usePipelineConfigStore((s) => s.appDefaults);
  return useMemo(() => ({
    model_size: app.transcribe.modelSize,
    diarize: app.transcribe.diarize,
    num_speakers: "",
    llm_mode: "",
    llm_provider_profile: "",
    llm_key_name: "",
    llm_model: "",
    context: "",
    target_lang: app.targetLang,
    rag_model: "",  // not part of step-status comparison
    rag_chunker: "",
  }), [app]);
}
