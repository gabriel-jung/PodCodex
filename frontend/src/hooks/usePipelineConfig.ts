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

/** Stable defaults object for step-status comparison (used by episode queries). */
export function usePipelineDefaults() {
  const { tc, targetLang } = usePipelineConfig();
  return useMemo(() => ({
    model_size: tc.modelSize,
    diarize: tc.diarize,
    llm_mode: "",
    llm_provider: "",
    llm_model: "",
    target_lang: targetLang,
  }), [tc.modelSize, tc.diarize, targetLang]);
}
