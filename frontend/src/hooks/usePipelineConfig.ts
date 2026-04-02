/** Convenience hook that bundles all pipeline config selectors from the store. */

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
