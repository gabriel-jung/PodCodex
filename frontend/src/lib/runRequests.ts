/** Request fields shared by the episode panels and batch runs.
 *
 *  Both paths start a run from the same working config; mapping it twice is
 *  how they drifted (diarize on CPU, provider fields outside API mode). The
 *  episode routes take these names as is; the batch route prefixes some of
 *  them (`llm_model`, `llm_batch_minutes`), which `ShowPage.runStep` does. */

import type { LLMConfig, TranscribeConfig } from "@/stores/pipelineConfigStore";

/** LLM fields of a correct or translate run. Provider and key only travel
 *  in API mode; local runs resolve neither. */
export function llmRunFields(config: LLMConfig) {
  const api = config.mode === "api";
  return {
    mode: api ? "api" : "ollama",
    provider_profile: api ? config.providerProfile || undefined : undefined,
    key_name: api ? config.keyName || undefined : undefined,
    model: config.model,
    context: config.context,
    source_lang: config.sourceLang,
    batch_minutes: config.batchMinutes,
  } as const;
}

/** Transcribe fields of a run. `clean` is the caller's: the episode panel
 *  picks it per run, the batch dialog reads the stored toggle. */
export function transcribeRunFields(
  tc: TranscribeConfig,
  { isCpu, clean }: { isCpu: boolean; clean: boolean },
) {
  return {
    model_size: tc.modelSize,
    batch_size: tc.batchSize ?? undefined,
    // CPU mode forces diarize off whatever the stored preference: pyannote
    // needs a GPU in practice, and both forms hide the toggle on CPU.
    diarize: isCpu ? false : tc.diarize,
    clean,
    hf_token: tc.hfToken || undefined,
    num_speakers: tc.numSpeakers ? Number(tc.numSpeakers) : undefined,
  };
}
