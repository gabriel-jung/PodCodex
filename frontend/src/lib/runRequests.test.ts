import { describe, expect, it } from "vitest";
import type { LLMConfig, TranscribeConfig } from "@/stores/pipelineConfigStore";
import { llmRunFields, transcribeRunFields } from "./runRequests";

/** Both the episode panels and batch runs build their requests here, so a
 *  rule changed in one place reaches both. */

const tc: TranscribeConfig = {
  modelSize: "small",
  batchSize: null,
  diarize: true,
  clean: false,
  hfToken: "",
  numSpeakers: "2",
  language: "",
};

describe("transcribeRunFields", () => {
  it("forces diarization off on CPU whatever the stored toggle", () => {
    expect(transcribeRunFields(tc, { isCpu: true, clean: false }).diarize).toBe(false);
    expect(transcribeRunFields(tc, { isCpu: false, clean: false }).diarize).toBe(true);
  });

  it("sends an auto batch size and an empty token as absent", () => {
    const f = transcribeRunFields(tc, { isCpu: false, clean: true });
    expect(f.batch_size).toBeUndefined();
    expect(f.hf_token).toBeUndefined();
    expect(f.num_speakers).toBe(2);
    expect(f.clean).toBe(true);
  });
});

describe("llmRunFields", () => {
  const llm = {
    mode: "ollama",
    providerProfile: "openai",
    keyName: "work",
    model: "m",
    context: "",
    sourceLang: "French",
    batchMinutes: 15,
  } as LLMConfig;

  it("drops provider and key outside API mode", () => {
    const f = llmRunFields(llm);
    expect(f.mode).toBe("ollama");
    expect(f.provider_profile).toBeUndefined();
    expect(f.key_name).toBeUndefined();
  });

  it("sends provider and key in API mode", () => {
    const f = llmRunFields({ ...llm, mode: "api" });
    expect(f.provider_profile).toBe("openai");
    expect(f.key_name).toBe("work");
  });
});
