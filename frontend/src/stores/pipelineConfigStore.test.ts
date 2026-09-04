import { beforeEach, describe, expect, it } from "vitest";
import { LEGACY_STORAGE_KEY, readLegacyAppDefaults } from "./pipelineConfigStore";

/**
 * The localStorage migration runs once per install and then the key is
 * removed, so a wrong result migrates once, permanently, and no real user
 * flow can re-run it. It also reimplements the value normalisations from a
 * deleted zustand `migrate` chain, which is exactly the part a future edit
 * would drop by accident.
 */

/** Read the migration result, failing the test if it produced nothing.
 *  Keeps every assertion below free of optional chaining. */
function migrated() {
  const out = readLegacyAppDefaults();
  expect(out).not.toBeNull();
  return out as NonNullable<ReturnType<typeof readLegacyAppDefaults>> & {
    transcribe: NonNullable<
      NonNullable<ReturnType<typeof readLegacyAppDefaults>>["transcribe"]
    >;
    llm: NonNullable<NonNullable<ReturnType<typeof readLegacyAppDefaults>>["llm"]>;
  };
}

function seed(state: unknown) {
  localStorage.setItem(LEGACY_STORAGE_KEY, JSON.stringify({ state }));
}

beforeEach(() => {
  localStorage.clear();
});

describe("readLegacyAppDefaults", () => {
  it("returns null when there is nothing to migrate", () => {
    expect(readLegacyAppDefaults()).toBeNull();
  });

  it("returns null for a slice that is not a config bundle", () => {
    seed({ somethingElse: 1 });
    expect(readLegacyAppDefaults()).toBeNull();
  });

  it("returns null rather than throwing on a corrupt value", () => {
    localStorage.setItem(LEGACY_STORAGE_KEY, "{not json");
    expect(readLegacyAppDefaults()).toBeNull();
  });

  it("reads the nested shape written after the app/working split", () => {
    seed({
      appDefaults: {
        transcribe: { modelSize: "large-v3" },
        llm: { mode: "ollama" },
      },
    });
    const out = migrated();
    expect(out.transcribe.model_size).toBe("large-v3");
    expect(out.llm.mode).toBe("ollama");
  });

  it("reads the flat shape written before that split", () => {
    seed({ transcribe: { modelSize: "small" }, llm: { mode: "api" } });
    const out = migrated();
    expect(out.transcribe.model_size).toBe("small");
    expect(out.llm.mode).toBe("api");
  });

  it("fills fields a pre-split slice never had from the built-ins", () => {
    seed({ transcribe: { modelSize: "small" }, llm: { mode: "api" } });
    const out = migrated();
    expect(out.index_model).toBe("bge-m3");
    expect(out.index_chunker).toBe("semantic");
    expect(out.target_lang).toBe("French");
  });

  it("never carries a stored HF token forward", () => {
    seed({ transcribe: { modelSize: "small", hfToken: "hf_secret" }, llm: {} });
    expect(JSON.stringify(readLegacyAppDefaults())).not.toContain("hf_secret");
  });

  // ── The three value rewrites from the deleted migrate chain ──

  it("v3: clears a hardcoded batchSize of 16 so auto-detect comes back", () => {
    seed({ transcribe: { modelSize: "small", batchSize: 16 }, llm: {} });
    expect(migrated().transcribe.batch_size).toBeNull();
  });

  it("v3: leaves any other explicit batch size alone", () => {
    seed({ transcribe: { modelSize: "small", batchSize: 8 }, llm: {} });
    expect(migrated().transcribe.batch_size).toBe(8);
  });

  it("v2: an existing preset counts as a choice, so the auto-upgrade cannot fire", () => {
    seed({ transcribe: {}, llm: { mode: "ollama" }, llmPreset: "local" });
    expect(migrated().llm_preset_touched).toBe(true);
  });

  it("v2: leaves the flag alone when there was no preset", () => {
    seed({ transcribe: {}, llm: { mode: "ollama" }, llmPreset: "" });
    expect(migrated().llm_preset_touched).toBe(false);
  });

  it("v5: seeds the per-mode model stash from the single model field", () => {
    seed({ transcribe: {}, llm: { mode: "ollama", model: "qwen3" } });
    expect(migrated().llm.models_by_mode?.ollama).toBe("qwen3");
  });

  it("v5: does not overwrite a stash that already has an entry", () => {
    seed({
      transcribe: {},
      llm: { mode: "ollama", model: "qwen3", modelsByMode: { api: "gpt" } },
    });
    const byMode = migrated().llm.models_by_mode ?? {};
    expect(byMode.api).toBe("gpt");
    // Not seeded, and not backfilled here either: a partial stash keeps only
    // the modes it had, and `serverToBundle` fills the rest from the
    // built-ins when the value comes back down.
    expect(byMode.ollama).toBeUndefined();
  });
});
