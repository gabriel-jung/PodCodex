/** Pipeline configuration store.
 *
 * Two layers:
 *  - `appDefaults`: the persisted, app-wide defaults. Edited only on
 *    Settings → Pipeline.
 *  - the flat working config (`transcribe`/`llm`/`engine`/`targetLang`/
 *    `indexModel` + preset keys): what the episode panels and the batch
 *    modal read and edit. It is re-seeded from the per-show config merged
 *    over `appDefaults` whenever a show opens (`seedWorkingFromShow`), so a
 *    panel edit is a per-run tweak that never reaches the app defaults or
 *    another show.
 */

import { useEffect, useRef } from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { PipelineDefaults } from "@/api/types";

export type LLMMode = "api" | "ollama" | "manual";

export interface LLMConfig {
  mode: LLMMode;
  /** Name of a profile from /api/provider-profiles (built-in or custom). */
  providerProfile: string;
  /** Name of an entry in the API key pool (/api/keys). Empty for ollama. */
  keyName: string;
  /** Live model for the current mode. Mirrors `modelsByMode[mode]`. */
  model: string;
  /** Per-mode model stash so switching cloud→local→cloud restores prior
   *  picks instead of clobbering them. `model` is always the current view. */
  modelsByMode: Record<LLMMode, string>;
  context: string;
  sourceLang: string;
  batchMinutes: number;
}

export interface TranscribeConfig {
  modelSize: string;
  /** Whisper batch size. ``null`` means "auto" — the backend's
   *  ``default_batch_size()`` returns 8 for ≤10 GB VRAM, 16 above.
   *  Sending a concrete number overrides that auto-detection. */
  batchSize: number | null;
  diarize: boolean;
  clean: boolean;
  hfToken: string;
  numSpeakers: string;
  language: string;
}

/** One full set of pipeline values, used both for `appDefaults` and as the
 *  shape of the flat working config. */
export interface ConfigBundle {
  transcribe: TranscribeConfig;
  llm: LLMConfig;
  /** Correct-specific transcript-source override ("" = auto-detect). */
  engine: string;
  targetLang: string;
  indexModel: string;
  indexChunker: string;
  transcribePreset: string;
  llmPreset: string;
  /** True once the user explicitly picks an LLM preset, stopping the
   *  auto-default (switch to "cloud" when an API key is detected). */
  llmPresetTouched: boolean;
  indexPreset: string;
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

export type LLMPresetKey = keyof typeof LLM_PRESETS;

/** Inverse of `LLM_PRESETS[key].mode` — `mode` uniquely determines a preset. */
export function modeToPreset(mode: LLMConfig["mode"]): LLMPresetKey {
  if (mode === "api") return "cloud";
  if (mode === "ollama") return "local";
  return "manual";
}

export const INDEX_PRESETS = {
  fast: { label: "Fast", desc: "Small model, ideal for light CPU", model: "e5-small" },
  balanced: { label: "Balanced", desc: "Good quality, works on CPU", model: "bge-m3" },
  gpu: { label: "GPU", desc: "Context-aware, very slow on CPU", model: "pplx-0.6B" },
} as const;

export type IndexPresetKey = keyof typeof INDEX_PRESETS;

/** Reverse lookup: model key → preset key (or "" if no preset matches). */
export function modelToIndexPreset(model: string): IndexPresetKey | "" {
  for (const [key, spec] of Object.entries(INDEX_PRESETS)) {
    if (spec.model === model) return key as IndexPresetKey;
  }
  return "";
}

// ── Bundle reducers (pure; shared by working + appDefaults setters) ──

const INITIAL_BUNDLE: ConfigBundle = {
  transcribe: {
    modelSize: "large-v3-turbo",
    batchSize: null,
    diarize: false,
    clean: false,
    hfToken: "",
    numSpeakers: "",
    language: "",
  },
  llm: {
    mode: "manual",
    providerProfile: "",
    keyName: "",
    model: "",
    modelsByMode: { api: "", ollama: "", manual: "" },
    context: "",
    sourceLang: "English",
    batchMinutes: 15,
  },
  engine: "",
  targetLang: "French",
  indexModel: "bge-m3",
  indexChunker: "semantic",
  transcribePreset: "gpu",
  llmPreset: "manual",
  llmPresetTouched: false,
  indexPreset: "balanced",
};

function reduceTranscribe(b: ConfigBundle, patch: Partial<TranscribeConfig>): ConfigBundle {
  return {
    ...b,
    transcribe: { ...b.transcribe, ...patch },
    // Only reset the preset when the model changes (what presets control).
    transcribePreset: "modelSize" in patch ? "" : b.transcribePreset,
  };
}

function reduceLLM(b: ConfigBundle, patch: Partial<LLMConfig>): ConfigBundle {
  const next = { ...b.llm, ...patch };
  if ("mode" in patch && patch.mode && patch.mode !== b.llm.mode) {
    next.modelsByMode = { ...b.llm.modelsByMode, [b.llm.mode]: b.llm.model };
    next.model = next.modelsByMode[patch.mode] ?? "";
  } else if ("model" in patch) {
    next.modelsByMode = { ...b.llm.modelsByMode, [next.mode]: patch.model ?? "" };
  }
  // Only invalidate the preset when `mode` (the field a preset controls)
  // changes; tweaking sourceLang/context/etc. keeps the preset selection.
  return { ...b, llm: next, llmPreset: "mode" in patch ? "" : b.llmPreset };
}

function reduceTranscribePreset(b: ConfigBundle, key: keyof typeof TRANSCRIBE_PRESETS): ConfigBundle {
  const p = TRANSCRIBE_PRESETS[key];
  return p
    ? { ...b, transcribe: { ...b.transcribe, modelSize: p.modelSize }, transcribePreset: key }
    : b;
}

function reduceLLMPreset(b: ConfigBundle, key: LLMPresetKey, providerProfile?: string): ConfigBundle {
  const p = LLM_PRESETS[key];
  if (!p) return b;
  const modeChanged = p.mode !== b.llm.mode;
  const modelsByMode = modeChanged
    ? { ...b.llm.modelsByMode, [b.llm.mode]: b.llm.model }
    : b.llm.modelsByMode;
  const model = modeChanged ? modelsByMode[p.mode] ?? "" : b.llm.model;
  return {
    ...b,
    llm: {
      ...b.llm,
      mode: p.mode,
      model,
      modelsByMode,
      ...(providerProfile ? { providerProfile } : {}),
    },
    llmPreset: key,
    llmPresetTouched: true,
  };
}

function reduceIndexModel(b: ConfigBundle, indexModel: string): ConfigBundle {
  return { ...b, indexModel, indexPreset: modelToIndexPreset(indexModel) };
}

function reduceIndexPreset(b: ConfigBundle, key: keyof typeof INDEX_PRESETS): ConfigBundle {
  const p = INDEX_PRESETS[key];
  return p ? { ...b, indexModel: p.model, indexPreset: key } : b;
}

/** Merge a show's per-show pipeline config over the app defaults. Empty
 *  strings / null on the show side mean "inherit the app default". */
export function effectiveBundle(
  app: ConfigBundle,
  p: PipelineDefaults | null | undefined,
): ConfigBundle {
  if (!p) return app;
  const mode = (p.llm_mode || app.llm.mode) as LLMMode;
  const model = p.llm_model || app.llm.modelsByMode[mode] || app.llm.model;
  const indexModel = p.rag_model || app.indexModel;
  return {
    ...app,
    transcribe: {
      ...app.transcribe,
      modelSize: p.model_size || app.transcribe.modelSize,
      diarize: p.diarize ?? app.transcribe.diarize,
      numSpeakers: p.num_speakers || app.transcribe.numSpeakers,
    },
    llm: {
      ...app.llm,
      mode,
      providerProfile: p.llm_provider_profile || app.llm.providerProfile,
      keyName: p.llm_key_name || app.llm.keyName,
      model,
      modelsByMode: { ...app.llm.modelsByMode, [mode]: model },
      context: p.context || app.llm.context,
    },
    targetLang: p.target_lang || app.targetLang,
    indexModel,
    indexChunker: p.rag_chunker || app.indexChunker,
    transcribePreset: p.model_size ? "" : app.transcribePreset,
    llmPreset: modeToPreset(mode),
    llmPresetTouched: true,
    indexPreset: modelToIndexPreset(indexModel),
  };
}

// ── Store ────────────────────────────────────────────────

export interface PipelineConfigState extends ConfigBundle {
  /** Persisted app-wide defaults; edited only on Settings → Pipeline. */
  appDefaults: ConfigBundle;

  // Working-config setters (episode panels + batch modal).
  setTranscribe: (patch: Partial<TranscribeConfig>) => void;
  setLLM: (patch: Partial<LLMConfig>) => void;
  setEngine: (engine: string) => void;
  setTargetLang: (lang: string) => void;
  setIndexModel: (model: string) => void;
  setIndexChunker: (chunker: string) => void;
  applyTranscribePreset: (key: keyof typeof TRANSCRIBE_PRESETS) => void;
  applyLLMPreset: (key: LLMPresetKey, providerProfile?: string) => void;
  applyIndexPreset: (key: keyof typeof INDEX_PRESETS) => void;

  // App-default setters (Settings → Pipeline).
  setAppTranscribe: (patch: Partial<TranscribeConfig>) => void;
  setAppLLM: (patch: Partial<LLMConfig>) => void;
  setAppTargetLang: (lang: string) => void;
  setAppIndexModel: (model: string) => void;
  setAppIndexChunker: (chunker: string) => void;

  /** Re-seed the working config from a show's per-show config merged over
   *  the app defaults. Called when a show opens. */
  seedWorkingFromShow: (pipeline: PipelineDefaults | null | undefined) => void;
}

/** Extract the working ConfigBundle from the flat store state. */
function working(s: PipelineConfigState): ConfigBundle {
  return {
    transcribe: s.transcribe,
    llm: s.llm,
    engine: s.engine,
    targetLang: s.targetLang,
    indexModel: s.indexModel,
    indexChunker: s.indexChunker,
    transcribePreset: s.transcribePreset,
    llmPreset: s.llmPreset,
    llmPresetTouched: s.llmPresetTouched,
    indexPreset: s.indexPreset,
  };
}

export const usePipelineConfigStore = create<PipelineConfigState>()(
  persist(
    (set) => ({
      ...INITIAL_BUNDLE,
      appDefaults: INITIAL_BUNDLE,

      // ── Working setters ──
      setTranscribe: (patch) => set((s) => reduceTranscribe(working(s), patch)),
      setLLM: (patch) =>
        set((s) => {
          // No-op short-circuit: zustand subscribers re-render on any new
          // `llm` reference even when every patched field equals current.
          const changed = (Object.keys(patch) as (keyof LLMConfig)[]).some(
            (k) => patch[k] !== s.llm[k],
          );
          return changed ? reduceLLM(working(s), patch) : s;
        }),
      setEngine: (engine) => set({ engine }),
      setTargetLang: (targetLang) => set({ targetLang }),
      setIndexModel: (model) => set((s) => reduceIndexModel(working(s), model)),
      setIndexChunker: (indexChunker) => set({ indexChunker }),
      applyTranscribePreset: (key) =>
        set((s) => reduceTranscribePreset(working(s), key)),
      applyLLMPreset: (key, providerProfile) =>
        set((s) => reduceLLMPreset(working(s), key, providerProfile)),
      applyIndexPreset: (key) => set((s) => reduceIndexPreset(working(s), key)),

      // ── App-default setters ──
      setAppTranscribe: (patch) =>
        set((s) => ({ appDefaults: reduceTranscribe(s.appDefaults, patch) })),
      setAppLLM: (patch) =>
        set((s) => ({ appDefaults: reduceLLM(s.appDefaults, patch) })),
      setAppTargetLang: (targetLang) =>
        set((s) => ({ appDefaults: { ...s.appDefaults, targetLang } })),
      setAppIndexModel: (model) =>
        set((s) => ({ appDefaults: reduceIndexModel(s.appDefaults, model) })),
      setAppIndexChunker: (indexChunker) =>
        set((s) => ({ appDefaults: { ...s.appDefaults, indexChunker } })),

      seedWorkingFromShow: (pipeline) =>
        set((s) => effectiveBundle(s.appDefaults, pipeline)),
    }),
    {
      name: "podcodex-pipeline-config",
      version: 7,
      migrate(persisted: unknown, fromVersion: number) {
        const s = persisted as Record<string, unknown>;
        if (fromVersion < 1) {
          const tc = s.transcribe as Record<string, unknown> | undefined;
          if (tc && tc.clean === undefined) tc.clean = false;
          if (!s.transcribePreset) s.transcribePreset = "";
          if (!s.llmPreset) s.llmPreset = "";
          if (!s.indexPreset) s.indexPreset = "";
          if (!s.indexModel) s.indexModel = "bge-m3";
        }
        if (fromVersion < 2) {
          // A persisted v1 record means the app was opened under the
          // previous UI, so treat the user as having touched the setting:
          // silently auto-upgrading their LLM config on next load would be
          // surprising. Fresh installs never hit this branch; their initial
          // `llmPresetTouched: false` opts them into the auto-upgrade.
          const preset = (s.llmPreset as string | undefined) || "";
          s.llmPresetTouched = preset !== "";
        }
        if (fromVersion < 3) {
          // The old default of 16 was hardcoded and bypassed the backend's
          // VRAM-based auto-detect; flip it to null so auto-detect runs.
          const tc = s.transcribe as Record<string, unknown> | undefined;
          if (tc && tc.batchSize === 16) tc.batchSize = null;
        }
        if (fromVersion < 4) {
          // LLM credentials moved from provider/apiKey/apiBaseUrl on the LLM
          // config to a named key pool + profile catalog. Old picks are no
          // longer addressable; reset so the user re-picks from the new UI.
          const llm = s.llm as Record<string, unknown> | undefined;
          if (llm) {
            delete llm.provider;
            delete llm.apiKey;
            delete llm.apiBaseUrl;
            llm.providerProfile = "";
            llm.keyName = "";
          }
        }
        if (fromVersion < 5) {
          // Per-mode model stash. Seed it from the existing `model` so the
          // current pick lives under the current mode, losing no value.
          const llm = s.llm as Record<string, unknown> | undefined;
          if (llm) {
            const mode = (llm.mode as LLMMode | undefined) ?? "manual";
            const model = (llm.model as string | undefined) ?? "";
            llm.modelsByMode = { api: "", ollama: "", manual: "", [mode]: model };
          }
        }
        if (fromVersion < 6) {
          // The flat config used to be both the app default and the
          // working copy. Promote the persisted flat fields into the new
          // `appDefaults` bundle; the working copy is now seeded per show.
          s.appDefaults = {
            transcribe: s.transcribe ?? INITIAL_BUNDLE.transcribe,
            llm: s.llm ?? INITIAL_BUNDLE.llm,
            engine: s.engine ?? INITIAL_BUNDLE.engine,
            targetLang: s.targetLang ?? INITIAL_BUNDLE.targetLang,
            indexModel: s.indexModel ?? INITIAL_BUNDLE.indexModel,
            transcribePreset: s.transcribePreset ?? INITIAL_BUNDLE.transcribePreset,
            llmPreset: s.llmPreset ?? INITIAL_BUNDLE.llmPreset,
            llmPresetTouched: s.llmPresetTouched ?? INITIAL_BUNDLE.llmPresetTouched,
            indexPreset: s.indexPreset ?? INITIAL_BUNDLE.indexPreset,
          };
        }
        if (fromVersion < 7) {
          // Search-index chunker became an app default to mirror indexModel.
          // Seed it with the previous hardcoded fallback so existing installs
          // keep the same behavior.
          const app = s.appDefaults as Record<string, unknown> | undefined;
          if (app && app.indexChunker === undefined) app.indexChunker = "semantic";
        }
        return s as unknown as PipelineConfigState;
      },
      // Persist only the app defaults (the working copy is reseeded per
      // show). Don't persist the Hugging Face token.
      partialize: (s) => ({
        appDefaults: {
          ...s.appDefaults,
          transcribe: { ...s.appDefaults.transcribe, hfToken: "" },
        },
      }),
    },
  ),
);

/** Seed the working pipeline config from a show once per show open.
 *  Re-seeds when the folder changes; a same-show meta refetch does not
 *  re-seed, so a panel edit survives background refetches. */
export function useSeedPipelineFromShow(
  folder: string | undefined,
  pipeline: PipelineDefaults | null | undefined,
  ready: boolean,
): void {
  const seed = usePipelineConfigStore((s) => s.seedWorkingFromShow);
  const seededFolder = useRef<string | null>(null);
  useEffect(() => {
    if (!ready) return;
    if (seededFolder.current === (folder ?? null)) return;
    seededFolder.current = folder ?? null;
    seed(pipeline);
  }, [folder, ready, pipeline, seed]);
}
