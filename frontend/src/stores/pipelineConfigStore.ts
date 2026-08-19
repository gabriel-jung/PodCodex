/** Pipeline configuration store.
 *
 * Two layers:
 *  - `appDefaults`: the app-wide defaults, owned by the server
 *    (config.json `pipeline_defaults`). Hydrated once at startup via
 *    `useHydrateAppDefaults`; edits on Settings → Pipeline write through
 *    with a debounced PUT. Nothing here persists in the browser.
 *  - the flat working config (`transcribe`/`llm`/`engine`/`targetLang`/
 *    `indexModel` + preset keys): what the episode panels and the batch
 *    modal read and edit. It is re-seeded from the per-show config merged
 *    over `appDefaults` whenever a show opens (`seedWorkingFromShow`), so a
 *    panel edit is a per-run tweak that never reaches the app defaults or
 *    another show.
 */

import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { create } from "zustand";
import type { PipelineAppDefaults, PipelineDefaults } from "@/api/types";
import { queryKeys } from "@/api/queryKeys";
import { getConfig, putPipelineDefaults } from "@/api/shows";
import { BOOT_RETRY } from "@/api/health";

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
  const showModels = Object.fromEntries(
    Object.entries(p.llm_models_by_mode ?? {}).filter(([, v]) => !!v),
  );
  const model = showModels[mode] || app.llm.modelsByMode[mode] || "";
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
      // Show's per-mode entries overlay app's so a panel mode switch picks
      // up the show-saved value instead of the (possibly empty) app default.
      modelsByMode: { ...app.llm.modelsByMode, ...showModels, [mode]: model },
      context: p.context || app.llm.context,
      batchMinutes:
        p.llm_batch_minutes != null && p.llm_batch_minutes > 0
          ? p.llm_batch_minutes
          : app.llm.batchMinutes,
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

// ── Server ⇄ store conversion ────────────────────────────

/** Store bundle → server model (snake_case; the HF token never leaves the
 *  secrets file, so it is not part of the server shape). */
export function bundleToServer(b: ConfigBundle): PipelineAppDefaults {
  return {
    transcribe: {
      model_size: b.transcribe.modelSize,
      batch_size: b.transcribe.batchSize,
      diarize: b.transcribe.diarize,
      clean: b.transcribe.clean,
      num_speakers: b.transcribe.numSpeakers,
      language: b.transcribe.language,
    },
    llm: {
      mode: b.llm.mode,
      provider_profile: b.llm.providerProfile,
      key_name: b.llm.keyName,
      model: b.llm.model,
      models_by_mode: b.llm.modelsByMode,
      context: b.llm.context,
      source_lang: b.llm.sourceLang,
      batch_minutes: b.llm.batchMinutes,
    },
    engine: b.engine,
    target_lang: b.targetLang,
    index_model: b.indexModel,
    index_chunker: b.indexChunker,
    transcribe_preset: b.transcribePreset,
    llm_preset: b.llmPreset,
    llm_preset_touched: b.llmPresetTouched,
    index_preset: b.indexPreset,
  };
}

/** Server model → store bundle. Pydantic serializes complete sub-models,
 *  so only the optional pieces (`transcribe`/`llm` on a partial payload and
 *  the two nullable fields) need fallbacks. */
export function serverToBundle(d: PipelineAppDefaults): ConfigBundle {
  const t = d.transcribe ?? INITIAL_SERVER.transcribe!;
  const llm = d.llm ?? INITIAL_SERVER.llm!;
  return {
    transcribe: {
      modelSize: t.model_size,
      batchSize: t.batch_size ?? null,
      diarize: t.diarize,
      clean: t.clean,
      hfToken: "",
      numSpeakers: t.num_speakers,
      language: t.language,
    },
    llm: {
      mode: llm.mode as LLMMode,
      providerProfile: llm.provider_profile,
      keyName: llm.key_name,
      model: llm.model,
      modelsByMode: {
        ...INITIAL_BUNDLE.llm.modelsByMode,
        ...llm.models_by_mode,
      },
      context: llm.context,
      sourceLang: llm.source_lang,
      batchMinutes: llm.batch_minutes,
    },
    engine: d.engine,
    targetLang: d.target_lang,
    indexModel: d.index_model,
    indexChunker: d.index_chunker,
    transcribePreset: d.transcribe_preset,
    llmPreset: d.llm_preset,
    llmPresetTouched: d.llm_preset_touched,
    indexPreset: d.index_preset,
  };
}

const INITIAL_SERVER = bundleToServer(INITIAL_BUNDLE);

// Write-through on the *leading* edge: the first edit saves immediately, and
// further edits inside the window coalesce into one trailing save. There is
// deliberately no flush-on-close to fall back on — registering a Tauri close
// handler makes the app unclosable (see platform/tauri.ts) — so a single
// toggle, the common case, has to be durable the moment it is made. Only the
// tail of a rapid burst, e.g. still typing into a number field, is at risk.
const PUSH_DEBOUNCE_MS = 600;
let pushTimer: ReturnType<typeof setTimeout> | null = null;
let lastPushAt = 0;

function pushNow(): Promise<void> {
  pushTimer = null;
  lastPushAt = Date.now();
  return putPipelineDefaults(
    bundleToServer(usePipelineConfigStore.getState().appDefaults),
  ).then(
    () => undefined,
    (err: unknown) => {
      console.warn("Saving pipeline defaults failed:", err);
    },
  );
}

function schedulePush(): void {
  // Never write defaults we did not successfully read. A failed hydration
  // leaves `appDefaults` at the built-ins, so pushing would replace the
  // user's stored defaults with factory values plus whatever they just
  // touched — data loss the old localStorage persistence could not cause.
  if (!hydrationSucceeded) {
    console.warn("Pipeline defaults not loaded; skipping save.");
    return;
  }
  if (pushTimer) clearTimeout(pushTimer);
  const since = Date.now() - lastPushAt;
  if (since >= PUSH_DEBOUNCE_MS) {
    void pushNow();
    return;
  }
  pushTimer = setTimeout(() => void pushNow(), PUSH_DEBOUNCE_MS - since);
}

// ── Store ────────────────────────────────────────────────

export interface PipelineConfigState extends ConfigBundle {
  /** App-wide defaults (server-owned); edited only on Settings → Pipeline. */
  appDefaults: ConfigBundle;
  /** True once `useHydrateAppDefaults` resolved the server value (or its
   *  absence). Seeding the working config waits for this so a show open
   *  never seeds from the built-ins while the real defaults are in flight. */
  appDefaultsReady: boolean;
  /** True when the defaults could not be read. Writes stay blocked (we will
   *  not overwrite settings we never saw), so the UI must say so rather than
   *  accept edits it is going to drop. */
  appDefaultsFailed: boolean;
  hydrateAppDefaults: (defaults: PipelineAppDefaults | null, ok?: boolean) => void;

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

export const usePipelineConfigStore = create<PipelineConfigState>()((set) => {
  // Shared write-through shape of every app-default setter.
  const setAppDefaults = (reduce: (b: ConfigBundle) => ConfigBundle) => {
    set((s) => ({ appDefaults: reduce(s.appDefaults) }));
    schedulePush();
  };
  return {
  ...INITIAL_BUNDLE,
  appDefaults: INITIAL_BUNDLE,
  appDefaultsReady: false,
  appDefaultsFailed: false,
  hydrateAppDefaults: (defaults, ok = true) =>
    set({
      appDefaults: defaults ? serverToBundle(defaults) : INITIAL_BUNDLE,
      appDefaultsReady: true,
      appDefaultsFailed: !ok,
    }),

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

  // ── App-default setters (write through to the server, debounced) ──
  setAppTranscribe: (patch) => setAppDefaults((b) => reduceTranscribe(b, patch)),
  setAppLLM: (patch) => setAppDefaults((b) => reduceLLM(b, patch)),
  setAppTargetLang: (targetLang) => setAppDefaults((b) => ({ ...b, targetLang })),
  setAppIndexModel: (model) => setAppDefaults((b) => reduceIndexModel(b, model)),
  setAppIndexChunker: (indexChunker) =>
    setAppDefaults((b) => ({ ...b, indexChunker })),

  seedWorkingFromShow: (pipeline) =>
    set((s) => effectiveBundle(s.appDefaults, pipeline)),
  };
});

/** localStorage key of the pre-server-config zustand persist slice. Read
 *  once by the hydration hook to migrate old installs, then removed. */
const LEGACY_STORAGE_KEY = "podcodex-pipeline-config";

// Module-level guards: hydration runs once per app start. StrictMode mounts
// the effect twice; a component-level cancel flag would let the surviving
// mount skip hydration, so guard here and let the single async run land its
// result in the store (safe after unmount — it's not component state).
let hydrationStarted = false;
// Only a successful read unlocks writes; see `schedulePush`.
let hydrationSucceeded = false;

/** Hydrate `appDefaults` from server config once at app startup, and keep
 *  a pending debounced save from being lost on window close.
 *
 *  Server value wins. When the server has none (fresh install or first run
 *  after the localStorage era), a legacy persisted slice is promoted to the
 *  server once and the key removed; otherwise the built-ins stand. */
export function useHydrateAppDefaults(): void {
  const queryClient = useQueryClient();
  useEffect(() => {
    if (hydrationStarted) return;
    hydrationStarted = true;
    void (async () => {
      let defaults: PipelineAppDefaults | null = null;
      try {
        // fetchQuery primes the shared ["config"] cache (HomePage, panels)
        // instead of firing a duplicate request beside it. BOOT_RETRY because
        // this races the sidecar's 10-30s first-launch extraction, exactly
        // what /api/health's schedule exists for; the default `retry: 1`
        // gives up after ~1s and would leave every session cold-started.
        const cfg = await queryClient.fetchQuery({
          queryKey: queryKeys.config(),
          queryFn: getConfig,
          ...BOOT_RETRY,
        });
        defaults = cfg.pipeline_defaults ?? null;
        if (!defaults) {
          const legacy = readLegacyAppDefaults();
          if (legacy) defaults = await putPipelineDefaults(legacy);
        }
        localStorage.removeItem(LEGACY_STORAGE_KEY);
        // Only now unlock writes. Unlocking before the migration PUT would
        // let a Settings edit store the built-ins first, and the next start
        // would take that as the server's answer and never retry the
        // migration — the legacy values would be shadowed for good.
        hydrationSucceeded = true;
      } catch (err) {
        // Unreachable server, or the migration PUT failed: stay on the
        // built-ins for this session with writes still blocked, and let the
        // next app start retry (the legacy key is only removed on success).
        console.warn("Hydrating pipeline defaults failed:", err);
      }
      usePipelineConfigStore
        .getState()
        .hydrateAppDefaults(defaults, hydrationSucceeded);
    })();
  }, [queryClient]);
}

function readLegacyAppDefaults(): PipelineAppDefaults | null {
  const raw = localStorage.getItem(LEGACY_STORAGE_KEY);
  if (!raw) return null;
  try {
    const state = (JSON.parse(raw) as { state?: Record<string, unknown> }).state;
    if (!state) return null;
    // Two shapes to accept. Slices written before the app/working split kept
    // the bundle's fields at the top level; only later ones nest them under
    // `appDefaults`. The zustand migration chain used to promote the old
    // shape, and it went away with the persist layer, so handle it here or
    // those installs silently start over on the built-in defaults.
    const bundle = (state.appDefaults ?? state) as Partial<ConfigBundle>;
    if (!bundle.transcribe && !bundle.llm) return null;
    // Old slices can also predate later bundle fields; fill from the built-ins.
    const merged: ConfigBundle = {
      ...INITIAL_BUNDLE,
      ...bundle,
      transcribe: {
        ...INITIAL_BUNDLE.transcribe,
        ...bundle.transcribe,
        hfToken: "",
      },
      llm: { ...INITIAL_BUNDLE.llm, ...bundle.llm },
    };

    // The deleted zustand `migrate` chain also normalized *values*, not just
    // the shape. Those steps have to happen here or an old slice migrates
    // once, permanently, with the wrong ones.
    //   v3: a hardcoded batchSize of 16 overrode the backend's VRAM-based
    //       auto-detect; null re-enables it.
    if (merged.transcribe.batchSize === 16) merged.transcribe.batchSize = null;
    //   v2: an existing preset means the user already chose, so don't let the
    //       "switch to cloud when an API key shows up" auto-upgrade fire.
    if (!merged.llmPresetTouched && merged.llmPreset) {
      merged.llmPresetTouched = true;
    }
    //   v5: per-mode model stash, seeded from the single `model` field.
    if (!Object.values(merged.llm.modelsByMode).some(Boolean) && merged.llm.model) {
      merged.llm.modelsByMode = {
        ...merged.llm.modelsByMode,
        [merged.llm.mode]: merged.llm.model,
      };
    }
    return bundleToServer(merged);
  } catch {
    return null;
  }
}

/** Seed the working pipeline config from a show once per show open.
 *  Re-seeds when the folder changes; a same-show meta refetch does not
 *  re-seed, so a panel edit survives background refetches. */
export function useSeedPipelineFromShow(
  folder: string | undefined,
  pipeline: PipelineDefaults | null | undefined,
  ready: boolean,
): void {
  const seed = usePipelineConfigStore((s) => s.seedWorkingFromShow);
  // Also wait for the server-owned app defaults: seeding from the built-ins
  // while hydration is in flight would bake the wrong base into the working
  // copy for the whole show visit.
  const defaultsReady = usePipelineConfigStore((s) => s.appDefaultsReady);
  const seededFolder = useRef<string | null>(null);
  useEffect(() => {
    if (!ready || !defaultsReady) return;
    if (seededFolder.current === (folder ?? null)) return;
    seededFolder.current = folder ?? null;
    seed(pipeline);
  }, [folder, ready, defaultsReady, pipeline, seed]);
}
