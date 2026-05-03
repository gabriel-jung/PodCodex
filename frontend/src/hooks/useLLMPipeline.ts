import { useCallback, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getCorrectSegments, getSegments } from "@/api/client";
import { getAllVersions } from "@/api/search";
import { queryKeys } from "@/api/queryKeys";
import { buildDefaultContext } from "@/lib/utils";
import { filterVersionsForStep, type PipelineInputStep } from "@/lib/pipelineInputs";
import type { LLMConfig, LLMPresetKey } from "@/stores/pipelineConfigStore";
import type { Episode, ShowMeta, Segment } from "@/api/types";
import { usePipelineConfigStore } from "@/stores";
import { useCapabilities } from "@/hooks/useCapabilities";
import { useApiKeys } from "@/hooks/useApiKeys";

// Priority order for the one-shot "promote to cloud" preset when an
// imported key has a hint. Built-in profile names only.
const AUTO_PROFILES = ["openai", "anthropic", "mistral"] as const;

/**
 * Shared LLM configuration state for pipeline panels (correct, translate).
 * Reads from and writes to the persisted pipeline config store.
 * Initialises sourceLang and context from show/episode metadata on first use.
 * Also promotes the preset from "manual" to "cloud" the first time the app
 * is opened with an API key configured — keeps manual as the out-of-the-box
 * default, but respects the user's credentials when present.
 */
export function useLLMConfig(
  episode: Episode | null | undefined,
  showMeta: ShowMeta | null,
): [LLMConfig, (patch: LLMConfig | ((prev: LLMConfig) => LLMConfig)) => void] {
  const llm = usePipelineConfigStore((s) => s.llm);
  const setLLM = usePipelineConfigStore((s) => s.setLLM);
  const llmPresetTouched = usePipelineConfigStore((s) => s.llmPresetTouched);
  const applyLLMPreset = usePipelineConfigStore((s) => s.applyLLMPreset);
  const { keys: pooledKeys } = useApiKeys();

  // Sync sourceLang and context when the episode or show changes
  const episodeId = episode?.id;
  const showName = showMeta?.name;
  useEffect(() => {
    if (!episode) return;
    const patches: Partial<LLMConfig> = {};
    if (showMeta?.language) patches.sourceLang = showMeta.language;
    const ctx = buildDefaultContext(episode, showMeta);
    if (ctx) patches.context = ctx;
    if (Object.keys(patches).length > 0) setLLM(patches);
  }, [episodeId, showName]); // eslint-disable-line react-hooks/exhaustive-deps

  // One-shot: when the user has never chosen a preset but their pool
  // contains a key with a known provider hint, switch to cloud + that
  // profile. Runs once because applyLLMPreset flips `llmPresetTouched`.
  useEffect(() => {
    if (llmPresetTouched || pooledKeys.length === 0) return;
    const profile = AUTO_PROFILES.find((p) =>
      pooledKeys.some((k) => k.suggested_provider === p),
    );
    if (profile) applyLLMPreset("cloud", profile);
  }, [llmPresetTouched, pooledKeys, applyLLMPreset]);

  const setter = useCallback(
    (valOrFn: LLMConfig | ((prev: LLMConfig) => LLMConfig)) => {
      if (typeof valOrFn === "function") {
        const prev = usePipelineConfigStore.getState().llm;
        setLLM(valOrFn(prev));
      } else {
        setLLM(valOrFn);
      }
    },
    [setLLM],
  );

  return [llm, setter];
}

/**
 * Build the common request fields from an LLMConfig + audioPath.
 * Panels spread this into their step-specific request, adding only
 * the extra fields they need (engine for correct, target_lang for translate).
 */
export function buildLLMRequest(audioPath: string, config: LLMConfig) {
  return {
    audio_path: audioPath,
    mode: config.mode === "api" ? "api" : "ollama",
    provider_profile: config.providerProfile || undefined,
    key_name: config.keyName || undefined,
    model: config.model,
    context: config.context,
    source_lang: config.sourceLang,
    batch_minutes: config.batchMinutes,
  } as const;
}

/**
 * Derive batch count from episode duration. The underlying store field is
 * `batchMinutes` (minutes per batch), but users think in "how many batches" —
 * this hook translates between the two. Falls back to raw minutes when
 * duration is unknown.
 */
export function useBatchCount(
  episode: Episode,
  config: LLMConfig,
  patch: (p: Partial<LLMConfig>) => void,
) {
  const episodeMinutes = episode.duration ? episode.duration / 60 : null;
  const batchCount = episodeMinutes && config.batchMinutes > 0
    ? Math.max(1, Math.round(episodeMinutes / config.batchMinutes))
    : 1;
  const setBatchCount = (count: number) => {
    if (!episodeMinutes) return;
    const n = Math.max(1, Math.min(20, Math.floor(count) || 1));
    const next = Math.max(1, Math.ceil(episodeMinutes / n));
    if (next !== config.batchMinutes) patch({ batchMinutes: next });
  };
  const minutesPerBatch = episodeMinutes ? Math.ceil(episodeMinutes / batchCount) : null;
  return { episodeMinutes, batchCount, setBatchCount, minutesPerBatch };
}

export function useLLMBackendStatus(activePreset: LLMPresetKey) {
  const { has } = useCapabilities();
  const hasOllama = has("ollama");
  const hasOpenAI = has("openai");
  const hasLLM = hasOllama || hasOpenAI;
  const backendMissing =
    (activePreset === "local" && !hasOllama) ||
    (activePreset === "cloud" && !hasOpenAI);
  const disabledTitle = backendMissing
    ? activePreset === "local"
      ? "Install Ollama to run locally — or switch to Cloud/Manual"
      : "Install the openai package to use cloud providers — or switch to Manual"
    : undefined;
  return { hasOllama, hasOpenAI, hasLLM, backendMissing, disabledTitle };
}

/**
 * Fetch all versions for an episode, filtered to those valid as input for
 * a given pipeline step. Used by CorrectPanel, TranslatePanel, IndexPanel.
 */
export function useInputVersions(
  audioPath: string | null | undefined,
  step: PipelineInputStep,
  enabled: boolean,
  outputDir?: string | null,
) {
  const { data: allVersions } = useQuery({
    queryKey: queryKeys.allVersions(audioPath ?? outputDir),
    queryFn: () => getAllVersions(audioPath, outputDir),
    enabled: (!!audioPath || !!outputDir) && enabled,
  });
  return useMemo(
    () => (allVersions ? filterVersionsForStep(allVersions, step) : undefined),
    [allVersions, step],
  );
}

/**
 * Load the best available source segments for an episode: tries corrected
 * first, falls back to raw transcribe segments. Used as the reference pane
 * in TranslatePanel.
 */
export function useBestSourceSegments(
  audioPath: string | null | undefined,
  opts: { enabled?: boolean; corrected?: boolean },
) {
  return useQuery<Segment[]>({
    queryKey: queryKeys.bestSourceSegments(audioPath),
    queryFn: async () => {
      if (!audioPath) return [];
      try {
        if (opts.corrected) return await getCorrectSegments(audioPath);
      } catch { /* fall through */ }
      return getSegments(audioPath);
    },
    enabled: !!audioPath && (opts.enabled ?? true),
  });
}
