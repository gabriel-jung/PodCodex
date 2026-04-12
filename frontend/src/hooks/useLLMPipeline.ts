import { useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { getCorrectSegments, getSegments } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { buildDefaultContext } from "@/lib/utils";
import type { LLMConfig, LLMPresetKey } from "@/stores/pipelineConfigStore";
import type { Episode, ShowMeta, Segment } from "@/api/types";
import { usePipelineConfigStore } from "@/stores";
import { useCapabilities } from "@/hooks/useCapabilities";

/**
 * Shared LLM configuration state for pipeline panels (correct, translate).
 * Reads from and writes to the persisted pipeline config store.
 * Initialises sourceLang and context from show/episode metadata on first use.
 */
export function useLLMConfig(
  episode: Episode | null | undefined,
  showMeta: ShowMeta | null,
): [LLMConfig, (patch: LLMConfig | ((prev: LLMConfig) => LLMConfig)) => void] {
  const llm = usePipelineConfigStore((s) => s.llm);
  const setLLM = usePipelineConfigStore((s) => s.setLLM);

  // Sync sourceLang and context when the episode or show changes, without an
  // extra render cycle. Track previous identity and apply patches during render.
  const prevRef = useRef<{ episodeId?: string; showName?: string }>({});
  const episodeId = episode?.id;
  const showName = showMeta?.name;
  if (episodeId !== prevRef.current.episodeId || showName !== prevRef.current.showName) {
    prevRef.current = { episodeId, showName };
    if (episode) {
      const patches: Partial<LLMConfig> = {};
      if (showMeta?.language) patches.sourceLang = showMeta.language;
      const ctx = buildDefaultContext(episode, showMeta);
      if (ctx) patches.context = ctx;
      if (Object.keys(patches).length > 0) setLLM(patches);
    }
  }

  // Provide a setState-like API for compatibility with existing panels
  const setter = (valOrFn: LLMConfig | ((prev: LLMConfig) => LLMConfig)) => {
    if (typeof valOrFn === "function") {
      const next = valOrFn(llm);
      setLLM(next);
    } else {
      setLLM(valOrFn);
    }
  };

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
    provider: config.mode === "api" && config.provider !== "custom" ? config.provider : undefined,
    model: config.model,
    context: config.context,
    source_lang: config.sourceLang,
    batch_minutes: config.batchMinutes,
    api_base_url: config.apiBaseUrl || undefined,
    api_key: config.apiKey || undefined,
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
