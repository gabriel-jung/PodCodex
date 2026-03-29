import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getPolishSegments, getSegments } from "@/api/client";
import { buildDefaultContext } from "@/lib/utils";
import type { LLMConfig } from "@/components/common/LLMControls";
import type { Episode, ShowMeta, Segment } from "@/api/types";
import { usePipelineConfigStore } from "@/stores";

/**
 * Shared LLM configuration state for pipeline panels (polish, translate).
 * Reads from and writes to the persisted pipeline config store.
 * Initialises sourceLang and context from show/episode metadata on first use.
 */
export function useLLMConfig(
  episode: Episode,
  showMeta: ShowMeta | null,
): [LLMConfig, (patch: LLMConfig | ((prev: LLMConfig) => LLMConfig)) => void] {
  const llm = usePipelineConfigStore((s) => s.llm);
  const setLLM = usePipelineConfigStore((s) => s.setLLM);

  // Seed sourceLang and context from show metadata when they're empty
  useEffect(() => {
    const patches: Partial<LLMConfig> = {};
    if (!llm.sourceLang && showMeta?.language) {
      patches.sourceLang = showMeta.language;
    }
    if (!llm.context) {
      const ctx = buildDefaultContext(episode, showMeta);
      if (ctx) patches.context = ctx;
    }
    if (Object.keys(patches).length > 0) setLLM(patches);
  }, [episode, showMeta]); // eslint-disable-line react-hooks/exhaustive-deps

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
 * Load the best available source segments for an episode:
 * tries polished first, falls back to raw transcribe segments.
 *
 * Used by TranslatePanel (and any future panel) that wants the
 * highest-quality text as a reference.
 */
export function useBestSourceSegments(
  audioPath: string | null | undefined,
  opts: { enabled?: boolean; polished?: boolean },
) {
  return useQuery<Segment[]>({
    queryKey: ["best-source-segments", audioPath],
    queryFn: async () => {
      if (!audioPath) return [];
      try {
        if (opts.polished) return await getPolishSegments(audioPath);
      } catch { /* fall through */ }
      return getSegments(audioPath);
    },
    enabled: !!audioPath && (opts.enabled ?? true),
  });
}
