import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getPolishSegments, getSegments } from "@/api/client";
import { buildDefaultContext } from "@/lib/utils";
import type { LLMConfig } from "@/components/common/LLMControls";
import type { Episode, ShowMeta, Segment } from "@/api/types";

/**
 * Shared default LLM configuration state for pipeline panels (polish, translate).
 */
export function useLLMConfig(episode: Episode, showMeta: ShowMeta | null) {
  return useState<LLMConfig>({
    mode: "ollama",
    provider: "openai",
    model: "",
    context: buildDefaultContext(episode, showMeta),
    sourceLang: showMeta?.language || "French",
    batchSize: 10,
    apiBaseUrl: "",
    apiKey: "",
  });
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
