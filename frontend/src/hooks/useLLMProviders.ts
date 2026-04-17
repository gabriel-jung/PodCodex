/** Shared hook for LLM provider data from the backend pipeline config. */

import { useMemo, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { getPipelineConfig } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import type { LLMProviderSpec } from "@/api/types";

export function useLLMProviders() {
  const { data: pipelineConfig } = useQuery({
    queryKey: queryKeys.pipelineConfig(),
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  const apiProviders = useMemo(
    () =>
      pipelineConfig
        ? Object.entries(pipelineConfig.llm_providers).filter(
            ([k]) => k !== "ollama",
          )
        : [],
    [pipelineConfig],
  );

  const getProviderInfo = useCallback(
    (key: string): LLMProviderSpec | null =>
      (pipelineConfig?.llm_providers[key] as LLMProviderSpec) ?? null,
    [pipelineConfig],
  );

  return {
    pipelineConfig,
    apiProviders,
    getProviderInfo,
    detectedKeys: pipelineConfig?.detected_keys ?? {},
    whisperModels: pipelineConfig?.whisper_models ?? {},
  };
}
