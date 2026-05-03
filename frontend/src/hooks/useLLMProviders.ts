/** Pipeline-config bits that aren't part of the new key/profile pool.
 *
 * `apiProviders` and per-provider key detection moved to `useApiKeys` +
 * `useProviderProfiles`. What's left here:
 *  - whisper model catalog (transcribe step)
 *  - `detectedKeys.hf_token` for the diarize gate
 */

import { useQuery } from "@tanstack/react-query";
import { getPipelineConfig } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";

export function useLLMProviders() {
  const { data: pipelineConfig } = useQuery({
    queryKey: queryKeys.pipelineConfig(),
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  return {
    pipelineConfig,
    detectedKeys: pipelineConfig?.detected_keys ?? {},
    whisperModels: pipelineConfig?.whisper_models ?? {},
  };
}
