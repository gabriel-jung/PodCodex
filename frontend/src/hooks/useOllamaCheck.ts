import { useQuery } from "@tanstack/react-query";
import { checkOllama } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";

/** 30s staleTime so a fresh `ollama pull` shows up after Refresh
 *  without hammering the daemon on every Local-mode toggle. */
export function useOllamaCheck(enabled: boolean = true) {
  return useQuery({
    queryKey: queryKeys.ollamaCheck(),
    queryFn: checkOllama,
    enabled,
    staleTime: 30_000,
    gcTime: 5 * 60_000,
    refetchOnWindowFocus: false,
    retry: false,
  });
}
