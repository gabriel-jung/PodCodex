import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getExtras } from "@/api/client";
import type { ExtrasResponse } from "@/api/types";

/**
 * Hook that returns current capabilities and extra install status.
 * Data is cached for the entire session — only refetched on window focus
 * or when explicitly invalidated (e.g. after installing an extra).
 */
export function useCapabilities() {
  const queryClient = useQueryClient();

  const { data, refetch } = useQuery({
    queryKey: ["system", "extras"],
    queryFn: getExtras,
    staleTime: Infinity,       // never auto-refetch; install flow invalidates manually
    gcTime: Infinity,          // keep in cache forever
    refetchOnWindowFocus: false,
    // Use any previously-fetched data as placeholder so we never flash "not installed"
    placeholderData: () =>
      queryClient.getQueryData<ExtrasResponse>(["system", "extras"]),
  });

  return {
    capabilities: data?.capabilities ?? {},
    extras: data?.extras ?? {},
    has: (cap: string) => data?.capabilities?.[cap] ?? false,
    refetch,
  };
}
