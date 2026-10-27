/** Read hook for provider profiles (built-in + custom). */

import { useQuery } from "@tanstack/react-query";
import { listProviderProfiles } from "@/api/providerProfiles";
import { queryKeys } from "@/api/queryKeys";

export function useProviderProfiles() {
  const query = useQuery({
    queryKey: queryKeys.providerProfiles(),
    queryFn: listProviderProfiles,
    staleTime: 30_000,
  });

  return {
    profiles: query.data?.profiles ?? [],
    isLoading: query.isLoading,
    // A failed read is not an empty list: callers show it instead of
    // "none yet" and an empty picker.
    error: query.isError ? query.error : null,
    refetch: query.refetch,
  };
}
