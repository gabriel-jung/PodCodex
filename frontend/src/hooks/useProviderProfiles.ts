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
    refetch: query.refetch,
  };
}
