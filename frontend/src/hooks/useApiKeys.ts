/** Read hook for the API key pool. */

import { useQuery } from "@tanstack/react-query";
import { listApiKeys } from "@/api/keys";
import { queryKeys } from "@/api/queryKeys";

export function useApiKeys() {
  const query = useQuery({
    queryKey: queryKeys.apiKeys(),
    queryFn: listApiKeys,
    staleTime: 30_000,
  });

  return {
    keys: query.data?.keys ?? [],
    path: query.data?.path ?? "",
    isLoading: query.isLoading,
    refetch: query.refetch,
  };
}
