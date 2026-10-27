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
    // A failed read is not an empty list: callers show it instead of
    // "none yet" and an empty picker.
    error: query.isError ? query.error : null,
    refetch: query.refetch,
  };
}
