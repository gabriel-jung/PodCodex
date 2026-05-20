/** Shared query for the index config (embedding models + chunking strategies). */

import { useQuery } from "@tanstack/react-query";
import { getIndexConfig } from "@/api/search";
import { queryKeys } from "@/api/queryKeys";

export function useIndexConfig() {
  return useQuery({
    queryKey: queryKeys.indexConfig(),
    queryFn: getIndexConfig,
  });
}
