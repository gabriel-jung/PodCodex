import type { QueryClient } from "@tanstack/react-query";

const PATH_SEPARATORS = ["/", "\\"];

/**
 * Remove every cached query whose key references `path` either exactly or
 * as a parent of a slash-separated value (e.g. a show folder and the audio
 * paths inside it). Tighter than substring `includes`, which would also
 * wipe unrelated keys whenever `path` is a short or common token.
 */
export function removeQueriesUnderPath(qc: QueryClient, path: string): void {
  if (!path) return;
  qc.removeQueries({
    predicate: (q) =>
      q.queryKey.some((k) => {
        if (typeof k !== "string") return false;
        if (k === path) return true;
        return PATH_SEPARATORS.some((sep) => k.startsWith(path + sep));
      }),
  });
}
