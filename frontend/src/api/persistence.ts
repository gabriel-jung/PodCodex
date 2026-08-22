/**
 * Query-cache persistence: what survives a restart, and why so little.
 *
 * The app shell paints long before the sidecar answers, so without this the
 * first frame of every launch is an empty shell with skeletons. Writing a
 * narrow slice of the cache to localStorage lets the last known shows and
 * settings render on the first paint instead; the queries then refetch and
 * reconcile once the backend is listening.
 *
 * The allowlist is deliberate, not a starting point to grow casually.
 * localStorage is synchronous, capped around 5 MB, and read on the main
 * thread during boot — the one moment being optimized. The entries below
 * total a few KB (the shows list measures ~6.5 KB across 14 shows). The
 * episode list does not qualify: ~490 KB for a single large show, and its
 * per-episode pipeline flags are exactly what goes stale while the app is
 * closed. It reloads inside the second the backend needs anyway.
 */

import type { Query } from "@tanstack/react-query";
import { queryKeys } from "./queryKeys";

/** localStorage key. Bump the version suffix when a persisted response
 *  shape changes, so an old entry is dropped rather than hydrated into
 *  components that no longer understand it.
 *
 *  Upgrades do not need a bump: `buster` in main.tsx is the app version, so
 *  every release discards the previous build's entries. The allowlist only
 *  filters what is *written*; restore hydrates whatever is in storage, which
 *  is why invalidation has to be explicit. */
export const PERSIST_KEY = "podcodex-query-cache-v1";

/** A stale entry is still worth showing for a moment, but a week-old show
 *  list is noise. Entries older than this are discarded at hydration. */
export const PERSIST_MAX_AGE_MS = 24 * 60 * 60 * 1000;

/**
 * Exact keys allowed to persist, taken from the `queryKeys` factory so a
 * rename there is a type error here rather than silent persistence loss.
 *
 * Matched exactly, not by prefix. Prefix matching is coarser than the key
 * hierarchy and leaks: `["config"]` is a prefix of `queryKeys.secrets()`
 * (`["config", "secrets"]`), which would have written masked API-key
 * prefixes and the absolute path of `secrets.env` to localStorage for a
 * day. `["system"]` likewise covers `queryKeys.ollamaCheck()`, persisting a
 * day-old "Ollama reachable" verdict that the UI would trust on boot.
 */
const PERSISTED_KEYS: ReadonlySet<string> = new Set(
  [
    queryKeys.shows(), // the home screen's content
    queryKeys.config(), // settings, so that screen is not blank
    queryKeys.capabilities(), // ffmpeg/extras, which gate UI affordances
  ].map((key) => JSON.stringify(key)),
);

/** Per-show metadata (title, cover, counts) is keyed by folder, so it is
 *  matched on its namespace rather than enumerated. `showMeta` has no
 *  sub-namespaces, so this cannot widen the way a prefix match would. */
const PERSISTED_NAMESPACE = queryKeys.showMetaAll()[0];

/**
 * Deliberately not persisted: `shellVersion`. It is a local IPC call, so
 * there is nothing to save, and persisting it survives the one event that
 * changes it. An upgrade would restore the *previous* version from disk and
 * never refetch it (`staleTime: Infinity`), while `health` reports the new
 * one, so the version-mismatch banner would fire forever and a restart could
 * not clear it.
 *
 * Deliberately not persisted: `health`. It is the one query the boot banner
 * keys on, so a persisted copy would make the app claim the backend is up
 * before it is, and render capabilities from a day-old answer. It costs one
 * cheap request.
 */
export function shouldPersistQuery(query: Query): boolean {
  // Never persist a failure: a cached error would render as a broken app on
  // the next launch even when the backend is healthy again.
  if (query.state.status !== "success") return false;
  if (query.queryKey[0] === PERSISTED_NAMESPACE) return true;
  return PERSISTED_KEYS.has(JSON.stringify(query.queryKey));
}
