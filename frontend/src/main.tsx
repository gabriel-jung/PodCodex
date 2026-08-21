import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import {
  MutationCache,
  QueryCache,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import {
  persistQueryClientRestore,
  persistQueryClientSave,
} from "@tanstack/react-query-persist-client";
import { createSyncStoragePersister } from "@tanstack/query-sync-storage-persister";
import "@fontsource-variable/inter";
import "@fontsource-variable/fraunces";
import "@fontsource-variable/jetbrains-mono";
import App from "./App";
import { queryKeys } from "./api/queryKeys";
import { bumpArtworkEpoch, markArtworkOffline } from "./lib/showArtwork";
import {
  PERSIST_KEY,
  PERSIST_MAX_AGE_MS,
  shouldPersistQuery,
} from "./api/persistence";
import {
  CONNECT_RETRY,
  initApiToken,
  isConnectionError,
  onApiTokenAcquired,
} from "./api/client";
import "./index.css";

// Cache-level invalidation: mutations declare what they invalidate via
// `meta: { invalidates: [...] }`, as query keys or as functions for sweeps
// that need the client. This runs here, at the MutationCache level, because
// component-level onSuccess callbacks are skipped when the owning component
// unmounts, and users routinely navigate away while long mutations (feed
// refresh, downloads, imports, editor saves) are still running.
const queryClient = new QueryClient({
  // React Query *returns* query errors instead of throwing them, so a failed
  // fetch in a component that only reads `data` renders as nothing at all —
  // no error, no empty state, a blank page and a clean console. Surfacing
  // every failure here, once, keeps that from being invisible again.
  queryCache: new QueryCache({
    onError: (error, query) => {
      console.error("Query failed:", JSON.stringify(query.queryKey), error);
    },
  }),
  mutationCache: new MutationCache({
    onSuccess: (_data, _variables, _context, mutation) => {
      for (const entry of mutation.meta?.invalidates ?? []) {
        if (typeof entry === "function") entry(queryClient);
        else void queryClient.invalidateQueries({ queryKey: entry });
      }
    },
  }),
  defaultOptions: {
    queries: {
      // Boot tolerance, applied once for every query instead of per call
      // site. The app shell renders before the sidecar is listening, so on
      // each launch the first round of queries hits a closed port. Those
      // reject with a TypeError and get a few patient retries; anything the
      // server actually answered (an ApiError) is a real failure and still
      // gives up after one, so a 404 or a 500 reaches its error state as
      // fast as it always did.
      //
      // Only a few: roughly eighteen queries mount at once on the home
      // route, and there is no point in each running its own long ladder
      // against the same closed port. `/api/health` is the one query that
      // waits the boot out, and its success refetches the rest (below).
      retry: (failureCount, error) =>
        isConnectionError(error)
          ? failureCount < CONNECT_RETRY.limit
          : failureCount < 1,
      retryDelay: (attempt, error) =>
        isConnectionError(error)
          ? CONNECT_RETRY.delay(attempt)
          : // React Query's own default, restated because supplying the
            // function form replaces it wholesale.
            Math.min(1000 * 2 ** attempt, 30_000),
      staleTime: 5 * 60_000,
      gcTime: 10 * 60_000,
    },
  },
});

// Restore a narrow slice of the cache (see api/persistence.ts) so the first
// paint of a launch carries the last known shows and settings instead of an
// empty shell. `throttleTime` batches the writes; the cache churns during a
// pipeline run and each write is a synchronous localStorage round-trip.
const persister = createSyncStoragePersister({
  storage: window.localStorage,
  key: PERSIST_KEY,
  throttleTime: 2_000,
});

const persistOptions = {
  queryClient,
  persister,
  maxAge: PERSIST_MAX_AGE_MS,
  dehydrateOptions: { shouldDehydrateQuery: shouldPersistQuery },
};

/** Save on cache events, but only the ones that can change the persisted
 *  slice.
 *
 *  `PersistQueryClientProvider` subscribes to every query *and* mutation
 *  event and re-runs `dehydrate` for each — walking the whole cache and
 *  cloning each allowed entry — before the throttle discards almost all of
 *  that work. During a pipeline run the churning namespaces (episodes,
 *  versions, synthesize, index) fire tens of events per second and none of
 *  them are persisted, so filtering first skips the dehydrate entirely for
 *  events that could not have mattered. */
function subscribeToPersistedChanges(): void {
  queryClient.getQueryCache().subscribe((event) => {
    if (shouldPersistQuery(event.query)) void persistQueryClientSave(persistOptions);
  });
}

/** Retry the panes that gave up while the backend was still booting.
 *
 *  Panes mount before the sidecar is listening and give up after a few
 *  connection retries (see CONNECT_RETRY). `/api/health` is the one query
 *  patient enough to wait a slow launch out, so its first success is the
 *  signal that the rest are worth retrying.
 *
 *  Only the failed ones: on a warm launch health resolves in tens of ms
 *  while the ~18 home-route queries are still in flight, and invalidating
 *  everything would add a second full round of requests to the fast path
 *  this whole change exists to protect. Latched, so it runs once. */
function refetchFailedQueriesOnceBackendIsUp(restoredEarlyContent: boolean): void {
  let seen = false;
  const healthKey = queryKeys.health()[0];
  queryClient.getQueryCache().subscribe((event) => {
    if (seen) return;
    if (event.query.queryKey[0] !== healthKey) return;
    if (event.query.state.status !== "success") return;
    seen = true;
    // Covers rendered from the restored cache asked for images the closed
    // port refused, and a failed <img> never retries by itself. Bumping the
    // epoch re-renders every cover with a new URL, so the browser asks
    // again. Only when a restore actually put something on screen; without
    // it nothing was requested early and the URLs should stay cacheable.
    if (restoredEarlyContent) bumpArtworkEpoch();
    void queryClient.invalidateQueries({
      predicate: (query) => query.state.status === "error",
    });
  });
}

// On a first launch the sidecar may not have written the token file yet, so
// `initApiToken` keeps polling for it. Anything already rendered embedded an
// unauthenticated URL (artwork, audio, exports), so refetch once it lands and
// let those rebuild.
onApiTokenAcquired(() => void queryClient.invalidateQueries());

// Resolve the loopback API token before anything renders: queries fire on
// mount, and URL builders (artwork, audio) need the token synchronously.
// Restore before the first render rather than through
// `PersistQueryClientProvider`, which restores in an effect and pauses every
// query until it settles — so the first paint would show the empty shell and
// only the second would carry the persisted data.
void initApiToken()
  .finally(() => persistQueryClientRestore(persistOptions))
  .finally(() => {
    // Whether the restore actually produced content the first frame can
    // paint. Covers only need re-requesting if they were rendered before
    // the backend was up.
    const restoredEarlyContent =
      queryClient.getQueryData(queryKeys.shows()) !== undefined;
    // Before the first render, so those covers never point at the closed
    // port in the first place.
    if (restoredEarlyContent) markArtworkOffline();

    subscribeToPersistedChanges();
    refetchFailedQueriesOnceBackendIsUp(restoredEarlyContent);
    createRoot(document.getElementById("root")!).render(
      <StrictMode>
        <QueryClientProvider client={queryClient}>
          <App />
        </QueryClientProvider>
      </StrictMode>,
    );
  });
