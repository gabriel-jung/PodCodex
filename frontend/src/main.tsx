import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import {
  MutationCache,
  QueryCache,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import "@fontsource-variable/inter";
import "@fontsource-variable/fraunces";
import "@fontsource-variable/jetbrains-mono";
import App from "./App";
import { initApiToken, onApiTokenAcquired } from "./api/client";
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
    queries: { retry: 1, staleTime: 5 * 60_000, gcTime: 10 * 60_000 },
  },
});

// On a first launch the sidecar may not have written the token file yet, so
// `initApiToken` keeps polling for it. Anything already rendered embedded an
// unauthenticated URL (artwork, audio, exports), so refetch once it lands and
// let those rebuild.
onApiTokenAcquired(() => void queryClient.invalidateQueries());

// Resolve the loopback API token before anything renders: queries fire on
// mount, and URL builders (artwork, audio) need the token synchronously.
void initApiToken().finally(() => {
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </StrictMode>,
  );
});
