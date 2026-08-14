import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MutationCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@fontsource-variable/inter";
import "@fontsource-variable/fraunces";
import "@fontsource-variable/jetbrains-mono";
import App from "./App";
import "./index.css";

// Cache-level invalidation: mutations declare the query keys they invalidate
// via `meta: { invalidates: [...] }`. This runs here, at the MutationCache
// level, because component-level onSuccess callbacks are skipped when the
// owning component unmounts, and users routinely navigate away while long
// mutations (feed refresh, downloads, imports) are still running.
const queryClient = new QueryClient({
  mutationCache: new MutationCache({
    onSuccess: (_data, _variables, _context, mutation) => {
      for (const queryKey of mutation.meta?.invalidates ?? []) {
        void queryClient.invalidateQueries({ queryKey });
      }
    },
  }),
  defaultOptions: {
    queries: { retry: 1, staleTime: 5 * 60_000, gcTime: 10 * 60_000 },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
