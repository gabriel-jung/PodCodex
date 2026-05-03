/** Persisted search workspace — last query. */

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SearchState {
  /** Last search query text. */
  lastQuery: string;
  setLastQuery: (query: string) => void;
}

export const useSearchStore = create<SearchState>()(
  persist(
    (set) => ({
      lastQuery: "",
      setLastQuery: (query) => set({ lastQuery: query }),
    }),
    { name: "podcodex-search" },
  ),
);
