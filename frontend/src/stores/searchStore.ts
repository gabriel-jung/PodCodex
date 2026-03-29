/** Persisted search workspace — query, filters, history. */

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SearchState {
  /** Last search query text. */
  lastQuery: string;
  /** Recent queries (max 20). */
  history: string[];
  setLastQuery: (query: string) => void;
  addToHistory: (query: string) => void;
  clearHistory: () => void;
}

export const useSearchStore = create<SearchState>()(
  persist(
    (set, get) => ({
      lastQuery: "",
      history: [],
      setLastQuery: (query) => set({ lastQuery: query }),
      addToHistory: (query) => {
        const trimmed = query.trim();
        if (!trimmed) return;
        const history = get().history.filter((q) => q !== trimmed);
        set({ history: [trimmed, ...history].slice(0, 20) });
      },
      clearHistory: () => set({ history: [], lastQuery: "" }),
    }),
    { name: "podcodex-search" },
  ),
);
