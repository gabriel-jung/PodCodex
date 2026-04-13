/** Persisted history of completed batch runs for retry / review. */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { BatchEpisode } from "./taskStore";

const MAX_HISTORY = 20;

export interface BatchHistoryEntry {
  id: string;
  timestamp: number;
  step: string;
  folder: string;
  showName?: string;
  episodes: BatchEpisode[];
  failed: { title: string; stem: string; error: string }[];
  totalCount: number;
  successCount: number;
  status: "completed" | "failed" | "cancelled";
}

interface BatchHistoryState {
  entries: BatchHistoryEntry[];
  /** Transient — not persisted. */
  isOpen: boolean;
  open: () => void;
  close: () => void;
  add: (entry: Omit<BatchHistoryEntry, "id" | "timestamp">) => void;
  remove: (id: string) => void;
  clear: () => void;
}

export const useBatchHistoryStore = create<BatchHistoryState>()(
  persist(
    (set) => ({
      entries: [],
      isOpen: false,
      open: () => set({ isOpen: true }),
      close: () => set({ isOpen: false }),
      add: (entry) =>
        set((s) => ({
          entries: [
            { ...entry, id: crypto.randomUUID(), timestamp: Date.now() },
            ...s.entries,
          ].slice(0, MAX_HISTORY),
        })),
      remove: (id) => set((s) => ({ entries: s.entries.filter((e) => e.id !== id) })),
      clear: () => set({ entries: [] }),
    }),
    {
      name: "podcodex-batch-history",
      partialize: (s) => ({ entries: s.entries }),
    },
  ),
);
