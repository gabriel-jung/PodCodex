/** Persisted UI layout / view preferences. */

import { create } from "zustand";
import { persist } from "zustand/middleware";

type ShowGroupBy = "none" | "source";

interface LayoutState {
  /** Show view mode on the home page. */
  showViewMode: "list" | "card";
  setShowViewMode: (mode: "list" | "card") => void;
  /** Card size (columns) for the show grid. 1-5. */
  showCardSize: number;
  setShowCardSize: (size: number) => void;
  /** Group shows on the home page. */
  showGroupBy: ShowGroupBy;
  setShowGroupBy: (g: ShowGroupBy) => void;
  /** Sidebar expanded state (shared across all pages). */
  sidebarExpanded: boolean;
  setSidebarExpanded: (v: boolean) => void;
  /** Compact mode hides secondary metadata on show + episode lists. */
  compact: boolean;
  setCompact: (v: boolean) => void;
  /** Show folder last chosen as the standalone-import destination. */
  lastImportTarget: string | null;
  setLastImportTarget: (folder: string | null) => void;
}

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set) => ({
      showViewMode: "card",
      setShowViewMode: (mode) => set({ showViewMode: mode }),
      showCardSize: 3,
      setShowCardSize: (size) => set({ showCardSize: size }),
      showGroupBy: "none" as ShowGroupBy,
      setShowGroupBy: (g) => set({ showGroupBy: g }),
      sidebarExpanded: false,
      setSidebarExpanded: (v) => set({ sidebarExpanded: v }),
      compact: false,
      setCompact: (v) => set({ compact: v }),
      lastImportTarget: null,
      setLastImportTarget: (folder) => set({ lastImportTarget: folder }),
    }),
    {
      name: "podcodex-layout",
      partialize: (s) => ({
        showViewMode: s.showViewMode,
        showCardSize: s.showCardSize,
        showGroupBy: s.showGroupBy,
        sidebarExpanded: s.sidebarExpanded,
        compact: s.compact,
        lastImportTarget: s.lastImportTarget,
      }),
    },
  ),
);
