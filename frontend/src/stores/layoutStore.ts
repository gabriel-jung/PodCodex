/** Persisted UI layout / view preferences. */

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface LayoutState {
  hideAppSidebar: boolean;
  setHideAppSidebar: (v: boolean) => void;
  /** Show view mode on the home page. */
  showViewMode: "list" | "card";
  setShowViewMode: (mode: "list" | "card") => void;
  /** Card size (columns) for the show grid. 1-5. */
  showCardSize: number;
  setShowCardSize: (size: number) => void;
}

export const useLayoutStore = create<LayoutState>()(
  persist(
    (set) => ({
      hideAppSidebar: false,
      setHideAppSidebar: (v) => set({ hideAppSidebar: v }),
      showViewMode: "card",
      setShowViewMode: (mode) => set({ showViewMode: mode }),
      showCardSize: 3,
      setShowCardSize: (size) => set({ showCardSize: size }),
    }),
    {
      name: "podcodex-layout",
      partialize: (s) => ({ showViewMode: s.showViewMode, showCardSize: s.showCardSize }),
    },
  ),
);
