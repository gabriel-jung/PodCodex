import { create } from "zustand";

interface LayoutState {
  hideAppSidebar: boolean;
  setHideAppSidebar: (v: boolean) => void;
}

export const useLayoutStore = create<LayoutState>((set) => ({
  hideAppSidebar: false,
  setHideAppSidebar: (v) => set({ hideAppSidebar: v }),
}));
