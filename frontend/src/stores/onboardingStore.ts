/** Tracks whether the first-launch onboarding wizard has been completed or skipped. */

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface OnboardingState {
  seen: boolean;
  setSeen: (v: boolean) => void;
}

export const useOnboardingStore = create<OnboardingState>()(
  persist(
    (set) => ({
      seen: false,
      setSeen: (v) => set({ seen: v }),
    }),
    { name: "podcodex-onboarding" },
  ),
);
