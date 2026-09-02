/** Last mutation failure nobody else surfaced, for the shell-level banner.
 *
 *  Fed by the MutationCache default onError in main.tsx, so a write that
 *  fails after its component unmounted (or in a component that only reads
 *  `data`) still reaches the user instead of the console alone. */

import { create } from "zustand";
import { errorMessage } from "@/lib/utils";

export interface MutationFailure {
  message: string;
  /** Wall-clock time of the failure; the banner keys its auto-dismiss on it. */
  at: number;
}

interface MutationErrorState {
  failure: MutationFailure | null;
  report: (error: unknown) => void;
  dismiss: () => void;
}

export const useMutationErrorStore = create<MutationErrorState>()((set) => ({
  failure: null,
  report: (error) => set({ failure: { message: errorMessage(error), at: Date.now() } }),
  dismiss: () => set({ failure: null }),
}));
