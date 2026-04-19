import { create } from "zustand";
import { persist } from "zustand/middleware";

interface FlagPatternsState {
  /** One pattern per line. Matched as case-insensitive substring. */
  patterns: string[];
  setPatterns: (patterns: string[]) => void;
}

export const useFlagPatternsStore = create<FlagPatternsState>()(
  persist(
    (set) => ({
      patterns: [],
      setPatterns: (patterns) => set({ patterns }),
    }),
    { name: "podcodex-flag-patterns" },
  ),
);

/** Return the first matching pattern, or null. Empty/whitespace patterns ignored. */
export function matchFlagPattern(text: string, patterns: string[]): string | null {
  const haystack = text.toLowerCase();
  for (const p of patterns) {
    const needle = p.trim().toLowerCase();
    if (needle && haystack.includes(needle)) return p.trim();
  }
  return null;
}
