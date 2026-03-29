import { useCallback, useSyncExternalStore } from "react";

type Theme = "light" | "dark" | "system";

const STORAGE_KEY = "podcodex-theme";

function getSystemTheme(): "light" | "dark" {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function getStored(): Theme {
  return (localStorage.getItem(STORAGE_KEY) as Theme) || "system";
}

function applyTheme(theme: Theme) {
  const resolved = theme === "system" ? getSystemTheme() : theme;
  document.documentElement.classList.toggle("dark", resolved === "dark");
}

// Apply immediately on load (avoids flash)
applyTheme(getStored());

// Notify subscribers when theme changes
const listeners = new Set<() => void>();
function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

// Also react to OS theme changes when in system mode
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (getStored() === "system") {
    applyTheme("system");
    listeners.forEach((cb) => cb());
  }
});

export function useTheme() {
  const theme = useSyncExternalStore(subscribe, getStored);

  const setTheme = useCallback((t: Theme) => {
    localStorage.setItem(STORAGE_KEY, t);
    applyTheme(t);
    listeners.forEach((cb) => cb());
  }, []);

  const resolved = theme === "system" ? getSystemTheme() : theme;

  return { theme, resolved, setTheme };
}
