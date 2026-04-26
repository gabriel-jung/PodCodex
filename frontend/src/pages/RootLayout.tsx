import { Suspense, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Outlet } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { getHealth } from "@/api/client";
import PanelLoading from "@/components/common/PanelLoading";
import { queryKeys } from "@/api/queryKeys";
import AudioBar from "@/components/layout/AudioBar";
import TaskBar from "@/components/layout/TaskBar";
import CommandPalette from "@/components/CommandPalette";
import ShortcutsHelp from "@/components/ShortcutsHelp";
import BatchHistoryModal from "@/components/BatchHistoryModal";
import { ConfirmDialogHost } from "@/components/ui/confirm-dialog";
import { PlatformProvider } from "@/platform";
import { useGlobalShortcuts } from "@/hooks/useGlobalShortcuts";

// Boot phases shown while waiting on /api/health. Times are best-effort
// guesses tuned for the PyInstaller onefile cold-start (Tauri release build);
// the sidecar typically takes 10-30 s on first launch each session.
const BOOT_PHASES = [
  { afterMs: 0, label: "Starting up..." },
  { afterMs: 2_000, label: "Setting up the backend..." },
  { afterMs: 8_000, label: "Loading machine learning libraries..." },
  { afterMs: 20_000, label: "Almost ready, hang tight..." },
] as const;

function useElapsedMs(): number {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const id = window.setInterval(() => {
      setElapsed(performance.now() - start);
    }, 500);
    return () => window.clearInterval(id);
  }, []);
  return elapsed;
}

function pickPhaseLabel(elapsedMs: number): string {
  let current: string = BOOT_PHASES[0].label;
  for (const phase of BOOT_PHASES) {
    if (elapsedMs >= phase.afterMs) current = phase.label;
  }
  return current;
}

export default function RootLayout() {
  useGlobalShortcuts();
  // PyInstaller-bundled sidecar can take 10-30 s to extract + boot uvicorn on
  // the first launch each session, so we retry generously before surrendering.
  const { data: health, error } = useQuery({
    queryKey: queryKeys.health(),
    queryFn: getHealth,
    retry: 60,
    retryDelay: (attempt) => Math.min(500 + attempt * 500, 3000),
  });

  const elapsedMs = useElapsedMs();

  if (error) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center space-y-4">
          <h1 className="text-2xl font-bold text-destructive">
            Backend not reachable
          </h1>
          <p className="text-muted-foreground text-sm">
            Make sure the API is running on port 18811
          </p>
          <code className="text-xs text-muted-foreground block">make dev-api</code>
        </div>
      </div>
    );
  }

  if (!health) {
    const label = pickPhaseLabel(elapsedMs);
    const showHint = elapsedMs >= BOOT_PHASES[2].afterMs;
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-center">
          <Loader2 className="size-6 animate-spin text-muted-foreground" />
          <p className="text-muted-foreground">{label}</p>
          {showHint && (
            <p className="text-xs text-muted-foreground/70 max-w-xs">
              First launch each session can take up to 30 seconds while the
              bundled backend extracts.
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <PlatformProvider>
      <div className="flex flex-col h-screen overflow-hidden bg-background text-foreground">
        <main className="flex-1 overflow-hidden">
          <Suspense fallback={<PanelLoading />}>
            <Outlet />
          </Suspense>
        </main>
        <TaskBar />
        <AudioBar />
        <ConfirmDialogHost />
        <CommandPalette />
        <ShortcutsHelp />
        <BatchHistoryModal />
      </div>
    </PlatformProvider>
  );
}
