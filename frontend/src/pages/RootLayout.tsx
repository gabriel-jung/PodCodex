import { Suspense, useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Outlet } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { healthQueryOptions } from "@/api/client";
import PanelLoading from "@/components/common/PanelLoading";
import AudioBar from "@/components/layout/AudioBar";
import { sidebarPad } from "@/lib/sidebar";
import TaskBar from "@/components/layout/TaskBar";
import CommandPalette from "@/components/CommandPalette";
import ShortcutsHelp from "@/components/ShortcutsHelp";
import BatchHistoryModal from "@/components/BatchHistoryModal";
import { ConfirmDialogHost } from "@/components/ui/confirm-dialog";
import { PlatformProvider } from "@/platform";
import { useGlobalShortcuts } from "@/hooks/useGlobalShortcuts";
import { useVersions } from "@/hooks/useVersions";
import { useLayoutStore } from "@/stores";

// Labels for the boot banner. The shell renders immediately now, so these
// annotate a backend still coming up behind a usable UI rather than gating
// it. A warm launch reaches /api/health in a few seconds; the later phases
// only ever show on the first launch after an install, when the OS is still
// validating a freshly written sidecar.
const BOOT_PHASES = [
  { afterMs: 0, label: "Connecting to the backend..." },
  { afterMs: 8_000, label: "Still starting up..." },
  { afterMs: 20_000, label: "Almost ready, hang tight..." },
] as const;

// Grace period before the banner appears. A normal launch answers well
// inside it, so the banner never flashes on a healthy start.
const BANNER_AFTER_MS = 1_200;

/** Ticks only while ``running``. It drives the boot banner's label, so once
 *  the backend answers it must stop: this re-renders the whole shell
 *  (Outlet, sidebar, AudioBar, TaskBar) on every tick, and a session lasts
 *  hours. */
function useElapsedMs(running: boolean): number {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!running) return;
    const start = performance.now();
    const id = window.setInterval(() => {
      setElapsed(performance.now() - start);
    }, 500);
    return () => window.clearInterval(id);
  }, [running]);
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
  // Shown on the backend-unreachable screen: the shell version resolves
  // without the backend, so users can identify their build even when the
  // sidecar never comes up. On Windows there is no app menu to fall back on.
  const { display: displayVersion } = useVersions();
  // Sidebar is fixed full-window-height; the rest of the shell sits in the
  // column to its right so AudioBar/TaskBar growth never reflows the sidebar.
  const sidebarExpanded = useLayoutStore((s) => s.sidebarExpanded);
  // Retry schedule lives in healthQueryOptions: this is the one query
  // patient enough to wait out a slow first launch, and its first success
  // is what refetches everything else (see main.tsx). Every observer of
  // this key must agree on the schedule.
  const { data: health, error } = useQuery(healthQueryOptions);

  const elapsedMs = useElapsedMs(!health);

  if (error) {
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <div className="text-center space-y-4">
          <h1 className="text-2xl font-semibold text-destructive">
            Backend not reachable
          </h1>
          <p className="text-muted-foreground text-sm">
            Make sure the API is running on port 18811
          </p>
          <code className="text-xs text-muted-foreground block">make dev-api</code>
          {displayVersion && (
            <p className="font-mono text-xs text-muted-foreground/60">v{displayVersion}</p>
          )}
        </div>
      </div>
    );
  }

  // Deliberately no gate on `health` here. Nothing in the shell needs it:
  // every consumer already reads it as optional and degrades. Blocking the
  // whole UI behind the health round-trip made a launch feel as slow as the
  // sidecar's slowest phase, when the sidebar, routing and settings chrome
  // were ready the moment the webview painted. Data panes show their own
  // pending state, and the global retry policy in main.tsx keeps their
  // queries patient while the port is still closed.
  const showBootBanner = !health && elapsedMs >= BANNER_AFTER_MS;

  return (
    <PlatformProvider>
      <div
        className={`flex flex-col h-screen overflow-hidden bg-background text-foreground transition-[padding] duration-200 ${sidebarPad(sidebarExpanded)}`}
      >
        {showBootBanner && (
          <div
            role="status"
            className="flex items-center justify-center gap-2 border-b border-border bg-card px-3 py-1.5 text-xs text-muted-foreground"
          >
            <Loader2 className="size-3.5 animate-spin" />
            <span>{pickPhaseLabel(elapsedMs)}</span>
          </div>
        )}
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
