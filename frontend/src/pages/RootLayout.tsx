import { useQuery } from "@tanstack/react-query";
import { Outlet } from "@tanstack/react-router";
import { getHealth } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import AudioBar from "@/components/layout/AudioBar";
import TaskBar from "@/components/layout/TaskBar";
import CommandPalette from "@/components/CommandPalette";
import ShortcutsHelp from "@/components/ShortcutsHelp";
import BatchHistoryModal from "@/components/BatchHistoryModal";
import { ConfirmDialogHost } from "@/components/ui/confirm-dialog";
import { PlatformProvider } from "@/platform";
import { useGlobalShortcuts } from "@/hooks/useGlobalShortcuts";

export default function RootLayout() {
  useGlobalShortcuts();
  const { data: health, error } = useQuery({
    queryKey: queryKeys.health(),
    queryFn: getHealth,
    retry: 3,
    retryDelay: 1000,
  });

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
    return (
      <div className="min-h-screen bg-background text-foreground flex items-center justify-center">
        <p className="text-muted-foreground">Connecting...</p>
      </div>
    );
  }

  return (
    <PlatformProvider>
      <div className="flex flex-col h-screen overflow-hidden bg-background text-foreground">
        <main className="flex-1 overflow-hidden">
          <Outlet />
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
