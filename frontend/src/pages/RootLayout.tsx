import { useQuery } from "@tanstack/react-query";
import { Outlet } from "@tanstack/react-router";
import { getHealth, getExtras } from "@/api/client";
import AudioBar from "@/components/layout/AudioBar";
import TaskBar from "@/components/layout/TaskBar";
import { ConfirmDialogHost } from "@/components/ui/confirm-dialog";
import { PlatformProvider } from "@/platform";
import { useTheme } from "@/hooks/useTheme";
import { useNavigate } from "@tanstack/react-router";
import { Sun, Moon, Monitor, Settings } from "lucide-react";

export default function RootLayout() {
  const { data: health, error } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: 3,
    retryDelay: 1000,
  });

  // Prefetch capabilities at startup so panels never flash "not installed"
  useQuery({
    queryKey: ["system", "extras"],
    queryFn: getExtras,
    staleTime: Infinity,
    gcTime: Infinity,
    enabled: !!health,
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
      <div className="flex flex-col h-screen bg-background text-foreground">
        <main className="flex-1 overflow-hidden">
          <Outlet />
        </main>
        <TaskBar />
        <AudioBar />
        <FloatingActions />
        <ConfirmDialogHost />
      </div>
    </PlatformProvider>
  );
}

function FloatingActions() {
  const navigate = useNavigate();
  const { theme, setTheme } = useTheme();
  const next = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
  const ThemeIcon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor;
  return (
    <div className="fixed bottom-24 right-4 flex items-center gap-px rounded-lg bg-card border border-border shadow-sm z-50 overflow-hidden">
      <button
        onClick={() => navigate({ to: "/settings" })}
        title="Settings"
        className="p-2 text-muted-foreground hover:text-foreground hover:bg-accent transition"
      >
        <Settings className="w-4 h-4" />
      </button>
      <div className="w-px h-5 bg-border" />
      <button
        onClick={() => setTheme(next)}
        title={`Theme: ${theme} (click for ${next})`}
        className="p-2 text-muted-foreground hover:text-foreground hover:bg-accent transition"
      >
        <ThemeIcon className="w-4 h-4" />
      </button>
    </div>
  );
}
