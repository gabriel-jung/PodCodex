import { useQuery } from "@tanstack/react-query";
import { Outlet } from "@tanstack/react-router";
import { getHealth } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import AudioBar from "@/components/layout/AudioBar";
import TaskBar from "@/components/layout/TaskBar";
import CommandPalette from "@/components/CommandPalette";
import { ConfirmDialogHost } from "@/components/ui/confirm-dialog";
import { PlatformProvider } from "@/platform";
import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useTheme } from "@/hooks/useTheme";
import { useLayoutStore } from "@/stores";
import { Home, Sun, Moon, Monitor, Settings, PanelLeftOpen, PanelLeftClose } from "lucide-react";

export default function RootLayout() {
  const { data: health, error } = useQuery({
    queryKey: queryKeys.health(),
    queryFn: getHealth,
    retry: 3,
    retryDelay: 1000,
  });

  const hideAppSidebar = useLayoutStore((s) => s.hideAppSidebar);

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
      <div className="flex h-screen bg-background text-foreground">
        {!hideAppSidebar && <AppSidebar />}
        <div className="flex flex-col flex-1 overflow-hidden">
          <main className="flex-1 overflow-hidden">
            <Outlet />
          </main>
          <TaskBar />
          <AudioBar />
        </div>
        <ConfirmDialogHost />
        <CommandPalette />
      </div>
    </PlatformProvider>
  );
}

function AppSidebar() {
  const navigate = useNavigate();
  const [expanded, setExpanded] = useState(false);
  const { theme, setTheme } = useTheme();
  const nextTheme = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
  const ThemeIcon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor;

  const navItems = [
    { icon: Home, label: "Home", onClick: () => navigate({ to: "/" }) },
  ];
  const bottomItems = [
    { icon: Settings, label: "Settings", onClick: () => navigate({ to: "/settings" }) },
    { icon: ThemeIcon, label: `Theme: ${theme}`, onClick: () => setTheme(nextTheme) },
  ];

  const renderItem = ({ icon: Icon, label, onClick }: typeof navItems[number]) => (
    <button
      key={label}
      onClick={onClick}
      title={expanded ? undefined : label}
      className="w-full flex items-center gap-3 px-3 py-2 text-sm text-muted-foreground hover:text-foreground hover:bg-accent/50 transition"
    >
      <Icon className="w-4 h-4 shrink-0" />
      {expanded && <span className="truncate text-xs">{label}</span>}
    </button>
  );

  return (
    <div
      className={`border-r border-border flex flex-col shrink-0 transition-all duration-200 ${
        expanded ? "w-44" : "w-11"
      }`}
    >
      <nav className="py-3 flex flex-col gap-1">
        {navItems.map(renderItem)}
      </nav>

      <div className="flex-1" />

      <div className="flex flex-col gap-1 py-2 border-t border-border">
        {bottomItems.map(renderItem)}
      </div>

      <button
        onClick={() => setExpanded(!expanded)}
        className="px-3 py-2 text-muted-foreground hover:text-foreground transition border-t border-border"
      >
        {expanded ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeftOpen className="w-4 h-4" />}
      </button>
    </div>
  );
}
