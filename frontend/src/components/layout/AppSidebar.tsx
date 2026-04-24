import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useTheme } from "@/hooks/useTheme";
import { useLayoutStore } from "@/stores";
import {
  ArrowLeft, Home, Podcast, Settings, SunMoon,
  PanelLeftOpen, PanelLeftClose,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

export interface SidebarItem {
  key: string;
  label: string;
  icon: LucideIcon;
  status?: "done" | "partial" | false;
}

export interface SidebarSection {
  items: SidebarItem[];
}

export default function AppSidebar({ parentLabel, onParent, pageSections, activeItem, onItemClick }: {
  /** Optional parent link shown between Back and Home (e.g. "Show name" on episode pages). */
  parentLabel?: string;
  onParent?: () => void;
  pageSections?: SidebarSection[];
  activeItem?: string;
  onItemClick?: (key: string) => void;
}) {
  const navigate = useNavigate();
  const isHome = useRouterState({ select: (s) => s.location.pathname === "/" });
  const historyIndex = useRouterState({ select: (s) => s.location.state.__TSR_index ?? 0 });
  const hideBack = historyIndex === 0;
  const expanded = useLayoutStore((s) => s.sidebarExpanded);
  const setExpanded = useLayoutStore((s) => s.setSidebarExpanded);
  const { theme, setTheme } = useTheme();
  const nextTheme = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";

  return (
    <div
      className={`border-r border-border flex flex-col shrink-0 transition-all duration-200 ${
        expanded ? "w-48" : "w-14"
      }`}
    >
      <nav className="flex-1 py-2 flex flex-col overflow-y-auto">
        {/* Back + Parent + Home */}
        {!hideBack && (
          <SidebarBtn icon={ArrowLeft} label="Back" expanded={expanded} onClick={() => {
            if (historyIndex > 0) window.history.back();
            else navigate({ to: "/" });
          }} />
        )}
        {parentLabel && onParent && (
          <SidebarBtn icon={Podcast} label={parentLabel} expanded={expanded} onClick={onParent} />
        )}
        {!isHome && <SidebarBtn icon={Home} label="Home" expanded={expanded} onClick={() => navigate({ to: "/" })} />}

        {/* Page-specific sections */}
        {pageSections?.map((section, si) => (
          <div key={si}>
            <div className="mx-3 my-1.5 border-t border-border" />
            {section.items.map(({ key, label, icon: Icon, status }) => (
              <button
                key={key}
                onClick={() => onItemClick?.(key)}
                title={expanded ? undefined : label}
                className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm transition ${
                  activeItem === key
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                }`}
              >
                <Icon className="w-5 h-5 shrink-0" />
                {expanded && <span className="truncate">{label}</span>}
                {status && (
                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${expanded ? "ml-auto" : ""} ${status === "partial" ? "bg-blue-500" : "bg-success"}`} />
                )}
              </button>
            ))}
          </div>
        ))}
      </nav>

      {/* Bottom: Settings + Theme */}
      <div className="flex flex-col border-t border-border py-1">
        <SidebarBtn icon={Settings} label="Settings" expanded={expanded} onClick={() => navigate({ to: "/settings" })} />
        <SidebarBtn icon={SunMoon} label={`Theme: ${theme}`} expanded={expanded} onClick={() => setTheme(nextTheme)} />
      </div>

      {/* Expand toggle */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="px-4 py-3 text-muted-foreground hover:text-foreground transition border-t border-border"
      >
        {expanded ? <PanelLeftClose className="w-5 h-5" /> : <PanelLeftOpen className="w-5 h-5" />}
      </button>
    </div>
  );
}

function SidebarBtn({ icon: Icon, label, expanded, onClick }: {
  icon: LucideIcon;
  label: string;
  expanded: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={expanded ? undefined : label}
      aria-label={label}
      className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-muted-foreground hover:text-foreground hover:bg-accent/50 transition"
    >
      <Icon className="w-5 h-5 shrink-0" />
      {expanded && <span className="truncate">{label}</span>}
    </button>
  );
}
