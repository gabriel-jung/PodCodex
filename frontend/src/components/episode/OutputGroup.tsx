import { useState, type ReactNode } from "react";
import { ChevronRight, type LucideIcon } from "lucide-react";

interface Props {
  title: string;
  icon: LucideIcon;
  /** Number shown as a muted count next to the title. */
  count: number;
  /** One-line hint shown in collapsed state (e.g. latest version summary). */
  summary?: string;
  /** Open by default (caller decides — e.g. open if edited latest / small). */
  defaultOpen?: boolean;
  /** Body content — typically a stack of VersionRow-like rows. */
  children: ReactNode;
}

export default function OutputGroup({
  title,
  icon: Icon,
  count,
  summary,
  defaultOpen = false,
  children,
}: Props) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="rounded-lg border border-border/50 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-4 py-2.5 flex items-center gap-3 hover:bg-accent/30 transition text-left"
      >
        <ChevronRight
          className={`w-3.5 h-3.5 text-muted-foreground shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
        />
        <Icon className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
        <span className="text-sm font-medium shrink-0">{title}</span>
        <span className="text-2xs text-muted-foreground font-mono tabular-nums shrink-0">
          {count}
        </span>
        {summary && !open && (
          <span className="text-2xs text-muted-foreground truncate">
            {summary}
          </span>
        )}
      </button>
      {open && (
        <div className="divide-y divide-border/40 border-t border-border/40">
          {children}
        </div>
      )}
    </div>
  );
}
