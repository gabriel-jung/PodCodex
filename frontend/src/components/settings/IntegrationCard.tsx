import { ChevronRight } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { StatusDot, type StatusState } from "@/components/ui/status-dot";

interface IntegrationCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  statusState: StatusState;
  statusLabel: string;
  onOpen: () => void;
  className?: string;
}

export function IntegrationCard({
  icon: Icon,
  title,
  description,
  statusState,
  statusLabel,
  onOpen,
  className,
}: IntegrationCardProps) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className={cn(
        "group text-left flex flex-col gap-3 rounded-lg border border-border bg-card px-5 py-5 transition",
        "hover:border-primary/40 hover:shadow-sm hover:-translate-y-px",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary",
        className,
      )}
    >
      <div className="flex items-center justify-between">
        <Icon className="w-6 h-6 text-foreground" />
        <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
          <StatusDot state={statusState} />
          {statusLabel}
        </span>
      </div>
      <div className="min-w-0">
        <h3 className="text-base font-semibold">{title}</h3>
        <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
          {description}
        </p>
      </div>
      <div className="flex items-center gap-1 text-xs text-muted-foreground mt-auto pt-2 group-hover:text-foreground transition">
        Manage
        <ChevronRight className="w-3.5 h-3.5" />
      </div>
    </button>
  );
}
