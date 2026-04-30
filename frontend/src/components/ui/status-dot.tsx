import { cn } from "@/lib/utils";

export type StatusState = "ok" | "busy" | "warn" | "err" | "idle";

const CLASS_MAP: Record<StatusState, string> = {
  ok: "bg-success",
  busy: "bg-info animate-pulse",
  warn: "bg-warning",
  err: "bg-destructive",
  idle: "bg-muted-foreground/40",
};

interface StatusDotProps {
  state: StatusState;
  label?: string;
  className?: string;
}

export function StatusDot({ state, label, className }: StatusDotProps) {
  return (
    <span
      role={label ? "status" : undefined}
      aria-label={label}
      className={cn("inline-block w-1.5 h-1.5 rounded-full", CLASS_MAP[state], className)}
    />
  );
}
