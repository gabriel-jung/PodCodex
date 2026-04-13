import type { LucideIcon } from "lucide-react";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "./button";

export interface EmptyStateStep {
  label: string;
  done?: boolean;
  count?: number;
}

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  /** Ordered pipeline-style next-step hints. */
  steps?: EmptyStateStep[];
  /** Show a dashed border container. */
  dashed?: boolean;
  className?: string;
}

export function EmptyState({ icon: Icon, title, description, action, secondaryAction, steps, dashed, className }: EmptyStateProps) {
  return (
    <div className={cn(
      "flex flex-col items-center justify-center py-16 px-6 text-center",
      dashed && "border-2 border-dashed border-border rounded-xl mx-6 my-6",
      className,
    )}>
      {Icon && (
        <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
          <Icon className="w-6 h-6 text-muted-foreground" />
        </div>
      )}
      <h3 className="text-sm font-medium mb-1">{title}</h3>
      {description && (
        <p className="text-sm text-muted-foreground max-w-sm">{description}</p>
      )}
      {steps && steps.length > 0 && (
        <ol className="mt-5 space-y-1.5 text-left text-sm">
          {steps.map((s, i) => (
            <li key={i} className="flex items-center gap-2.5 text-muted-foreground">
              <span
                className={cn(
                  "w-5 h-5 rounded-full flex items-center justify-center text-2xs font-medium shrink-0",
                  s.done ? "bg-success text-white" : "bg-muted text-foreground",
                )}
              >
                {s.done ? <Check className="w-3 h-3" /> : i + 1}
              </span>
              <span className={cn(s.done && "line-through")}>{s.label}</span>
              {s.count != null && s.count > 0 && !s.done && (
                <span className="text-xs text-primary">({s.count} ready)</span>
              )}
            </li>
          ))}
        </ol>
      )}
      {(action || secondaryAction) && (
        <div className="flex gap-2 mt-5">
          {action && (
            <Button onClick={action.onClick} variant="outline" size="sm">
              {action.label}
            </Button>
          )}
          {secondaryAction && (
            <Button onClick={secondaryAction.onClick} variant="ghost" size="sm">
              {secondaryAction.label}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
