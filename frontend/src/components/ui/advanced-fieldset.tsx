import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface AdvancedFieldsetProps {
  legend: string;
  description?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
  className?: string;
}

export function AdvancedFieldset({
  legend,
  description,
  defaultOpen = false,
  children,
  className,
}: AdvancedFieldsetProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={cn("border-t border-border pt-5", className)}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex items-start gap-2 w-full text-left group"
      >
        <ChevronRight
          className={cn(
            "w-4 h-4 mt-0.5 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-90",
          )}
        />
        <div className="min-w-0">
          <span className="text-sm font-semibold">{legend}</span>
          {description && (
            <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
          )}
        </div>
      </button>
      {open && (
        <div className="mt-3 pl-6 divide-y divide-border/40">{children}</div>
      )}
    </div>
  );
}
