import { useState } from "react";
import { Settings2 } from "lucide-react";

interface AdvancedToggleProps {
  children: React.ReactNode;
  /** Extra classes on the outer wrapper. */
  className?: string;
}

/**
 * Collapsible "Advanced settings" section with the Settings2 gear icon.
 * Manages its own open/closed state.
 */
export default function AdvancedToggle({ children, className }: AdvancedToggleProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className={className}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
      >
        <Settings2 className="w-3 h-3" />
        <span className="font-semibold uppercase tracking-wide">
          {open ? "Hide advanced" : "Advanced settings"}
        </span>
      </button>
      {open && children}
    </div>
  );
}
