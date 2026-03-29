import { useState } from "react";
import { HelpCircle } from "lucide-react";

interface HelpLabelProps {
  label: string;
  help: string;
}

export default function HelpLabel({ label, help }: HelpLabelProps) {
  const [show, setShow] = useState(false);

  return (
    <label className="text-muted-foreground flex items-center gap-1 relative">
      {label}
      <button
        type="button"
        onClick={() => setShow(!show)}
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="text-muted-foreground/50 hover:text-muted-foreground transition"
      >
        <HelpCircle className="w-3 h-3" />
      </button>
      {show && (
        <div className="absolute left-0 top-full mt-1 z-50 bg-popover text-popover-foreground text-xs rounded-md border border-border shadow-lg px-2.5 py-1.5 max-w-60 whitespace-normal">
          {help}
        </div>
      )}
    </label>
  );
}
