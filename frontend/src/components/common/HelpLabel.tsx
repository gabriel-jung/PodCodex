import { useRef, useState } from "react";
import { HelpCircle } from "lucide-react";

interface HelpLabelProps {
  label: string;
  help: string;
}

export default function HelpLabel({ label, help }: HelpLabelProps) {
  const [show, setShow] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);

  const getStyle = (): React.CSSProperties | undefined => {
    if (!show || !btnRef.current) return undefined;
    const rect = btnRef.current.getBoundingClientRect();
    return { position: "fixed", left: rect.left, top: rect.bottom + 4, zIndex: 50 };
  };

  return (
    <label className="text-muted-foreground flex items-center gap-1">
      {label}
      <button
        ref={btnRef}
        type="button"
        onClick={() => setShow(!show)}
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="text-muted-foreground/50 hover:text-muted-foreground transition"
      >
        <HelpCircle className="w-3 h-3" />
      </button>
      {show && (
        <div style={getStyle()} className="bg-popover text-popover-foreground text-xs rounded-md border border-border shadow-lg px-2.5 py-1.5 max-w-sm whitespace-normal">
          {help}
        </div>
      )}
    </label>
  );
}
