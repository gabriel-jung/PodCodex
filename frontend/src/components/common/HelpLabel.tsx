import { useLayoutEffect, useRef, useState } from "react";
import { HelpCircle } from "lucide-react";

interface HelpLabelProps {
  label: string;
  /** Optional tooltip text. When omitted, the component renders as a plain
   *  row label — same typography, no help icon. */
  help?: string;
}

export default function HelpLabel({ label, help }: HelpLabelProps) {
  const [show, setShow] = useState(false);
  const [style, setStyle] = useState<React.CSSProperties | undefined>(undefined);
  const btnRef = useRef<HTMLButtonElement>(null);

  useLayoutEffect(() => {
    if (!show || !btnRef.current) {
      setStyle(undefined);
      return;
    }
    const rect = btnRef.current.getBoundingClientRect();
    setStyle({ position: "fixed", left: rect.left, top: rect.bottom + 4, zIndex: 50 });
  }, [show]);

  return (
    <label className="text-muted-foreground flex items-center gap-1">
      {label}
      {help && (
        <>
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
            <div style={style} className="bg-popover text-popover-foreground text-xs rounded-md border border-border shadow-lg px-2.5 py-1.5 max-w-sm whitespace-normal">
              {help}
            </div>
          )}
        </>
      )}
    </label>
  );
}
