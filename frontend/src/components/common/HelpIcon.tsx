import { useLayoutEffect, useRef, useState } from "react";
import { HelpCircle } from "lucide-react";

interface HelpIconProps {
  help: string;
}

/**
 * Small `(?)` icon that reveals a positioned tooltip on hover / click.
 * Shared between HelpLabel (field-level) and SectionHeader (section-level)
 * so the popover behaviour stays identical across both surfaces.
 */
export default function HelpIcon({ help }: HelpIconProps) {
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
        <div
          style={style}
          className="bg-popover text-popover-foreground text-xs rounded-md border border-border shadow-lg px-2.5 py-1.5 max-w-sm whitespace-normal normal-case font-normal"
        >
          {help}
        </div>
      )}
    </>
  );
}
