/**
 * Single facility for "show task log" widgets: Terminal icon, ChevronRight /
 * ChevronDown toggle, a "Logs (N)" count, CopyButton, and a scrollable
 * monospace pre that auto-scrolls to the latest line when opened. Used by
 * TaskBar (Batch / Episode strips) and the inline ProgressBar.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Terminal } from "lucide-react";
import CopyButton from "@/components/common/CopyButton";

export interface LogSectionProps {
  log: string[];
  /** Optional className for the outer container; lets callers tune padding /
   *  borders without forking the component. */
  className?: string;
}

export default function LogSection({ log, className }: LogSectionProps) {
  const [showLog, setShowLog] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showLog) return;
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log.length, showLog]);

  // Memoize so CopyButton receives a stable string and the join only fires
  // when the log array actually changes (logs can grow to hundreds of lines).
  const joined = useMemo(() => log.join("\n"), [log]);

  if (log.length === 0) return null;

  return (
    <div className={className ?? "px-4 pb-2"}>
      <div className="flex items-center gap-2">
        <button
          onClick={(e) => {
            e.stopPropagation();
            setShowLog(!showLog);
          }}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
        >
          <Terminal className="w-3 h-3" />
          {showLog ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          Logs ({log.length})
        </button>
        <CopyButton value={joined} title="Copy log" />
      </div>
      {showLog && (
        <pre className="mt-1.5 p-2 bg-muted rounded text-3xs leading-normal text-muted-foreground max-h-80 overflow-auto font-mono">
          {log.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
          <div ref={logEndRef} />
        </pre>
      )}
    </div>
  );
}
