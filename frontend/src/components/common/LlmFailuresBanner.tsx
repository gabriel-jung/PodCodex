/** Warning banner for batches an auto correct/translate run silently
 *  rejected (count drift / parse failure). Expandable to the raw LLM
 *  response + input segments so the user can fix the batch by hand. */

import { useState } from "react";
import { AlertTriangle, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { LlmBatchRecord, LlmFailures } from "@/api/llmFailures";

function BatchDetail({ record }: { record: LlmBatchRecord }) {
  const [showRaw, setShowRaw] = useState(false);
  return (
    <div className="px-3 py-2 space-y-1">
      <div className="text-muted-foreground">
        Batch {record.batch} — {record.reason} (expected {record.expected}, got {record.got})
      </div>
      <button
        onClick={() => setShowRaw(!showRaw)}
        className="text-2xs text-muted-foreground/80 underline"
      >
        {showRaw ? "Hide" : "Show"} raw response &amp; input
      </button>
      {showRaw && (
        <div className="grid grid-cols-2 gap-2 pt-1">
          <div>
            <div className="text-2xs text-muted-foreground mb-1">
              Input segments ({record.input.length})
            </div>
            <div className="max-h-48 overflow-y-auto text-2xs leading-relaxed bg-background/50 rounded p-2">
              {record.input.map((s) => (
                <div key={s.index}>
                  <span className="font-mono text-muted-foreground/60 mr-1">[{s.index}]</span>
                  {s.text}
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="text-2xs text-muted-foreground mb-1">Raw LLM response</div>
            <pre className="max-h-48 overflow-auto text-2xs bg-background/50 rounded p-2 whitespace-pre-wrap">
              {record.raw}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default function LlmFailuresBanner({
  failures,
  onDismiss,
  dismissing,
}: {
  failures: LlmFailures | null | undefined;
  onDismiss: () => void;
  dismissing?: boolean;
}) {
  const [open, setOpen] = useState(false);
  if (!failures || failures.rejected === 0) return null;
  const rejected = failures.batches.filter((b) => b.status === "rejected");

  return (
    <div className="rounded border border-destructive/30 bg-destructive/10 text-xs">
      <div className="flex items-center gap-2 px-3 py-2">
        <AlertTriangle className="w-3.5 h-3.5 text-destructive shrink-0" />
        <span className="text-destructive flex-1">
          {failures.rejected} of {failures.total_batches} batches rejected in the
          last {failures.mode} run — those segments kept their original text.
        </span>
        <button
          onClick={() => setOpen(!open)}
          className="text-destructive/80 hover:text-destructive shrink-0"
          title={open ? "Hide details" : "Show details"}
        >
          <ChevronRight className={`w-3.5 h-3.5 transition-transform ${open ? "rotate-90" : ""}`} />
        </button>
        <Button
          onClick={onDismiss}
          disabled={dismissing}
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs shrink-0"
        >
          Dismiss
        </Button>
      </div>
      {open && (
        <div className="border-t border-destructive/20 divide-y divide-destructive/10">
          {rejected.map((b) => (
            <BatchDetail key={b.batch} record={b} />
          ))}
        </div>
      )}
    </div>
  );
}
