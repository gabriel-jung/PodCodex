/** Modal for hand-fixing a single LLM batch whose response did not line up
 *  with its input segments (count drift / parse failure). Wraps ReconcileView,
 *  owns validation and the apply action. Used by LlmFailuresBanner (fix a
 *  rejected auto-run batch) and ManualModePanel (fix a pasted batch). */

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import ReconcileView from "@/components/common/ReconcileView";
import { reconcileBatch, parseResponseObjects, type InputEntry } from "@/lib/reconcile";
import { errorMessage } from "@/lib/utils";

/** Pretty-print a raw response so the formatted view reads one entry per line. */
function prettify(raw: string): string {
  const objs = parseResponseObjects(raw);
  return objs ? JSON.stringify(objs, null, 2) : raw;
}

export default function BatchReconcileModal({
  open,
  onOpenChange,
  title,
  inputEntries,
  initialResponse,
  onResolve,
  applyLabel = "Apply fix",
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  inputEntries: InputEntry[];
  initialResponse: string;
  /** Called with the validated, count-matched response entries. */
  onResolve: (objs: Record<string, unknown>[]) => Promise<void> | void;
  applyLabel?: string;
}) {
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [applying, setApplying] = useState(false);

  useEffect(() => {
    if (open) {
      setText(prettify(initialResponse));
      setError(null);
      setApplying(false);
    }
  }, [open, initialResponse]);

  const handleApply = async () => {
    const result = reconcileBatch(text, inputEntries);
    if ("error" in result) {
      setError(result.error);
      return;
    }
    setError(null);
    setApplying(true);
    try {
      await onResolve(result.objs);
    } catch (e) {
      setError(errorMessage(e));
      return;
    } finally {
      setApplying(false);
    }
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-3xl bg-popover">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>
            Edit the response so it has exactly {inputEntries.length}{" "}
            {inputEntries.length === 1 ? "entry" : "entries"}, one per input segment.
          </DialogDescription>
        </DialogHeader>
        <ReconcileView
          inputEntries={inputEntries}
          value={text}
          onChange={(t) => { setText(t); setError(null); }}
        />
        {error && <p className="text-destructive text-xs">{error}</p>}
        <DialogFooter>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onOpenChange(false)}
            disabled={applying}
          >
            Cancel
          </Button>
          <Button size="sm" onClick={handleApply} disabled={applying || !text.trim()}>
            {applying ? "Applying..." : applyLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
