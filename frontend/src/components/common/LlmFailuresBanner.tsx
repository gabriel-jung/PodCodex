/** Warning banner for batches an auto correct/translate run silently
 *  rejected (count drift / parse failure). Each rejected batch is hand-fixed
 *  in the reconcile modal; fixes accumulate and apply together as one new
 *  version, so fixing N batches does not create N versions. */

import { useState } from "react";
import { AlertTriangle, ChevronRight, Wrench, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { errorMessage } from "@/lib/utils";
import BatchReconcileModal from "@/components/common/BatchReconcileModal";
import type { LlmBatchRecord, LlmFailures } from "@/api/llmFailures";
import type { BatchFix } from "@/api/types";

export default function LlmFailuresBanner({
  failures,
  stepLabel,
  onDismiss,
  dismissing,
  onApplyFixes,
}: {
  failures: LlmFailures | null | undefined;
  /** "correction" or "translation" — names the run in the banner copy. */
  stepLabel: string;
  onDismiss: () => void;
  dismissing?: boolean;
  /** Apply the accumulated batch fixes. Resolves once the new version saves. */
  onApplyFixes: (fixes: BatchFix[]) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [fixRecord, setFixRecord] = useState<LlmBatchRecord | null>(null);
  const [fixes, setFixes] = useState<BatchFix[]>([]);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!failures || failures.rejected === 0) return null;
  const rejected = failures.batches.filter((b) => b.status === "rejected");
  const editingFix = fixRecord
    ? fixes.find((f) => f.batch === fixRecord.batch)
    : undefined;

  const handleApply = async () => {
    setApplying(true);
    setError(null);
    try {
      await onApplyFixes(fixes);
      setFixes([]);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setApplying(false);
    }
  };

  return (
    <div className="rounded border border-destructive/30 bg-destructive/10 text-xs">
      <div className="flex items-center gap-2 px-3 py-2">
        <AlertTriangle className="w-3 h-3 text-destructive shrink-0" />
        <span className="text-destructive flex-1">
          {failures.rejected} of {failures.total_batches} batches rejected in the
          last {stepLabel} run ({failures.mode}); those segments kept their
          original text.
        </span>
        <button
          onClick={() => setOpen(!open)}
          className="text-destructive/80 hover:text-destructive shrink-0"
          title={open ? "Hide details" : "Show details"}
        >
          <ChevronRight className={`w-3 h-3 transition-transform ${open ? "rotate-90" : ""}`} />
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
        <div className="border-t border-destructive/20">
          <div className="divide-y divide-destructive/10">
            {rejected.map((b) => {
              const fixed = fixes.some((f) => f.batch === b.batch);
              return (
                <div key={b.batch} className="flex items-center gap-2 px-3 py-2">
                  {fixed ? (
                    <Check className="w-3 h-3 text-success shrink-0" />
                  ) : (
                    <span className="w-3 shrink-0" aria-hidden />
                  )}
                  <span className="flex-1 text-muted-foreground">
                    Batch {b.batch}: {b.reason} (expected {b.expected}, got {b.got})
                  </span>
                  <Button
                    onClick={() => setFixRecord(b)}
                    variant="outline"
                    size="sm"
                    className="h-6 px-2 text-xs shrink-0"
                  >
                    {fixed ? (
                      "Edit"
                    ) : (
                      <><Wrench className="w-3 h-3 mr-1" /> Fix</>
                    )}
                  </Button>
                </div>
              );
            })}
          </div>
          <div className="flex items-center gap-2 px-3 py-2 border-t border-destructive/20">
            <span className="flex-1 text-muted-foreground">
              {fixes.length} of {rejected.length} fixed
            </span>
            {error && <span className="text-destructive shrink-0">{error}</span>}
            <Button
              onClick={handleApply}
              disabled={fixes.length === 0 || applying}
              size="sm"
              className="h-6 px-2 text-xs shrink-0"
            >
              {applying
                ? "Applying…"
                : `Apply ${fixes.length} fix${fixes.length === 1 ? "" : "es"}`}
            </Button>
          </div>
        </div>
      )}
      {fixRecord && (
        <BatchReconcileModal
          open
          onOpenChange={(o) => { if (!o) setFixRecord(null); }}
          title={`Fix batch ${fixRecord.batch}`}
          inputEntries={fixRecord.input}
          initialResponse={
            editingFix
              ? JSON.stringify(editingFix.corrections, null, 2)
              : fixRecord.raw
          }
          onResolve={(objs) => {
            setFixes((prev) => [
              ...prev.filter((f) => f.batch !== fixRecord.batch),
              { batch: fixRecord.batch, corrections: objs },
            ]);
            setFixRecord(null);
          }}
          applyLabel="Save fix"
        />
      )}
    </div>
  );
}
