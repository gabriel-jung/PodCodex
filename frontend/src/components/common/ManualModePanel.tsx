import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { errorMessage } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Copy, Check, ChevronLeft, ChevronRight } from "lucide-react";

interface ManualModePanelProps {
  batchMinutes: number;
  generatePrompts: (batchMinutes: number) => Promise<PromptBatch[]>;
  applyCorrections: (corrections: unknown[]) => Promise<unknown>;
  onApplied?: () => void;
}

interface PromptBatch {
  batch_index: number;
  prompt: string;
  segment_count: number;
}

export default function ManualModePanel({
  batchMinutes,
  generatePrompts,
  applyCorrections,
  onApplied,
}: ManualModePanelProps) {
  const [prompts, setPrompts] = useState<PromptBatch[] | null>(null);
  const [currentBatch, setCurrentBatch] = useState(0);
  const [batchResults, setBatchResults] = useState<Record<number, unknown[]>>({});
  const [pastedText, setPastedText] = useState("");
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  const generateMutation = useMutation({
    mutationFn: () => generatePrompts(batchMinutes),
    onSuccess: (data) => {
      setPrompts(data);
      setCurrentBatch(0);
      setBatchResults({});
      setPastedText("");
    },
  });

  const applyMutation = useMutation({
    mutationFn: () => {
      // Combine all batch results in order
      const allCorrections: unknown[] = [];
      for (let i = 0; i < prompts!.length; i++) {
        const batch = batchResults[i];
        if (batch) allCorrections.push(...batch);
      }
      return applyCorrections(allCorrections);
    },
    onSuccess: () => {
      setPrompts(null);
      setBatchResults({});
      setPastedText("");
      setCurrentBatch(0);
      onApplied?.();
    },
  });

  const copyToClipboard = async (text: string, idx: number) => {
    await navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  const validateBatch = () => {
    setParseError(null);
    try {
      const parsed = JSON.parse(pastedText);
      const arr = Array.isArray(parsed) ? parsed : [parsed];
      setBatchResults({ ...batchResults, [currentBatch]: arr });
      setPastedText("");
      // Auto-advance to next unfinished batch
      if (prompts) {
        const next = findNextUnfinished(currentBatch, prompts.length, { ...batchResults, [currentBatch]: arr });
        if (next !== null) setCurrentBatch(next);
      }
    } catch (e) {
      setParseError(`Invalid JSON: ${errorMessage(e)}`);
    }
  };

  const allDone = prompts != null && prompts.every((_, i) => batchResults[i] != null);
  const doneCount = prompts ? prompts.filter((_, i) => batchResults[i] != null).length : 0;
  const batch = prompts?.[currentBatch];
  const batchDone = batchResults[currentBatch] != null;

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        Generate prompts, paste them into your LLM of choice one batch at a time, then paste the JSON response back.
      </p>

      {/* Generate */}
      <div className="flex items-center gap-3 text-sm">
        <Button
          onClick={() => generateMutation.mutate()}
          disabled={generateMutation.isPending}
          variant="outline"
          size="sm"
        >
          {generateMutation.isPending ? "Generating..." : prompts ? "Regenerate" : "Generate prompts"}
        </Button>
      </div>
      {generateMutation.isError && (
        <p className="text-destructive text-xs">{errorMessage(generateMutation.error)}</p>
      )}

      {/* Batch-by-batch workflow */}
      {prompts && batch && (
        <div className="space-y-4">
          {/* Progress bar */}
          <div className="flex items-center gap-3">
            <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${(doneCount / prompts.length) * 100}%` }}
              />
            </div>
            <span className="text-xs text-muted-foreground shrink-0">
              {doneCount}/{prompts.length} batches
            </span>
          </div>

          {/* Batch navigation */}
          <div className="flex items-center gap-2">
            <Button
              onClick={() => setCurrentBatch(currentBatch - 1)}
              disabled={currentBatch === 0}
              variant="ghost"
              size="sm"
              className="h-7 px-2"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </Button>

            <div className="flex gap-1">
              {prompts.map((_, i) => (
                <button
                  key={i}
                  onClick={() => { setCurrentBatch(i); setPastedText(""); setParseError(null); }}
                  className={`w-6 h-6 rounded text-2xs font-medium transition ${
                    i === currentBatch
                      ? "bg-primary text-primary-foreground"
                      : batchResults[i] != null
                        ? "bg-success/20 text-success border border-success/30"
                        : "bg-secondary text-muted-foreground border border-border"
                  }`}
                >
                  {i + 1}
                </button>
              ))}
            </div>

            <Button
              onClick={() => setCurrentBatch(currentBatch + 1)}
              disabled={currentBatch >= prompts.length - 1}
              variant="ghost"
              size="sm"
              className="h-7 px-2"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </Button>
          </div>

          {/* Current batch prompt */}
          <div className="border border-border rounded">
            <div className="flex items-center justify-between px-3 py-1.5 bg-secondary/50 border-b border-border">
              <span className="text-xs text-muted-foreground">
                Batch {currentBatch + 1} — {batch.segment_count} segments
                {batchDone && <span className="text-success ml-2">validated</span>}
              </span>
              <Button
                onClick={() => copyToClipboard(batch.prompt, batch.batch_index)}
                variant="ghost"
                size="sm"
                className="h-6 px-2"
              >
                {copiedIdx === batch.batch_index ? (
                  <Check className="w-3 h-3 text-success" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
              </Button>
            </div>
            <pre className="p-3 text-xs max-h-60 overflow-y-auto whitespace-pre-wrap leading-relaxed">
              {batch.prompt}
            </pre>
          </div>

          {/* Paste + validate for this batch */}
          {!batchDone ? (
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground block">
                Paste LLM response for batch {currentBatch + 1}:
              </label>
              <textarea
                value={pastedText}
                onChange={(e) => { setPastedText(e.target.value); setParseError(null); }}
                placeholder='[{"speaker": "...", "text": "...", "start": 0, "end": 0}, ...]'
                className="input text-xs w-full resize-y"
                rows={6}
              />
              {parseError && <p className="text-destructive text-xs">{parseError}</p>}
              <Button
                onClick={validateBatch}
                disabled={!pastedText.trim()}
                size="sm"
              >
                Validate batch {currentBatch + 1}
              </Button>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs">
              <Check className="w-3.5 h-3.5 text-success" />
              <span className="text-success">Batch {currentBatch + 1} validated ({(batchResults[currentBatch] as unknown[]).length} segments)</span>
              <Button
                onClick={() => {
                  const next = { ...batchResults };
                  delete next[currentBatch];
                  setBatchResults(next);
                }}
                variant="ghost"
                size="sm"
                className="h-6 text-xs text-muted-foreground"
              >
                Redo
              </Button>
            </div>
          )}

          {/* Apply all / Cancel */}
          <div className="flex gap-2 border-t border-border/50 pt-3">
            <Button
              onClick={() => applyMutation.mutate()}
              disabled={!allDone || applyMutation.isPending}
              size="sm"
            >
              {applyMutation.isPending ? "Applying..." : `Apply all ${prompts.length} batches`}
            </Button>
            <Button
              onClick={() => { setPrompts(null); setBatchResults({}); setPastedText(""); }}
              variant="ghost"
              size="sm"
            >
              Cancel
            </Button>
          </div>
          {applyMutation.isError && (
            <p className="text-destructive text-xs">{errorMessage(applyMutation.error)}</p>
          )}
        </div>
      )}
    </div>
  );
}

/** Find the next batch index that hasn't been validated yet. */
function findNextUnfinished(
  current: number,
  total: number,
  results: Record<number, unknown[]>,
): number | null {
  // Look forward first
  for (let i = current + 1; i < total; i++) {
    if (results[i] == null) return i;
  }
  // Then wrap around
  for (let i = 0; i < current; i++) {
    if (results[i] == null) return i;
  }
  return null;
}
