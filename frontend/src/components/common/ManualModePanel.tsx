import { useState, useMemo, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { errorMessage } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Copy, Check, ChevronLeft, ChevronRight, Search, Trash2, Plus } from "lucide-react";

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
  const [reconcileOpen, setReconcileOpen] = useState(false);

  const generateMutation = useMutation({
    mutationFn: () => generatePrompts(batchMinutes),
    onSuccess: (data) => {
      setPrompts(data);
      setCurrentBatch(0);
      setBatchResults({});
      setPastedText("");
      setReconcileOpen(false);
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
      const expected = prompts?.[currentBatch]?.segment_count;
      // Per-batch count check catches LLM drift before the whole apply runs.
      // Without this, one off-by-one batch would cause the backend to reject
      // all corrections as an index-drift mismatch.
      if (expected != null && arr.length !== expected) {
        const startIndex = (prompts ?? [])
          .slice(0, currentBatch)
          .reduce((s, p) => s + p.segment_count, 0);
        const drift = describeDrift(arr, prompts?.[currentBatch]?.prompt ?? "", startIndex);
        setParseError(
          `Batch ${currentBatch + 1} expects ${expected} entries, got ${arr.length}. ` +
          (drift ? drift + " " : "") +
          "Regenerate this batch or trim the response so the counts match.",
        );
        return;
      }
      setBatchResults({ ...batchResults, [currentBatch]: arr });
      setPastedText("");
      setReconcileOpen(false);
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

      <Button
        onClick={() => generateMutation.mutate()}
        disabled={generateMutation.isPending}
        size="sm"
      >
        {generateMutation.isPending ? "Generating..." : prompts ? "Regenerate prompts" : "Generate prompts"}
      </Button>
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
                  onClick={() => { setCurrentBatch(i); setPastedText(""); setParseError(null); setReconcileOpen(false); }}
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
                Batch {currentBatch + 1}: {batch.segment_count} segments
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
            reconcileOpen ? (
              <ReconcileView
                prompt={batch.prompt}
                expected={batch.segment_count}
                value={pastedText}
                onChange={setPastedText}
                error={parseError}
                onRevalidate={validateBatch}
                onClose={() => setReconcileOpen(false)}
              />
            ) : (
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
                {parseError && (
                  <div className="flex items-start gap-2">
                    <p className="text-destructive text-xs flex-1">{parseError}</p>
                    <Button
                      onClick={() => {
                        // Pretty-print so the reconcile view's raw mode reads
                        // with one entry per line.
                        const objs = parseResponseObjects(pastedText);
                        if (objs) setPastedText(JSON.stringify(objs, null, 2));
                        setReconcileOpen(true);
                      }}
                      variant="outline"
                      size="sm"
                      className="h-6 px-2 text-xs shrink-0"
                    >
                      Fix by hand
                    </Button>
                  </div>
                )}
                <Button
                  onClick={validateBatch}
                  disabled={!pastedText.trim()}
                  size="sm"
                >
                  Validate batch {currentBatch + 1}
                </Button>
              </div>
            )
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
              {applyMutation.isPending
                ? "Applying..."
                : prompts.length === 1
                  ? "Apply corrections"
                  : `Apply all ${prompts.length} batches`}
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

/** Parse a pasted JSON response into entry objects, or null when it does not
 *  parse. Bare strings become `{text}`; other fields are preserved. */
function parseResponseObjects(text: string): Record<string, unknown>[] | null {
  try {
    const parsed = JSON.parse(text);
    const arr = Array.isArray(parsed) ? parsed : [parsed];
    return arr.map((x) =>
      typeof x === "string" ? { text: x } : ((x as Record<string, unknown>) ?? {}),
    );
  } catch {
    return null;
  }
}

function entryText(obj: Record<string, unknown>): string {
  return String(obj.text ?? "");
}

/** Side-by-side reconcile view: the batch's input segments next to the
 *  pasted response. The response is editable per-entry (numbered, aligned
 *  with the input) so the user can hand-fix a malformed LLM response, with a
 *  raw-JSON mode for bulk edits and a find box to jump within long batches. */
function ReconcileView({
  prompt,
  expected,
  value,
  onChange,
  error,
  onRevalidate,
  onClose,
}: {
  prompt: string;
  expected: number;
  value: string;
  onChange: (text: string) => void;
  error: string | null;
  onRevalidate: () => void;
  onClose: () => void;
}) {
  const [rawMode, setRawMode] = useState(false);
  const [query, setQuery] = useState("");
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);

  const inputEntries = useMemo(
    () => [...parsePromptEntries(prompt).entries()].sort((a, b) => a[0] - b[0]),
    [prompt],
  );
  const firstIdx = inputEntries[0]?.[0] ?? 0;
  const respObjs = useMemo(() => parseResponseObjects(value), [value]);
  const got = respObjs?.length ?? null;

  // First response row that diverges from the input. Exact when the LLM
  // echoed `index` fields; otherwise the first surplus row. -1 = no drift
  // (or a deletion, which cannot be pinpointed).
  const driftFrom = useMemo(() => {
    if (!respObjs) return -1;
    for (let i = 0; i < respObjs.length; i++) {
      const idx = respObjs[i].index;
      if (typeof idx === "number" && idx !== firstIdx + i) return i;
    }
    return respObjs.length > inputEntries.length ? inputEntries.length : -1;
  }, [respObjs, firstIdx, inputEntries.length]);

  const writeObjs = (objs: Record<string, unknown>[]) =>
    onChange(JSON.stringify(objs, null, 2));
  const updateEntry = (i: number, text: string) => {
    if (respObjs) writeObjs(respObjs.map((o, j) => (j === i ? { ...o, text } : o)));
  };
  const deleteEntry = (i: number) => {
    if (respObjs) writeObjs(respObjs.filter((_, j) => j !== i));
  };
  const insertEntry = (i: number) => {
    const objs = respObjs ?? [];
    writeObjs([...objs.slice(0, i + 1), { text: "" }, ...objs.slice(i + 1)]);
  };
  const addEntry = () => writeObjs([...(respObjs ?? []), { text: "" }]);

  const runSearch = () => {
    const q = query.trim().toLowerCase();
    if (!q) return;
    const li = inputEntries.findIndex(([, t]) => t.toLowerCase().includes(q));
    const ri = respObjs?.findIndex((o) => entryText(o).toLowerCase().includes(q)) ?? -1;
    const opts: ScrollIntoViewOptions = { block: "center" };
    (leftRef.current?.children[li >= 0 ? li : ri] as HTMLElement | undefined)?.scrollIntoView(opts);
    (rightRef.current?.children[ri >= 0 ? ri : li] as HTMLElement | undefined)?.scrollIntoView(opts);
  };

  return (
    <div className="space-y-2 border border-border rounded p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium shrink-0">Compare &amp; fix by hand</span>
        <div className="flex items-center gap-1 flex-1 max-w-xs">
          <Search className="w-3 h-3 text-muted-foreground shrink-0" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") runSearch(); }}
            placeholder="Find text, Enter to jump..."
            className="input text-xs h-6 w-full"
          />
        </div>
        <Button onClick={onClose} variant="ghost" size="sm" className="h-6 px-2 text-xs shrink-0">
          Close
        </Button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="border border-border rounded overflow-hidden">
          <div className="px-2 py-1 bg-secondary/50 border-b border-border text-2xs text-muted-foreground">
            Input — {expected} segments
          </div>
          <div ref={leftRef} className="p-2 text-xs max-h-72 overflow-y-auto space-y-1 leading-relaxed">
            {inputEntries.map(([idx, text]) => (
              <div key={idx}>
                <span className="text-muted-foreground/60 font-mono mr-1">[{idx}]</span>
                {text}
              </div>
            ))}
          </div>
        </div>
        <div className="border border-border rounded overflow-hidden flex flex-col">
          <div className="px-2 py-1 bg-secondary/50 border-b border-border text-2xs flex items-center gap-2">
            <span className="text-muted-foreground">Your response</span>
            <div className="flex gap-0.5">
              {([["Formatted", false], ["Raw", true]] as const).map(([label, raw]) => (
                <button
                  key={label}
                  onClick={() => setRawMode(raw)}
                  className={`px-1.5 rounded ${
                    rawMode === raw
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            <span className={`ml-auto ${got === expected ? "text-success" : "text-destructive"}`}>
              {got == null ? "invalid JSON" : `${got} / ${expected} entries`}
            </span>
          </div>
          {rawMode ? (
            <textarea
              value={value}
              onChange={(e) => onChange(e.target.value)}
              className="input text-xs w-full resize-y rounded-none border-0 flex-1"
              rows={14}
            />
          ) : respObjs == null ? (
            <p className="p-2 text-destructive text-2xs">
              Response is not valid JSON — switch to Raw to fix it.
            </p>
          ) : (
            <div ref={rightRef} className="p-2 max-h-72 overflow-y-auto space-y-1">
              {respObjs.map((obj, i) => {
                const drift = driftFrom >= 0 && i >= driftFrom;
                return (
                  <div
                    key={i}
                    className={`flex items-start gap-1 ${drift ? "bg-destructive/10 rounded" : ""}`}
                  >
                    <span className="text-muted-foreground/60 font-mono text-xs mt-1 shrink-0">
                      [{firstIdx + i}]
                    </span>
                    <textarea
                      value={entryText(obj)}
                      onChange={(e) => updateEntry(i, e.target.value)}
                      rows={1}
                      className="input text-xs flex-1 resize-none overflow-hidden [field-sizing:content] py-0.5"
                    />
                    <div className="flex flex-col gap-0.5 mt-1 shrink-0">
                      <button
                        onClick={() => insertEntry(i)}
                        className="text-muted-foreground/50 hover:text-foreground"
                        title="Insert an entry below"
                      >
                        <Plus className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => deleteEntry(i)}
                        className="text-muted-foreground/50 hover:text-destructive"
                        title="Delete this entry"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                );
              })}
              <button
                onClick={addEntry}
                className="flex items-center gap-1 text-2xs text-muted-foreground hover:text-foreground pt-1"
              >
                <Plus className="w-3 h-3" /> Add entry
              </button>
            </div>
          )}
        </div>
      </div>
      {error && <p className="text-destructive text-xs">{error}</p>}
      <Button onClick={onRevalidate} disabled={!value.trim()} size="sm">
        Re-validate batch
      </Button>
    </div>
  );
}

/** Pull expected `[N] text` lines out of a prompt. Ignores trailing
 *  instruction block. Returns map {absIndex: text}. */
function parsePromptEntries(prompt: string): Map<number, string> {
  const map = new Map<number, string>();
  const re = /^\[(\d+)\]\s+(.+)$/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(prompt)) !== null) {
    map.set(Number(m[1]), m[2]);
  }
  return map;
}

/** Locate the first point where the LLM response diverges from the prompt.
 *  Three signals, in order of reliability:
 *    1. `index` field on entries (LLM often echoes the prompt's [N] markers).
 *    2. Text similarity to the corresponding prompt entry.
 *    3. Fallback: surface the first and last few entries so the user can
 *       eyeball where the list got too long or short. */
function describeDrift(
  arr: unknown[],
  prompt: string,
  startIndex: number,
): string | null {
  const expected = parsePromptEntries(prompt);
  const preview = (s: unknown, n = 50) =>
    String(s ?? "").replace(/\s+/g, " ").slice(0, n);

  // Signal 1: explicit index field drift
  for (let i = 0; i < arr.length; i++) {
    const item = arr[i] as { index?: unknown; text?: unknown } | null;
    if (item && typeof item.index === "number") {
      const want = i + startIndex;
      if (item.index !== want) {
        const expText = expected.get(want) ?? "(prompt entry missing)";
        return (
          `Drift begins around segment [${want}]: expected ` +
          `"${preview(expText)}…", got index=${item.index} ` +
          `("${preview(item.text)}…").`
        );
      }
    }
  }

  // Signal 2: text similarity (Jaccard on first ~40 chars)
  if (expected.size > 0) {
    for (let i = 0; i < arr.length; i++) {
      const expText = expected.get(i + startIndex);
      if (!expText) continue;
      const gotText = (arr[i] as { text?: unknown })?.text;
      if (typeof gotText !== "string") continue;
      if (!looksRelated(expText, gotText)) {
        return (
          `Drift begins around segment [${i + startIndex}]: prompt had ` +
          `"${preview(expText)}…", response has "${preview(gotText)}…".`
        );
      }
    }
  }

  // Signal 3: count-only difference — show head/tail so user can scan
  const diff = arr.length - expected.size;
  if (diff !== 0 && arr.length > 0) {
    const head = arr
      .slice(0, 2)
      .map((x, i) => `[${startIndex + i}]: "${preview((x as { text?: unknown })?.text)}…"`);
    const tail = arr
      .slice(-2)
      .map(
        (x, i) =>
          `[${startIndex + arr.length - 2 + i}]: ` +
          `"${preview((x as { text?: unknown })?.text)}…"`,
      );
    return `No index field to pinpoint drift. Head: ${head.join(", ")}. Tail: ${tail.join(", ")}.`;
  }
  return null;
}

/** Cheap "same-ish line" check. Normalises punctuation and diacritics
 *  before tokenising so LLM corrections like "moi" → "muy" or "Si" → "Sí"
 *  still count as related. Low threshold (0.2) keeps short segments from
 *  producing false positives — the goal is only to catch clearly different
 *  lines, not to verify wording. */
function looksRelated(a: string, b: string): boolean {
  const toks = (s: string) =>
    new Set(
      s
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "") // strip combining diacritics
        .toLowerCase()
        .replace(/[^\p{L}\p{N}\s]/gu, " ") // strip punctuation
        .split(/\s+/)
        .slice(0, 10)
        .filter((w) => w.length > 1),
    );
  const A = toks(a);
  const B = toks(b);
  if (A.size === 0 || B.size === 0) return true;
  let inter = 0;
  for (const w of A) if (B.has(w)) inter++;
  return inter / Math.min(A.size, B.size) >= 0.2;
}
