/** Side-by-side reconcile editor: a batch's input segments next to the LLM
 *  response. The response is editable per-entry (numbered, aligned with the
 *  input) so a malformed response can be hand-fixed, with a raw-JSON mode for
 *  bulk edits and a find box to jump within long batches.
 *
 *  Presentational — the parent owns validation and apply (see
 *  BatchReconcileModal and ManualModePanel). */

import { useState, useMemo, useRef, useLayoutEffect } from "react";
import { Search, Trash2, Plus } from "lucide-react";
import { parseResponseObjects, entryText, type InputEntry } from "@/lib/reconcile";

/** Textarea that grows to fit its content. The webview does not reliably
 *  support `field-sizing: content`, so height is synced from scrollHeight. */
function AutoGrowTextarea({
  value,
  onChange,
  className,
}: {
  value: string;
  onChange: (text: string) => void;
  className?: string;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const resize = () => {
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight}px`;
    };
    resize();
    // Width changes (window/panel resize) rewrap the text — re-sync height.
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [value]);
  return (
    <textarea
      ref={ref}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      rows={1}
      className={className}
    />
  );
}

export default function ReconcileView({
  inputEntries,
  value,
  onChange,
}: {
  inputEntries: InputEntry[];
  value: string;
  onChange: (text: string) => void;
}) {
  const [rawMode, setRawMode] = useState(false);
  const [query, setQuery] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  const expected = inputEntries.length;
  const firstIdx = inputEntries[0]?.index ?? 0;
  const respObjs = useMemo(() => parseResponseObjects(value), [value]);
  const got = respObjs?.length ?? null;
  const rowCount = Math.max(inputEntries.length, respObjs?.length ?? 0);

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
    for (let i = 0; i < rowCount; i++) {
      const inText = inputEntries[i]?.text.toLowerCase() ?? "";
      const respObj = respObjs?.[i];
      const respText = respObj ? entryText(respObj).toLowerCase() : "";
      if (inText.includes(q) || respText.includes(q)) {
        (listRef.current?.children[i] as HTMLElement | undefined)
          ?.scrollIntoView({ block: "center" });
        return;
      }
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium shrink-0">Compare &amp; fix by hand</span>
        <div className="flex items-center gap-1 flex-1 max-w-xs">
          <Search className="w-3 h-3 text-muted-foreground shrink-0" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") runSearch(); }}
            placeholder="Find…"
            className="input text-xs h-6 w-full"
          />
        </div>
      </div>

      <div className="border border-border rounded overflow-hidden">
        {/* Column headers — input left, response right, aligned with the rows */}
        <div className="flex items-center gap-2 px-2 py-1 bg-secondary/50 border-b border-border text-2xs">
          <span className="w-9 shrink-0" aria-hidden />
          <span className="flex-1 min-w-0 text-muted-foreground">Input ({expected} segments)</span>
          <span className="flex-1 min-w-0 text-muted-foreground">Your response</span>
          <div className="flex gap-0.5 shrink-0">
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
          <span className={`shrink-0 ${got === expected ? "text-success" : "text-destructive"}`}>
            {got == null ? "invalid JSON" : `${got} / ${expected}`}
          </span>
          <span className="w-4 shrink-0" aria-hidden />
        </div>

        {rawMode ? (
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            className="input text-xs w-full resize-y rounded-none border-0"
            rows={20}
          />
        ) : respObjs == null ? (
          <p className="p-2 text-destructive text-2xs">
            Response is not valid JSON. Switch to Raw to fix it.
          </p>
        ) : (
          <div ref={listRef} className="max-h-[60vh] overflow-y-auto">
            {Array.from({ length: rowCount }, (_, i) => {
              const input = inputEntries[i];
              const resp = respObjs[i];
              const hasResp = resp !== undefined;
              const drift = driftFrom >= 0 && i >= driftFrom;
              return (
                <div
                  key={i}
                  className={`flex items-start gap-2 px-2 py-1 border-b border-border/30 last:border-b-0 ${
                    drift ? "bg-destructive/10" : ""
                  }`}
                >
                  <span className="w-9 shrink-0 pt-1.5 text-right font-mono text-2xs text-muted-foreground/60">
                    [{input ? input.index : firstIdx + i}]
                  </span>
                  <div className="flex-1 min-w-0 pt-1 text-xs leading-relaxed break-words">
                    {input
                      ? input.text
                      : <span className="italic text-muted-foreground/40">no matching input</span>}
                  </div>
                  {hasResp ? (
                    <AutoGrowTextarea
                      value={entryText(resp)}
                      onChange={(text) => updateEntry(i, text)}
                      className="input text-xs flex-1 min-w-0 resize-none overflow-hidden py-0.5"
                    />
                  ) : (
                    <div className="flex-1 min-w-0 pt-1 text-xs italic text-muted-foreground/40">
                      missing (use Add entry below)
                    </div>
                  )}
                  <div className="flex flex-col gap-0.5 shrink-0 w-4 pt-1">
                    {hasResp && (
                      <>
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
                      </>
                    )}
                  </div>
                </div>
              );
            })}
            <button
              onClick={addEntry}
              className="flex items-center gap-1 px-2 py-1.5 text-2xs text-muted-foreground hover:text-foreground"
            >
              <Plus className="w-3 h-3" /> Add entry
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
