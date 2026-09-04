/**
 * Source picker for synthesis — same "Source text" dropdown pattern as the
 * Translate / Correct panels, plus a compact per-segment list.
 *
 * Presentational: the versions and segments it shows are fetched by
 * `useSynthSource` in the panel above and handed down, so this component
 * owns only the list's own UI state. It used to run both queries itself and
 * copy the results up through effects, which left the panel a render behind
 * and gave the shared data no single owner.
 *
 * Checkboxes default to ON (everything kept). Unchecking a row drops that
 * segment from the synthesis scope: it is NOT used for voice sampling and
 * NOT included in the generated output. Shift-click a checkbox to apply
 * the same state to the range between this click and the previous one.
 * Clicking a row body (outside the checkbox) expands/collapses long text.
 */

import { useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Play } from "lucide-react";
import SectionHeader from "@/components/common/SectionHeader";
import VersionPicker from "@/components/common/VersionPicker";
import { ErrorAlert } from "@/components/ui/error-alert";
import { formatTime } from "@/lib/utils";
import { segKey } from "@/lib/segKey";
import { speakerColor } from "@/lib/speakerColor";
import { BREAK_SPEAKER } from "@/lib/speakers";
import type { SynthSource } from "./useSynthSource";

export interface SourceSegmentPickerProps {
  audioPath: string | null;
  /** Everything fetched for the chosen source; see `useSynthSource`. */
  source: SynthSource;

  sourceVersionId: string | null;
  setSourceVersionId: (v: string | null) => void;

  selectedKeys: Set<string>;
  setSelectedKeys: (s: Set<string>) => void;

  seekTo: (path: string, time: number) => void;
}

export default function SourceSegmentPicker({
  audioPath,
  source,
  sourceVersionId,
  setSourceVersionId,
  selectedKeys,
  setSelectedKeys,
  seekTo,
}: SourceSegmentPickerProps) {
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [listOpen, setListOpen] = useState(false);

  const { inputVersions, selectedVersion, segments } = source;

  // Same predicate the panel's default-seed loop uses: non-BREAK rows are
  // candidate segments. Empty-speaker rows count (narrator fallback handles
  // them at synth time); excluding them here would silently drop their keys
  // from selectAll/None/invert despite being seeded into selectedKeys.
  const visibleSegments = useMemo(
    () => segments.filter((s) => s.speaker !== BREAK_SPEAKER),
    [segments],
  );

  // Records the most recent plain toggle — origin of a shift-click range.
  const lastToggleRef = useRef<{ index: number; checked: boolean } | null>(null);

  const applyToggle = (index: number, nextChecked: boolean, shiftHeld: boolean) => {
    if (shiftHeld && lastToggleRef.current) {
      const start = Math.min(lastToggleRef.current.index, index);
      const end = Math.max(lastToggleRef.current.index, index);
      const applyChecked = lastToggleRef.current.checked;
      const next = new Set(selectedKeys);
      for (let i = start; i <= end; i++) {
        const k = segKey(visibleSegments[i]);
        if (applyChecked) next.add(k);
        else next.delete(k);
      }
      setSelectedKeys(next);
      return;
    }
    const key = segKey(visibleSegments[index]);
    const next = new Set(selectedKeys);
    if (nextChecked) next.add(key);
    else next.delete(key);
    setSelectedKeys(next);
    lastToggleRef.current = { index, checked: nextChecked };
  };

  const selectAll = () => {
    const next = new Set<string>();
    for (const s of visibleSegments) next.add(segKey(s));
    setSelectedKeys(next);
  };
  const selectNone = () => setSelectedKeys(new Set());
  const invertSelection = () => {
    const next = new Set<string>();
    for (const s of visibleSegments) {
      const k = segKey(s);
      if (!selectedKeys.has(k)) next.add(k);
    }
    setSelectedKeys(next);
  };

  const keptCount = useMemo(() => {
    let n = 0;
    for (const s of visibleSegments) {
      if (selectedKeys.has(segKey(s))) n++;
    }
    return n;
  }, [visibleSegments, selectedKeys]);

  const allKept = keptCount === visibleSegments.length && visibleSegments.length > 0;

  return (
    <section className="space-y-3">
      <SectionHeader help="Pick the text the cloned voices will read aloud. Optionally narrow to specific segments.">
        1. Source
      </SectionHeader>

      {/* A failed load must not read as "nothing to synthesize yet": the
          version list's empty message and the segment list's "Loading…"
          placeholder both look like normal states. */}
      {source.versions.isError ? (
        <ErrorAlert error={source.versions.error} onRetry={source.versions.refetch} />
      ) : (
        <VersionPicker
          versions={inputVersions}
          value={sourceVersionId}
          onChange={setSourceVersionId}
          title="Which version of the episode the cloned voices will read aloud."
          emptyMessage="No transcript, correction, or translation versions available yet."
        />
      )}

      {selectedVersion && source.segmentsQuery.isError && (
        <ErrorAlert
          error={source.segmentsQuery.error}
          onRetry={source.segmentsQuery.refetch}
        />
      )}

      {selectedVersion && (
        <div className="space-y-1.5">
          <button
            type="button"
            onClick={() => setListOpen(!listOpen)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
          >
            {listOpen ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
            <span className="font-medium">
              {listOpen ? "Hide segment scope" : "Narrow segment scope"}
            </span>
            <span className="text-muted-foreground/60">
              · {keptCount} of {visibleSegments.length} kept
              {keptCount < visibleSegments.length && " · unchecked dropped"}
            </span>
          </button>

          {listOpen && (
            <div className="space-y-1.5 pl-5">
              <div className="flex items-center gap-2 text-xs text-muted-foreground flex-wrap">
                <button
                  type="button"
                  onClick={() => (allKept ? selectNone() : selectAll())}
                  className="hover:text-foreground transition"
                >
                  {allKept ? "None" : "All"}
                </button>
                <button
                  type="button"
                  onClick={invertSelection}
                  className="hover:text-foreground transition"
                >
                  Invert
                </button>
                <span className="text-muted-foreground/50">Shift-click for range</span>
              </div>

              <div className="max-h-80 overflow-y-auto border border-border/60 rounded-md divide-y divide-border/30 bg-background/40">
                {visibleSegments.length === 0 && (
                  <p className="p-3 text-xs text-muted-foreground italic">
                    {source.segmentsQuery.isError
                      ? "Could not load this version."
                      : source.segmentsQuery.isPending
                        ? "Loading…"
                        : "No segments in this version."}
                  </p>
                )}
                {visibleSegments.map((seg, index) => {
                  const key = segKey(seg);
                  const isKept = selectedKeys.has(key);
                  const isExpanded = expandedKey === key;
                  const speaker = seg.speaker || "";
                  return (
                    <div
                      key={key}
                      onClick={() => setExpandedKey(isExpanded ? null : key)}
                      className={`flex items-center gap-2 px-2 py-1 text-xs cursor-pointer transition ${
                        isKept
                          ? "hover:bg-secondary/70"
                          : "opacity-50 hover:bg-secondary/70 hover:opacity-80"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isKept}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => {
                          // nativeEvent carries shiftKey for both mouse
                          // (MouseEvent) and keyboard (KeyboardEvent) toggles.
                          const native = e.nativeEvent as MouseEvent | KeyboardEvent;
                          applyToggle(index, e.target.checked, !!native.shiftKey);
                        }}
                        className="w-3 h-3 accent-primary cursor-pointer"
                        aria-label={isKept ? "Drop this segment" : "Keep this segment"}
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (audioPath) seekTo(audioPath, seg.start);
                        }}
                        className="shrink-0 p-0.5 rounded hover:bg-accent transition"
                        title="Play from here"
                        aria-label={`Play segment at ${seg.start.toFixed(1)}s`}
                      >
                        <Play className="w-3 h-3" />
                      </button>
                      <span className="text-muted-foreground tabular-nums shrink-0 w-28">
                        {formatTime(seg.start, false)}–{formatTime(seg.end, false)}
                      </span>
                      <span
                        className="shrink-0 w-16 truncate font-medium"
                        style={{ color: speakerColor(speaker) }}
                        title={speaker}
                      >
                        {speaker}
                      </span>
                      <span
                        className={`flex-1 text-foreground/90 ${isExpanded ? "whitespace-normal" : "truncate"}`}
                      >
                        {seg.text}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
