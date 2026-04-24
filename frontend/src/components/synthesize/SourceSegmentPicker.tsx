/**
 * Source picker for synthesis — same "Source text" dropdown pattern as the
 * Translate / Correct panels, plus a compact per-segment list.
 *
 * Checkboxes default to ON (everything kept). Unchecking a row drops that
 * segment from the synthesis scope: it is NOT used for voice sampling and
 * NOT included in the generated output. Shift-click a checkbox to apply
 * the same state to the range between this click and the previous one.
 * Clicking a row body (outside the checkbox) expands/collapses long text.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Play } from "lucide-react";
import type { Episode, Segment, VersionEntry } from "@/api/types";
import {
  getAllVersions,
  getCorrectSegments,
  getSegments,
  getTranslateSegments,
  loadCorrectVersion,
  loadTranscribeVersion,
  loadTranslateVersion,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import SectionHeader from "@/components/common/SectionHeader";
import { formatTime, selectClass, versionOption } from "@/lib/utils";
import { segKey } from "@/lib/segKey";
import { speakerColor } from "@/lib/speakerColor";

/** Module-level stable reference so the onSegmentsChange effect doesn't fire
 *  a fresh `[]` literal on every render while the query is still loading. */
const EMPTY_SEGMENTS: Segment[] = [];

/** Resolved source info: which step the selected version belongs to, plus
 *  the payload fields the generate endpoint needs. */
export interface ResolvedSource {
  step: "transcript" | "corrected" | "translate";
  lang: string;
  sourceLang: string | undefined;
  sourceVersionId: string | null;
}

export interface SourceSegmentPickerProps {
  audioPath: string;
  episode: Episode;

  sourceVersionId: string | null;
  setSourceVersionId: (v: string | null) => void;

  onResolvedSourceChange: (resolved: ResolvedSource) => void;
  onSegmentsChange: (segments: Segment[]) => void;

  selectedKeys: Set<string>;
  setSelectedKeys: (s: Set<string>) => void;

  seekTo: (path: string, time: number) => void;
}

export default function SourceSegmentPicker({
  audioPath,
  episode,
  sourceVersionId,
  setSourceVersionId,
  onResolvedSourceChange,
  onSegmentsChange,
  selectedKeys,
  setSelectedKeys,
  seekTo,
}: SourceSegmentPickerProps) {
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [listOpen, setListOpen] = useState(false);

  const { data: allVersions } = useQuery({
    queryKey: queryKeys.allVersions(audioPath),
    queryFn: () => getAllVersions(audioPath),
    enabled: !!audioPath,
  });

  const translationSet = useMemo(
    () => new Set(episode.translations),
    [episode.translations],
  );

  const inputVersions = useMemo<VersionEntry[]>(() => {
    if (!allVersions) return [];
    return allVersions.filter((v) => {
      if (!v.step) return false;
      if (v.step === "transcript" || v.step === "corrected") return true;
      return translationSet.has(v.step);
    });
  }, [allVersions, translationSet]);

  const selectedVersion = useMemo<VersionEntry | null>(() => {
    if (!inputVersions.length) return null;
    if (sourceVersionId) {
      return inputVersions.find((v) => v.id === sourceVersionId) ?? inputVersions[0];
    }
    return inputVersions[0];
  }, [inputVersions, sourceVersionId]);

  const resolved = useMemo<ResolvedSource>(() => {
    if (!selectedVersion) {
      return { step: "transcript", lang: "", sourceLang: undefined, sourceVersionId: null };
    }
    const step = selectedVersion.step ?? "transcript";
    if (step === "transcript" || step === "corrected") {
      return { step, lang: "", sourceLang: undefined, sourceVersionId };
    }
    // Translation versions store the language as the step name.
    return { step: "translate", lang: step, sourceLang: step, sourceVersionId };
  }, [selectedVersion, sourceVersionId]);

  useEffect(() => {
    onResolvedSourceChange(resolved);
  }, [resolved, onResolvedSourceChange]);

  // Editor key matches the per-step editor pages so React Query dedupes
  // with Transcribe / Correct / Translate panels when the user later opens
  // one of them.
  const editorKey =
    resolved.step === "transcript"
      ? "transcribe"
      : resolved.step === "corrected"
        ? "correct"
        : `translate-${resolved.lang}`;

  // Segment cache is shared with the editors: stepVersionSegments for a
  // pinned version, stepSegments for "latest".
  const segmentsQueryKey = sourceVersionId
    ? queryKeys.stepVersionSegments(editorKey, audioPath, sourceVersionId)
    : queryKeys.stepSegments(editorKey, audioPath);

  const { data: segments } = useQuery({
    queryKey: segmentsQueryKey,
    queryFn: async (): Promise<Segment[]> => {
      // Load the pinned version when provided; otherwise the step's latest.
      if (sourceVersionId) {
        if (resolved.step === "transcript")
          return loadTranscribeVersion(audioPath, sourceVersionId);
        if (resolved.step === "corrected")
          return loadCorrectVersion(audioPath, sourceVersionId);
        return loadTranslateVersion(audioPath, resolved.lang, sourceVersionId);
      }
      if (resolved.step === "transcript") return getSegments(audioPath);
      if (resolved.step === "corrected") return getCorrectSegments(audioPath);
      return getTranslateSegments(audioPath, resolved.lang);
    },
    enabled: !!selectedVersion,
  });

  useEffect(() => {
    onSegmentsChange(segments ?? EMPTY_SEGMENTS);
  }, [segments, onSegmentsChange]);

  // Default-all-selected whenever the underlying source changes. Stamp
  // combines editorKey, version id, and segment count so a fresh source
  // resets scope to "all kept" without clobbering the user's unchecks on
  // a same-source revalidation.
  const lastInitStamp = useRef<string>("");
  useEffect(() => {
    if (!segments) return;
    const stamp = `${editorKey}|${sourceVersionId ?? "latest"}|${segments.length}`;
    if (lastInitStamp.current === stamp) return;
    lastInitStamp.current = stamp;
    const all = new Set<string>();
    for (const s of segments) {
      if (s.speaker === "[BREAK]") continue;
      all.add(segKey(s));
    }
    setSelectedKeys(all);
  }, [segments, editorKey, sourceVersionId, setSelectedKeys]);

  const visibleSegments = useMemo(
    () => (segments ?? []).filter((s) => s.speaker && s.speaker !== "[BREAK]"),
    [segments],
  );

  // Records the most recent plain toggle — origin of a shift-click range.
  const lastToggleRef = useRef<{ index: number; checked: boolean } | null>(null);
  // onMouseDown fires before onChange and is where we can observe the
  // shift-key state. onChange carries the authoritative checked value for
  // both mouse and keyboard, so toggles stay accessible via spacebar.
  const shiftDownRef = useRef(false);

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

      {inputVersions.length > 0 ? (
        <select
          value={sourceVersionId ?? ""}
          onChange={(e) => setSourceVersionId(e.target.value || null)}
          title="Which version of the episode the cloned voices will read aloud."
          className={`${selectClass} text-xs max-w-full min-w-0`}
        >
          <option value="">Latest, {versionOption(inputVersions[0])}</option>
          {inputVersions.map((v) => (
            <option key={v.id} value={v.id}>
              {versionOption(v)}
            </option>
          ))}
        </select>
      ) : (
        <p className="text-xs text-muted-foreground italic">
          No transcript, correction, or translation versions available yet.
        </p>
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
                    {segments ? "No segments in this version." : "Loading…"}
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
                        onMouseDown={(e) => {
                          shiftDownRef.current = e.shiftKey;
                        }}
                        onKeyDown={() => {
                          shiftDownRef.current = false;
                        }}
                        onChange={(e) => {
                          applyToggle(index, e.target.checked, shiftDownRef.current);
                          shiftDownRef.current = false;
                        }}
                        className="w-3 h-3 accent-primary cursor-pointer"
                        aria-label={isKept ? "Drop this segment" : "Keep this segment"}
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          seekTo(audioPath, seg.start);
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
