/**
 * TranscriptViewer — unified editable transcript.
 *
 * Combines the clean read-only layout (timestamp | speaker | text per row) with
 * inline editing. Every segment is directly editable; there is no separate
 * read/edit mode toggle.
 */

import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Segment, VersionEntry } from "@/api/types";
import { exportTextUrl, exportSrtUrl, exportVttUrl } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { useAudioStore } from "@/stores";
import { useSegments } from "@/hooks/useSegments";
import { useSegmentFiltering, useFilteredSegments, flagReason } from "@/hooks/useSegmentFiltering";
import { formatTime, versionDate, versionLabel, versionInfo, selectClass } from "@/lib/utils";
import { computeWordDiff } from "@/lib/diffUtils";
import { Button } from "@/components/ui/button";
import Pagination from "./Pagination";
import {
  Download,
  Save,
  Undo2,
  Play,
  Trash2,
  Merge,
  Scissors,
  ChevronDown,
  ChevronRight,
  Search,
  X,
  Diff,
  AlertTriangle,
  Activity,
} from "lucide-react";

// ── SVG icons for insert-before / insert-after ────────────────────────────────

function InsertBeforeIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <line x1="4" y1="4" x2="12" y2="4" />
      <line x1="8" y1="8" x2="8" y2="14" />
      <line x1="5" y1="11" x2="11" y2="11" />
      <polyline points="5.5,10 8,7.5 10.5,10" />
    </svg>
  );
}

function InsertAfterIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <line x1="4" y1="12" x2="12" y2="12" />
      <line x1="8" y1="2" x2="8" y2="8" />
      <line x1="5" y1="5" x2="11" y2="5" />
      <polyline points="5.5,6 8,8.5 10.5,6" />
    </svg>
  );
}

// ── Export dropdown ───────────────────────────────────────────────────────────

function ExportDropdown({ audioPath, source }: { audioPath: string; source: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <Button
        variant="outline"
        size="sm"
        className="text-xs h-7"
        onClick={() => setOpen(!open)}
      >
        <Download className="w-3 h-3 mr-1" />
        Export
      </Button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg py-1 min-w-36">
          {[
            { label: "Plain Text", url: exportTextUrl(audioPath, source) },
            { label: "SRT Subtitles", url: exportSrtUrl(audioPath, source) },
            { label: "WebVTT Subtitles", url: exportVttUrl(audioPath, source) },
          ].map(({ label, url }) => (
            <a
              key={label}
              href={url}
              download
              className="block px-3 py-1.5 text-xs hover:bg-accent transition"
              onClick={() => setOpen(false)}
            >
              {label}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Word-level diff view ─────────────────────────────────────────────────────

function DiffView({ original, current }: { original: string; current: string }) {
  const diff = useMemo(() => computeWordDiff(original, current), [original, current]);
  return (
    <div className="text-sm leading-relaxed py-0">
      {diff.map((part, i) => (
        <span key={i}>
          {i > 0 && " "}
          <span
            className={
              part.type === "removed"
                ? "bg-red-500/20 text-red-400 line-through"
                : part.type === "added"
                  ? "bg-green-500/20 text-green-400"
                  : "text-muted-foreground/70"
            }
          >
            {part.text}
          </span>
        </span>
      ))}
    </div>
  );
}

// ── Inline segment row ────────────────────────────────────────────────────────

interface SegmentViewRowProps {
  segment: Segment;
  originalIndex: number;
  isActive: boolean;
  isFlagged: boolean;
  flagReasonText: string | null;
  isChanged: boolean;
  selected: boolean;
  onToggleSelect: () => void;
  audioPath?: string;
  speakers: string[];
  showSpeaker: boolean;
  showDelete: boolean;
  onTextChange: (text: string) => void;
  onSpeakerChange: (speaker: string) => void;
  onTimestampChange: (field: "start" | "end", value: number) => void;
  onDelete: () => void;
  onDismissFlag?: () => void;
  onInsertBefore?: () => void;
  onInsertAfter?: () => void;
  onMergeNext?: () => void;
  onSplit?: (cursorPos: number) => void;
  referenceText?: string;
  referenceLabel?: string;
}

function SegmentViewRow({
  segment,
  isActive,
  isFlagged,
  selected,
  onToggleSelect,
  flagReasonText,
  isChanged,
  audioPath,
  speakers,
  showSpeaker,
  showDelete,
  onTextChange,
  onSpeakerChange,
  onTimestampChange,
  onDelete,
  onDismissFlag,
  onInsertBefore,
  onInsertAfter,
  onMergeNext,
  onSplit,
  referenceText,
  referenceLabel,
}: SegmentViewRowProps) {
  const seekTo = useAudioStore((s) => s.seekTo);
  const [editingSpeaker, setEditingSpeaker] = useState(false);
  const [tsExpanded, setTsExpanded] = useState(false);
  const [refExpanded, setRefExpanded] = useState(true);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const speakerInputRef = useRef<HTMLInputElement>(null);

  const getAudioTime = () => useAudioStore.getState().currentTime;

  // Auto-resize textarea
  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    el.style.height = "0";
    el.style.height = el.scrollHeight + "px";
  }, [segment.text]);

  // Focus speaker input when entering edit mode
  useEffect(() => {
    if (editingSpeaker) speakerInputRef.current?.select();
  }, [editingSpeaker]);

  const validTime = isFinite(segment.start) && isFinite(segment.end);

  const handleSeek = () => {
    if (audioPath && validTime) seekTo(audioPath, segment.start);
  };

  const hasRef = referenceText != null;
  const hasDiff = hasRef && referenceText !== segment.text;

  // [BREAK] divider
  if (segment.speaker === "[BREAK]") {
    const gap = segment.end - segment.start;
    return (
      <div className="py-2 flex items-center gap-2 text-muted-foreground/40 px-4">
        <div className="flex-1 border-t border-border" />
        <span className="text-2xs uppercase select-none">
          {gap > 0 ? `${gap.toFixed(0)}s pause` : "break"}
        </span>
        <div className="flex-1 border-t border-border" />
      </div>
    );
  }

  return (
    <div
      className={`group py-1.5 px-4 rounded transition-colors ${
        isActive ? "bg-accent/60" :
        isFlagged ? "border-l-2 border-l-yellow-500" :
        isChanged ? "border-l-2 border-l-blue-500/50" :
        "hover:bg-accent/20"
      }`}
    >
      {/* Main row: checkbox | timestamp | speaker | text */}
      <div className="flex gap-3">
        {/* Selection checkbox */}
        <div className="shrink-0 pt-1">
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="w-3 h-3 accent-primary cursor-pointer"
          />
        </div>
        {/* Timestamp — click to toggle editor */}
        <div className="shrink-0 pt-0.5">
          <button
            onClick={() => setTsExpanded(!tsExpanded)}
            className="text-xs text-muted-foreground hover:text-foreground font-mono text-right leading-tight transition"
            title="Edit timestamps"
          >
            <div className="tabular-nums">{validTime ? formatTime(segment.start) : "--:--"}</div>
            <div className="tabular-nums text-muted-foreground/40">
              {validTime ? formatTime(segment.end) : "--:--"}
            </div>
          </button>
        </div>

        {/* Speaker */}
        {showSpeaker && (
          <div className="shrink-0 w-20 pt-0.5">
            {editingSpeaker ? (
              speakers.length > 0 ? (
                <select
                  value={segment.speaker}
                  onChange={(e) => {
                    onSpeakerChange(e.target.value);
                    setEditingSpeaker(false);
                  }}
                  onBlur={() => setEditingSpeaker(false)}
                  autoFocus
                  className={`${selectClass} w-full text-xs py-0 h-5`}
                >
                  {!speakers.includes(segment.speaker) && (
                    <option value={segment.speaker}>{segment.speaker}</option>
                  )}
                  {speakers.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  ref={speakerInputRef}
                  type="text"
                  value={segment.speaker}
                  onChange={(e) => onSpeakerChange(e.target.value)}
                  onBlur={() => setEditingSpeaker(false)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === "Escape") setEditingSpeaker(false);
                  }}
                  className="w-full text-xs bg-secondary border border-border rounded px-1 py-0 h-5 outline-none"
                />
              )
            ) : (
              <button
                onClick={() => setEditingSpeaker(true)}
                className="text-xs font-medium text-primary/70 truncate w-full text-left hover:text-primary transition"
                title="Click to edit speaker"
              >
                {segment.speaker}
              </button>
            )}
          </div>
        )}

        {/* Text + reference side-by-side on wide screens, equal height */}
        <div className={`flex-1 min-w-0 ${hasRef ? "flex flex-col lg:flex-row lg:gap-2" : ""}`}>
          <div className={hasRef ? "flex-1 min-w-0" : ""}>
            <textarea
              ref={textRef}
              value={segment.text}
              onChange={(e) => onTextChange(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && onSplit) {
                  e.preventDefault();
                  const pos = e.currentTarget.selectionStart;
                  if (pos > 0 && pos < segment.text.length) onSplit(pos);
                }
              }}
              className="w-full bg-transparent text-sm leading-relaxed resize-none outline-none overflow-hidden rounded border border-transparent hover:border-border focus:border-primary/50 focus:bg-accent/10 px-1.5 py-0 transition"
              rows={1}
            />
          </div>
          {hasRef && (
            <div className="flex-1 min-w-0 mt-0.5 lg:mt-0">
              {refExpanded && (
                <div className="rounded border border-border/40 bg-secondary/30 px-1.5 py-0.5">
                  {hasDiff
                    ? <DiffView original={referenceText!} current={segment.text} />
                    : <p className="text-sm leading-relaxed text-muted-foreground/50">{referenceText}</p>
                  }
                </div>
              )}
              <button
                onClick={() => setRefExpanded(!refExpanded)}
                className="flex items-center gap-0.5 text-2xs text-muted-foreground/60 hover:text-muted-foreground transition mt-0.5"
                title={refExpanded ? "Hide reference" : "Show reference"}
              >
                <Diff className="w-3 h-3" />
                {refExpanded ? <ChevronDown className="w-2.5 h-2.5" /> : <ChevronRight className="w-2.5 h-2.5" />}
                <span>{referenceLabel}{!refExpanded && !hasDiff ? " ✓" : ""}</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Flag reason — always visible on flagged segments */}
      {isFlagged && flagReasonText && (
        <div className="flex items-center gap-1 mt-0.5 text-yellow-500 text-2xs">
          <AlertTriangle className="w-3 h-3" />
          <span>{flagReasonText}</span>
          {onDismissFlag && (
            <button
              onClick={onDismissFlag}
              className="hover:text-yellow-300 transition ml-0.5"
              title="Dismiss this flag"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      )}

      {/* Actions row (under timestamp+speaker, visible on hover) */}
      <div className="flex items-center gap-0.5 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity" style={{ width: "fit-content" }}>
        {audioPath && validTime && (
          <button
            onClick={handleSeek}
            className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-secondary transition"
            title="Play from here"
          >
            <Play className="w-3 h-3" />
          </button>
        )}
        {onInsertBefore && (
          <button
            onClick={onInsertBefore}
            className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-secondary transition"
            title="Insert segment before"
          >
            <InsertBeforeIcon className="w-3.5 h-3.5" />
          </button>
        )}
        {onMergeNext && (
          <button
            onClick={onMergeNext}
            className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-secondary transition"
            title="Merge with next"
          >
            <Merge className="w-3 h-3" />
          </button>
        )}
        {onSplit && (
          <button
            onClick={() => {
              const pos = textRef.current?.selectionStart ?? Math.floor(segment.text.length / 2);
              if (pos > 0 && pos < segment.text.length) onSplit(pos);
            }}
            className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-secondary transition"
            title="Split at cursor (or midpoint)"
          >
            <Scissors className="w-3 h-3" />
          </button>
        )}
        {onInsertAfter && (
          <button
            onClick={onInsertAfter}
            className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-secondary transition"
            title="Insert segment after"
          >
            <InsertAfterIcon className="w-3.5 h-3.5" />
          </button>
        )}
        {showDelete && (
          <button
            onClick={onDelete}
            className="text-muted-foreground hover:text-destructive p-0.5 rounded hover:bg-secondary transition"
            title="Delete segment"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        )}
      </div>

      {/* Timestamp editor (expandable) */}
      {tsExpanded && (
        <div className="flex items-center gap-3 pl-1 text-xs py-1">
          <label className="flex items-center gap-1">
            Start:
            <input
              type="number"
              value={segment.start}
              onChange={(e) => onTimestampChange("start", Number(e.target.value))}
              step={0.1}
              className="w-20 bg-secondary text-secondary-foreground rounded px-1.5 py-0.5 border border-border"
            />
            <button
              onClick={() => onTimestampChange("start", Math.round(getAudioTime() * 10) / 10)}
              className="text-2xs text-muted-foreground hover:text-foreground bg-secondary rounded px-1 py-0.5 border border-border"
              title="Set to current playback position"
            >
              &larr; current time
            </button>
          </label>
          <label className="flex items-center gap-1">
            End:
            <input
              type="number"
              value={segment.end}
              onChange={(e) => onTimestampChange("end", Number(e.target.value))}
              step={0.1}
              className="w-20 bg-secondary text-secondary-foreground rounded px-1.5 py-0.5 border border-border"
            />
            <button
              onClick={() => onTimestampChange("end", Math.round(getAudioTime() * 10) / 10)}
              className="text-2xs text-muted-foreground hover:text-foreground bg-secondary rounded px-1 py-0.5 border border-border"
              title="Set to current playback position"
            >
              &larr; current time
            </button>
          </label>
        </div>
      )}

    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export interface TranscriptViewerProps {
  editorKey: string;
  audioPath?: string;
  loadSegments: () => Promise<Segment[]>;
  saveSegments: (segments: Segment[]) => Promise<unknown>;
  // Version support
  loadVersions?: () => Promise<VersionEntry[]>;
  loadVersion?: (id: string) => Promise<Segment[]>;
  deleteVersion?: (id: string) => Promise<unknown>;
  // Export
  exportSource?: string;
  // Features
  showDelete?: boolean;
  showFlags?: boolean;
  showSpeaker?: boolean;
  speakers?: string[];
  // Reference segments (for correction diff)
  referenceSegments?: Segment[];
  referenceLabel?: string;
  // Source label fallback when no versions
  sourceLabel?: string;
}

export default function TranscriptViewer({
  editorKey,
  audioPath,
  loadSegments,
  saveSegments,
  loadVersions,
  loadVersion,
  deleteVersion,
  exportSource,
  showDelete = true,
  showFlags = true,
  showSpeaker = true,
  speakers: externalSpeakers,
  referenceSegments,
  referenceLabel = "Original",
  sourceLabel,
}: TranscriptViewerProps) {
  const queryClient = useQueryClient();

  // ── Data loading ──────────────────────────────────────────────────────────

  const { data: latestSegments } = useQuery({
    queryKey: queryKeys.stepSegments(editorKey, audioPath),
    queryFn: loadSegments,
  });

  const { data: versions } = useQuery({
    queryKey: queryKeys.stepVersions(editorKey, audioPath),
    queryFn: loadVersions!,
    enabled: !!loadVersions,
  });

  // ── Version selector state ────────────────────────────────────────────────

  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [expandedInfo, setExpandedInfo] = useState(false);
  const [showDensity, setShowDensity] = useState(false);

  const { data: versionSegments } = useQuery({
    queryKey: queryKeys.stepVersionSegments(editorKey, audioPath, selectedVersionId),
    queryFn: () => loadVersion!(selectedVersionId!),
    enabled: !!loadVersion && !!selectedVersionId,
  });

  // Segments to display/edit — version override or latest
  const sourceSegments = selectedVersionId ? (versionSegments ?? latestSegments) : latestSegments;

  const selectedVersion = selectedVersionId
    ? (versions?.find((v) => v.id === selectedVersionId) ?? null)
    : null;

  // ── Editor state ──────────────────────────────────────────────────────────

  const editor = useSegments(sourceSegments ?? []);

  // Reset when source changes (new version selected or fresh load)
  useEffect(() => {
    if (sourceSegments) {
      editor.reset(sourceSegments);
      if (audioPath) {
        useAudioStore.getState().setAudioSegments(
          audioPath,
          sourceSegments.map((s) => ({
            start: s.start,
            end: s.end,
            speaker: s.speaker,
            text: s.text,
          })),
        );
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceSegments]);

  // ── Save / delete version ─────────────────────────────────────────────────

  const saveMutation = useMutation({
    mutationFn: () => saveSegments(editor.editedSegments),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments(editorKey, audioPath) });
      queryClient.invalidateQueries({ queryKey: queryKeys.stepVersions(editorKey, audioPath) });
    },
  });

  const deleteVersionMutation = useMutation({
    mutationFn: (id: string) => deleteVersion!(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.stepVersions(editorKey, audioPath) });
      queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments(editorKey, audioPath) });
    },
  });

  // ── Speaker list ──────────────────────────────────────────────────────────

  const speakers = useMemo(() => {
    if (externalSpeakers && externalSpeakers.length > 0) return externalSpeakers;
    if (!sourceSegments) return [];
    const set = new Set<string>();
    for (const seg of sourceSegments) {
      if (seg.speaker && seg.speaker !== "[BREAK]") set.add(seg.speaker);
    }
    return Array.from(set).sort();
  }, [sourceSegments, externalSpeakers]);

  // ── Merge dialog (when speakers differ) ──────────────────────────────────

  const [mergeDialog, setMergeDialog] = useState<{
    index: number;
    speakers: [string, string];
  } | null>(null);

  const handleMerge = useCallback(
    (originalIndex: number, currentSpeaker: string) => {
      const next = editor.getNextSegment(originalIndex);
      if (!next) return;
      if (next.speaker === currentSpeaker) {
        editor.mergeWithNext(originalIndex);
      } else {
        setMergeDialog({ index: originalIndex, speakers: [currentSpeaker, next.speaker] });
      }
    },
    [editor],
  );

  // ── Compare-with selector ─────────────────────────────────────────────────

  const [refChoice, setRefChoice] = useState<"default" | "none" | string>(
    referenceSegments ? "default" : "none",
  );

  useEffect(() => {
    setRefChoice(referenceSegments ? "default" : "none");
  }, [referenceSegments]);

  const [versionRefSegments, setVersionRefSegments] = useState<Segment[] | null>(null);
  const [versionRefLabel, setVersionRefLabel] = useState("");

  const handleRefChoiceChange = async (choice: string) => {
    setRefChoice(choice);
    if (choice === "default" || choice === "none") {
      setVersionRefSegments(null);
      setVersionRefLabel("");
      return;
    }
    if (loadVersion) {
      try {
        const data = await loadVersion(choice);
        setVersionRefSegments(data);
        const v = versions?.find((ver) => ver.id === choice);
        setVersionRefLabel(
          v ? `Version ${new Date(v.timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric" })}` : "Version",
        );
      } catch {
        setVersionRefSegments(null);
        setVersionRefLabel("");
        setRefChoice("none");
      }
    }
  };

  const effectiveReference =
    refChoice === "default" ? referenceSegments :
    refChoice === "none" ? undefined :
    versionRefSegments ?? undefined;
  const effectiveRefLabel =
    refChoice === "default" ? referenceLabel :
    refChoice === "none" ? "" :
    versionRefLabel;

  // Map originalIndex → position in editedSegments
  const origToEditedIdx = useMemo(() => {
    const map = new Map<number, number>();
    for (let i = 0; i < editor.originalIndices.length; i++) {
      map.set(editor.originalIndices[i], i);
    }
    return map;
  }, [editor.originalIndices]);

  // Reference mapping: align reference to edited positions, skipping deletes
  const refMapping = useMemo(() => {
    if (!effectiveReference) return null;
    const deletedOriginals = new Set<number>();
    const survivingSet = new Set(editor.originalIndices);
    const maxOrig = editor.originalIndices.length > 0
      ? Math.max(...editor.originalIndices, effectiveReference.length - 1)
      : effectiveReference.length - 1;
    for (let i = 0; i <= maxOrig; i++) {
      if (!survivingSet.has(i)) deletedOriginals.add(i);
    }
    const activeRefIndices: number[] = [];
    for (let i = 0; i < effectiveReference.length; i++) {
      if (!deletedOriginals.has(i)) activeRefIndices.push(i);
    }
    const map = new Map<number, number>();
    for (let e = 0; e < editor.editedSegments.length && e < activeRefIndices.length; e++) {
      map.set(e, activeRefIndices[e]);
    }
    return map;
  }, [effectiveReference, editor.editedSegments, editor.originalIndices]);

  const getRef = (editedIdx: number) => {
    if (!effectiveReference || !refMapping) return undefined;
    const refIdx = refMapping.get(editedIdx);
    return refIdx != null ? effectiveReference[refIdx] : undefined;
  };

  const isChanged = (seg: Segment, origIdx: number) => {
    if (!effectiveReference || seg.speaker === "[BREAK]") return false;
    const editedIdx = origToEditedIdx.get(origIdx) ?? origIdx;
    const ref = getRef(editedIdx);
    return ref != null && ref.text !== seg.text;
  };

  const changedCount = useMemo(() => {
    if (!effectiveReference) return 0;
    let count = 0;
    for (let e = 0; e < editor.editedSegments.length; e++) {
      if (isChanged(editor.editedSegments[e], editor.originalIndices[e])) count++;
    }
    return count;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor.editedSegments, editor.originalIndices, effectiveReference]);

  const hasCompareOptions = !!referenceSegments || (versions && versions.length > 0);

  // ── Selection ─────────────────────────────────────────────────────────────

  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());

  const toggleSelect = useCallback((origIdx: number) => {
    setSelectedIndices((prev) => {
      const next = new Set(prev);
      if (next.has(origIdx)) next.delete(origIdx);
      else next.add(origIdx);
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => setSelectedIndices(new Set()), []);

  const bulkDelete = useCallback(() => {
    // Delete in reverse order so indices stay valid
    const sorted = Array.from(selectedIndices).sort((a, b) => b - a);
    for (const idx of sorted) editor.deleteSegment(idx);
    clearSelection();
  }, [selectedIndices, editor, clearSelection]);

  const bulkSpeaker = useCallback((speaker: string) => {
    for (const idx of selectedIndices) editor.updateSpeaker(idx, speaker);
    clearSelection();
  }, [selectedIndices, editor, clearSelection]);

  const bulkMerge = useCallback(() => {
    // Merge consecutive selected segments (sorted ascending)
    const sorted = Array.from(selectedIndices).sort((a, b) => a - b);
    if (sorted.length < 2) return;
    // Merge from last to first to preserve indices
    for (let i = sorted.length - 1; i > 0; i--) {
      if (sorted[i] === sorted[i - 1] + 1) {
        editor.mergeWithNext(sorted[i - 1]);
      }
    }
    clearSelection();
  }, [selectedIndices, editor, clearSelection]);

  // ── Filtering / pagination ────────────────────────────────────────────────

  const filters = useSegmentFiltering();
  const [dismissedFlags, setDismissedFlags] = useState<Set<number>>(new Set());

  const dismissFlag = (origIdx: number) => {
    setDismissedFlags((prev) => new Set(prev).add(origIdx));
  };

  const isFlaggedSeg = (seg: Segment, origIdx?: number): boolean => {
    if (origIdx != null && dismissedFlags.has(origIdx)) return false;
    return flagReason(seg, filters.densityThreshold, filters.maxDensityThreshold) !== null;
  };

  const { displaySegments, pageSegments, totalPages, flaggedCount } = useFilteredSegments(
    editor.editedSegments,
    editor.originalIndices,
    filters,
    { dismissedFlags, isChanged },
  );

  useEffect(() => {
    if (pageSegments.length > 0) {
      filters.setAnchorOrigIdx(pageSegments[0].originalIndex);
    }
  }, [pageSegments]); // eslint-disable-line react-hooks/exhaustive-deps

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
  }, [filters.page]);

  // ── Active segment tracking ───────────────────────────────────────────────

  const storeAudioPath = useAudioStore((s) => s.audioPath);
  const isPlayingThisFile = audioPath != null && storeAudioPath === audioPath;
  const [activeOrigIdx, setActiveOrigIdx] = useState<number | null>(null);

  useEffect(() => {
    if (!isPlayingThisFile) {
      setActiveOrigIdx(null);
      return;
    }
    const interval = setInterval(() => {
      const t = useAudioStore.getState().currentTime;
      if (!useAudioStore.getState().isPlaying) return;
      for (let e = editor.editedSegments.length - 1; e >= 0; e--) {
        const seg = editor.editedSegments[e];
        if (seg.start <= t && t < seg.end) {
          const origIdx = editor.originalIndices[e];
          setActiveOrigIdx((prev) => (prev === origIdx ? prev : origIdx));
          return;
        }
      }
      setActiveOrigIdx(null);
    }, 250);
    return () => clearInterval(interval);
  }, [isPlayingThisFile, editor.editedSegments, editor.originalIndices]);

  // Auto-scroll to active segment
  useEffect(() => {
    if (activeOrigIdx == null || !scrollRef.current) return;
    const el = scrollRef.current.querySelector(`[data-orig-idx="${activeOrigIdx}"]`);
    if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeOrigIdx]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────

  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editor.isDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [editor.isDirty]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      if (e.key === "s") {
        e.preventDefault();
        if (editor.isDirty && !saveMutation.isPending) saveMutation.mutate();
      } else if (e.key === "z" && !e.shiftKey) {
        if ((e.target as HTMLElement)?.tagName !== "TEXTAREA" && editor.canUndo) {
          e.preventDefault();
          editor.undo();
        }
      } else if (e.key === "f") {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [editor.isDirty, editor.canUndo, saveMutation]);

  // ── Loading state ─────────────────────────────────────────────────────────

  if (!sourceSegments) {
    return (
      <div className="p-6 text-muted-foreground text-sm">Loading transcript...</div>
    );
  }

  if (sourceSegments.length === 0) {
    return (
      <div className="p-6 text-muted-foreground text-sm">No segments available.</div>
    );
  }

  // ── Version info block ────────────────────────────────────────────────────

  const infoVersion = selectedVersion ?? (versions && versions.length > 0 ? versions[0] : null);
  const infoItems = infoVersion ? versionInfo(infoVersion) : [];

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full">
      {/* ── Top toolbar: title + version + compare ── */}
      <div className="px-4 py-2 border-b border-border space-y-1">
        {/* Row 1: title + version selector + file details */}
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold shrink-0">Transcript</span>
          {versions && versions.length > 0 ? (
            <div className="flex items-center gap-1.5">
              <select
                value={selectedVersionId ?? ""}
                onChange={(e) => setSelectedVersionId(e.target.value || null)}
                className={`${selectClass}`}
              >
                <option value="">Latest</option>
                {versions.map((v) => (
                  <option key={v.id} value={v.id}>
                    {versionDate(v)} · {versionLabel(v)} ({v.segment_count} seg)
                  </option>
                ))}
              </select>
              {selectedVersionId && deleteVersion && (
                <button
                  onClick={() => {
                    deleteVersionMutation.mutate(selectedVersionId);
                    setSelectedVersionId(null);
                  }}
                  className="text-xs text-muted-foreground hover:text-destructive transition"
                  title="Delete this version"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
              {infoItems.length > 0 && (
                <button
                  onClick={() => setExpandedInfo(!expandedInfo)}
                  className="text-xs text-muted-foreground/60 hover:text-muted-foreground transition shrink-0"
                >
                  File details
                </button>
              )}
            </div>
          ) : sourceLabel ? (
            <span className="text-xs text-muted-foreground">
              Source: <span className="font-mono">{sourceLabel}</span>
            </span>
          ) : null}
        </div>

        {/* Row 2: compare with (aligned with version selector) */}
        {hasCompareOptions && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground shrink-0">Compare with</span>
            <select
              value={refChoice}
              onChange={(e) => handleRefChoiceChange(e.target.value)}
              className={`${selectClass} text-xs`}
            >
              <option value="none">None</option>
              {referenceSegments && (
                <option value="default">{referenceLabel}</option>
              )}
              {versions && versions.map((v) => (
                <option key={v.id} value={v.id}>
                  {versionDate(v)} · {versionLabel(v)} ({v.segment_count} seg)
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Version info (expandable) */}
        {infoItems.length > 0 && expandedInfo && (
          <div className="bg-secondary/50 rounded border border-border/50 px-3 py-2 text-xs space-y-0.5">
            {infoItems.map(({ key, value }) => (
              <div key={key} className="flex gap-2">
                <span className="text-muted-foreground shrink-0 w-20">{key}</span>
                <span className="truncate">{value}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Action bar: select-all + search + badges + actions ── */}
      <div className="px-4 py-1.5 border-b border-border space-y-1">
        <div className="flex items-center gap-2 flex-wrap">
          {/* Select all checkbox */}
          <input
            type="checkbox"
            checked={selectedIndices.size > 0 && selectedIndices.size === pageSegments.filter((s) => s.segment.speaker !== "[BREAK]").length}
            ref={(el) => {
              if (el) {
                const nonBreak = pageSegments.filter((s) => s.segment.speaker !== "[BREAK]");
                el.indeterminate = selectedIndices.size > 0 && selectedIndices.size < nonBreak.length;
              }
            }}
            onChange={() => {
              const nonBreak = pageSegments.filter((s) => s.segment.speaker !== "[BREAK]");
              if (selectedIndices.size === nonBreak.length) {
                clearSelection();
              } else {
                setSelectedIndices(new Set(nonBreak.map((s) => s.originalIndex)));
              }
            }}
            className="w-3 h-3 accent-primary cursor-pointer"
            title="Select all"
          />
          {/* Search (always visible) */}
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground/50" />
            <input
              ref={searchRef}
              type="text"
              placeholder="Search…"
              value={filters.searchQuery}
              onChange={(e) => { filters.setSearchQuery(e.target.value); filters.setPage(0); }}
              className="h-6 w-40 text-xs bg-secondary border border-border rounded pl-6 pr-6 outline-none focus:border-primary/50"
            />
            {filters.searchQuery && (
              <button
                onClick={() => { filters.setSearchQuery(""); filters.setPage(0); }}
                className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
          {/* Speaker filter */}
          {speakers.length > 1 && (
            <select
              value={filters.speakerFilter}
              onChange={(e) => { filters.setSpeakerFilter(e.target.value); filters.setPage(0); }}
              className={`${selectClass} text-xs`}
            >
              <option value="">All speakers</option>
              {speakers.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          )}
          {/* Flagged badge */}
          {showFlags && flaggedCount > 0 && (
            <button
              onClick={() => { filters.setShowFlaggedOnly(!filters.showFlaggedOnly); filters.setPage(0); }}
              className={`flex items-center gap-1 text-xs px-1.5 py-0.5 rounded border transition ${
                filters.showFlaggedOnly ? "border-yellow-500/50 text-yellow-500" : "border-border text-muted-foreground hover:text-foreground"
              }`}
              title="Show flagged only"
            >
              <AlertTriangle className="w-3 h-3" />
              {flaggedCount}
            </button>
          )}
          {/* Density popover (icon only) */}
          <div className="relative">
            <button
              onClick={() => setShowDensity(!showDensity)}
              className={`p-1 rounded border transition ${
                showDensity ? "border-primary/50 text-foreground" : "border-border text-muted-foreground hover:text-foreground"
              }`}
              title="Speech density thresholds"
            >
              <Activity className="w-3 h-3" />
            </button>
            {showDensity && (
              <div className="absolute left-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg p-3 space-y-2 min-w-52">
                <p className="text-2xs text-muted-foreground font-medium uppercase">Speech density thresholds</p>
                <label className="flex items-center gap-2 text-xs">
                  <span className="shrink-0">{">"} </span>
                  <input
                    type="range"
                    min={1}
                    max={40}
                    value={filters.densityThreshold}
                    onChange={(e) => filters.setDensityThreshold(Number(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-muted-foreground w-20 text-right">{filters.densityThreshold} char/s</span>
                </label>
                <label className="flex items-center gap-2 text-xs">
                  <span className="shrink-0">{"<"} </span>
                  <input
                    type="range"
                    min={20}
                    max={150}
                    value={filters.maxDensityThreshold}
                    onChange={(e) => filters.setMaxDensityThreshold(Number(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-muted-foreground w-20 text-right">{filters.maxDensityThreshold} char/s</span>
                </label>
              </div>
            )}
          </div>
          {/* Changed badge */}
          {effectiveReference && changedCount > 0 && (
            <button
              onClick={() => { filters.setShowChangedOnly(!filters.showChangedOnly); filters.setPage(0); }}
              className={`flex items-center gap-1 text-xs px-1.5 py-0.5 rounded border transition ${
                filters.showChangedOnly ? "border-blue-500/50 text-blue-400" : "border-border text-muted-foreground hover:text-foreground"
              }`}
              title="Show changed only"
            >
              <Diff className="w-3 h-3" />
              {changedCount}
            </button>
          )}
          <div className="flex-1" />
          {/* Right: undo, export, save */}
          {editor.canUndo && (
            <Button
              variant="ghost"
              size="sm"
              className="text-xs h-7"
              onClick={editor.undo}
              title="Undo (Cmd+Z)"
            >
              <Undo2 className="w-3 h-3 mr-1" />
              Undo
            </Button>
          )}
          {exportSource && audioPath && (
            <ExportDropdown audioPath={audioPath} source={exportSource} />
          )}
          <Button
            variant={editor.isDirty ? "default" : "outline"}
            size="sm"
            className="text-xs h-7"
            onClick={() => saveMutation.mutate()}
            disabled={saveMutation.isPending || !editor.isDirty}
            title="Save (Cmd+S)"
          >
            <Save className="w-3 h-3 mr-1" />
            {saveMutation.isPending ? "Saving..." : "Save"}
          </Button>
        </div>

        {/* Selection actions bar */}
        {selectedIndices.size > 0 && (
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-muted-foreground">{selectedIndices.size} selected</span>
            <select
              value=""
              onChange={(e) => { if (e.target.value) { bulkSpeaker(e.target.value); } }}
              className={`${selectClass} text-xs`}
            >
              <option value="">Set speaker…</option>
              {speakers.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
            {selectedIndices.size >= 2 && (
              <Button variant="outline" size="sm" className="text-xs h-6" onClick={bulkMerge}>
                <Merge className="w-3 h-3 mr-1" />
                Merge
              </Button>
            )}
            <Button variant="outline" size="sm" className="text-xs h-6 text-destructive hover:text-destructive" onClick={bulkDelete}>
              <Trash2 className="w-3 h-3 mr-1" />
              Delete
            </Button>
            <button onClick={clearSelection} className="text-xs text-muted-foreground hover:text-foreground transition ml-1" aria-label="Clear selection">
              <X className="w-3 h-3" />
            </button>
          </div>
        )}
      </div>

      {/* ── Segment list ── */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto py-2">
        {pageSegments.map(({ segment, originalIndex }) => {
          const editedIdx = origToEditedIdx.get(originalIndex) ?? originalIndex;
          const ref = getRef(editedIdx);
          const isBreak = segment.speaker === "[BREAK]";
          return (
            <div key={`${editorKey}-${originalIndex}`} data-orig-idx={originalIndex}>
              <SegmentViewRow
                segment={segment}
                originalIndex={originalIndex}
                isActive={activeOrigIdx === originalIndex}
                isFlagged={showFlags ? isFlaggedSeg(segment, originalIndex) : false}
                flagReasonText={showFlags ? flagReason(segment, filters.densityThreshold, filters.maxDensityThreshold) : null}
                isChanged={isChanged(segment, originalIndex)}
                selected={selectedIndices.has(originalIndex)}
                onToggleSelect={() => toggleSelect(originalIndex)}
                audioPath={audioPath}
                speakers={speakers}
                showSpeaker={showSpeaker}
                showDelete={showDelete}
                onTextChange={(text) => editor.updateText(originalIndex, text)}
                onSpeakerChange={(speaker) => editor.updateSpeaker(originalIndex, speaker)}
                onTimestampChange={(field, value) => editor.updateTimestamp(originalIndex, field, value)}
                onDelete={() => editor.deleteSegment(originalIndex)}
                onDismissFlag={showFlags ? () => dismissFlag(originalIndex) : undefined}
                onInsertBefore={() =>
                  editor.insertAfter(originalIndex - 1, {
                    speaker: segment.speaker,
                    text: "",
                    start: segment.start,
                    end: segment.start,
                  })
                }
                onInsertAfter={() =>
                  editor.insertAfter(originalIndex, {
                    speaker: segment.speaker,
                    text: "",
                    start: segment.end,
                    end: segment.end,
                  })
                }
                onMergeNext={() => handleMerge(originalIndex, segment.speaker)}
                onSplit={(cursorPos) => editor.splitAt(originalIndex, cursorPos)}
                referenceText={ref && !isBreak ? ref.text : undefined}
                referenceLabel={effectiveRefLabel}
              />
            </div>
          );
        })}
      </div>

      {/* ── Pagination ── */}
      {totalPages > 1 && (
        <Pagination
          page={filters.page}
          totalPages={totalPages}
          pageSize={filters.pageSize}
          onPageChange={filters.setPage}
          onPageSizeChange={(s) => {
            const anchor = filters.anchorOrigIdx;
            filters.setPageSize(s);
            if (anchor != null) {
              const pos = displaySegments.findIndex((d) => d.originalIndex === anchor);
              filters.setPage(pos >= 0 ? Math.floor(pos / s) : 0);
            } else {
              filters.setPage(0);
            }
          }}
        />
      )}

      {/* ── Merge speaker dialog ── */}
      {mergeDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-popover border border-border rounded-lg p-4 shadow-lg space-y-3 max-w-xs">
            <p className="text-sm font-medium">Which speaker for the merged segment?</p>
            <div className="flex flex-col gap-2">
              {mergeDialog.speakers.map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    editor.mergeWithNext(mergeDialog.index, s);
                    setMergeDialog(null);
                  }}
                  className="px-3 py-2 text-sm rounded border border-border bg-secondary hover:bg-accent transition text-left"
                >
                  {s}
                </button>
              ))}
            </div>
            <button
              onClick={() => setMergeDialog(null)}
              className="text-xs text-muted-foreground hover:text-foreground transition"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
