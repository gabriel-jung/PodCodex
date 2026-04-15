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
import { formatTime, versionOption, versionInfo, selectClass } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";
import { computeWordDiff } from "@/lib/diffUtils";
import { Button } from "@/components/ui/button";
import Pagination from "./Pagination";
import SpeakerStrip from "./SpeakerStrip";
import SectionHeader from "@/components/common/SectionHeader";
import {
  Download,
  Save,
  Undo2,
  Play,
  Pause,
  Trash2,
  Merge,
  Scissors,
  Search,
  X,
  Diff,
  Filter,
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

function ExportDropdown({
  audioPath,
  source,
  filename,
}: {
  audioPath: string;
  source: string;
  filename?: string;
}) {
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
            { label: "Plain Text", ext: "txt", url: exportTextUrl(audioPath, source) },
            { label: "SRT Subtitles", ext: "srt", url: exportSrtUrl(audioPath, source) },
            { label: "WebVTT Subtitles", ext: "vtt", url: exportVttUrl(audioPath, source) },
          ].map(({ label, ext, url }) => (
            <a
              key={label}
              href={url}
              download={filename ? `${filename}.${ext}` : ""}
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
  isPendingRemoval?: boolean;
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
}

function SegmentViewRow({
  segment,
  isActive,
  isFlagged,
  selected,
  onToggleSelect,
  flagReasonText,
  isChanged,
  isPendingRemoval,
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
}: SegmentViewRowProps) {
  const seekTo = useAudioStore((s) => s.seekTo);
  const pauseAudio = useAudioStore((s) => s.pauseAudio);
  const isPlaying = useAudioStore((s) => s.isPlaying);
  const [editingSpeaker, setEditingSpeaker] = useState(false);
  const [tsExpanded, setTsExpanded] = useState(false);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const speakerInputRef = useRef<HTMLInputElement>(null);

  const getAudioTime = () => useAudioStore.getState().currentTime;

  // Auto-resize textarea — recomputes on text change AND on width change
  // (layout shifts: sidebar toggle, comparison column, window resize).
  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    let lastWidth = -1;
    const recompute = () => {
      const w = el.clientWidth;
      if (w === lastWidth) return;
      lastWidth = w;
      el.style.height = "0";
      el.style.height = el.scrollHeight + "px";
    };
    recompute();
    const ro = new ResizeObserver(recompute);
    ro.observe(el);
    return () => ro.disconnect();
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
        <span className="text-2xs select-none">
          {gap > 0 ? `${gap.toFixed(0)}s pause` : "break"}
        </span>
        <div className="flex-1 border-t border-border" />
      </div>
    );
  }

  return (
    <div
      className={`group py-1.5 px-4 rounded transition-colors ${
        isPendingRemoval ? "opacity-50 line-through bg-destructive/5 border-l-2 border-l-destructive/50" :
        isActive ? "bg-accent/60" :
        isFlagged ? "border-l-2 border-l-warning" :
        isChanged ? "border-l-2 border-l-blue-500/50" :
        "hover:bg-accent/20"
      }`}
    >
      {/* Main row: checkbox | timestamp | speaker | text */}
      <div className="flex gap-3">
        {/* Selection checkbox — revealed on hover or when any row is selected */}
        <div className={`shrink-0 pt-1 transition-opacity ${selected ? "opacity-100" : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"}`}>
          <input
            type="checkbox"
            checked={selected}
            onChange={onToggleSelect}
            className="w-3 h-3 accent-primary cursor-pointer"
            aria-label="Select segment"
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
                className="text-xs font-medium truncate w-full text-left hover:opacity-80 transition"
                style={{ color: speakerColor(segment.speaker) }}
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
            <div className="flex-1 min-w-0 mt-0.5 lg:mt-0 rounded border border-border/40 bg-secondary/30 px-1.5 py-0.5">
              {hasDiff
                ? <DiffView original={referenceText!} current={segment.text} />
                : <p className="text-sm leading-relaxed text-muted-foreground/50">{referenceText}</p>
              }
            </div>
          )}
        </div>
      </div>

      {/* Flag reason — always visible on flagged segments */}
      {isFlagged && flagReasonText && (
        <div className="flex items-center gap-1 mt-0.5 text-warning text-2xs">
          <AlertTriangle className="w-3 h-3" />
          <span>{flagReasonText}</span>
          {onDismissFlag && (
            <button
              onClick={onDismissFlag}
              className="hover:text-warning/70 transition ml-0.5"
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
          isActive && isPlaying ? (
            <button
              onClick={pauseAudio}
              className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-secondary transition"
              title="Pause"
            >
              <Pause className="w-3 h-3" />
            </button>
          ) : (
            <button
              onClick={handleSeek}
              className="text-muted-foreground hover:text-foreground p-0.5 rounded hover:bg-secondary transition"
              title="Play from here"
            >
              <Play className="w-3 h-3" />
            </button>
          )
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
  saveSpeakerMap?: (mapping: Record<string, string>) => Promise<unknown>;
  // Version support
  loadVersions?: () => Promise<VersionEntry[]>;
  loadVersion?: (id: string) => Promise<Segment[]>;
  deleteVersion?: (id: string) => Promise<unknown>;
  // Export
  exportSource?: string;
  exportFilename?: string;
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

/** Native <select> that sizes to its currently-selected label, not to its
 *  longest option. An invisible mirror span with the selected label defines
 *  the container's intrinsic width; the real select is absolutely positioned
 *  over it so its own longest-option measurement doesn't leak into layout. */
function AutoWidthSelect({
  value,
  onChange,
  selectedLabel,
  children,
}: {
  value: string;
  onChange: (v: string) => void;
  selectedLabel: string;
  children: React.ReactNode;
}) {
  return (
    <div className="relative shrink-0 max-w-[18rem]">
      <span
        aria-hidden
        className={`${selectClass} text-xs invisible block whitespace-nowrap pr-7`}
      >
        {selectedLabel}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`${selectClass} text-xs absolute inset-0 w-full`}
      >
        {children}
      </select>
    </div>
  );
}

export default function TranscriptViewer({
  editorKey,
  audioPath,
  loadSegments,
  saveSegments,
  saveSpeakerMap,
  loadVersions,
  loadVersion,
  deleteVersion,
  exportSource,
  exportFilename,
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

  const [pendingRenames, setPendingRenames] = useState<Record<string, string>>({});
  const [pendingRemovals, setPendingRemovals] = useState<Set<string>>(() => new Set());
  const [addedSpeakers, setAddedSpeakers] = useState<string[]>([]);

  const hasPendingStripChanges =
    Object.keys(pendingRenames).length > 0 || pendingRemovals.size > 0;

  // Reset when source changes (new version selected or fresh load)
  useEffect(() => {
    if (sourceSegments) {
      editor.reset(sourceSegments);
      setPendingRenames({});
      setPendingRemovals(new Set());
      setAddedSpeakers([]);
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
    mutationFn: async () => {
      const writes: Promise<unknown>[] = [];
      if (editor.isDirty || hasPendingStripChanges) {
        const finalSegments = editor.editedSegments
          .filter((seg) => !pendingRemovals.has(seg.speaker))
          .map((seg) => {
            const renamed = pendingRenames[seg.speaker];
            return renamed && renamed !== seg.speaker ? { ...seg, speaker: renamed } : seg;
          });
        writes.push(saveSegments(finalSegments));
      }
      if (hasPendingStripChanges && saveSpeakerMap) {
        const mapping: Record<string, string> = {};
        for (const [from, to] of Object.entries(pendingRenames)) {
          if (from !== to) mapping[from] = to;
        }
        if (Object.keys(mapping).length > 0) {
          writes.push(saveSpeakerMap(mapping));
        }
      }
      await Promise.all(writes);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments(editorKey, audioPath) });
      queryClient.invalidateQueries({ queryKey: queryKeys.stepVersions(editorKey, audioPath) });
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
    },
  });

  const deleteVersionMutation = useMutation({
    mutationFn: (id: string) => deleteVersion!(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.stepVersions(editorKey, audioPath) });
      queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments(editorKey, audioPath) });
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
    },
  });

  // ── Speaker list ──────────────────────────────────────────────────────────

  const speakers = useMemo(() => {
    const set = new Set<string>(externalSpeakers ?? []);
    for (const seg of sourceSegments ?? []) {
      if (seg.speaker && seg.speaker !== "[BREAK]") set.add(seg.speaker);
    }
    for (const name of addedSpeakers) set.add(name);
    for (const target of Object.values(pendingRenames)) set.add(target);
    for (const removed of pendingRemovals) set.delete(removed);
    return Array.from(set).sort();
  }, [sourceSegments, externalSpeakers, addedSpeakers, pendingRenames, pendingRemovals]);

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

  const handleRefChoiceChange = async (choice: string) => {
    setRefChoice(choice);
    if (choice === "default" || choice === "none") {
      setVersionRefSegments(null);
      return;
    }
    if (loadVersion) {
      try {
        const data = await loadVersion(choice);
        setVersionRefSegments(data);
      } catch {
        setVersionRefSegments(null);
        setRefChoice("none");
      }
    }
  };

  const effectiveReference =
    refChoice === "default" ? referenceSegments :
    refChoice === "none" ? undefined :
    versionRefSegments ?? undefined;
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

  const handleStripRename = useCallback((from: string, to: string) => {
    setPendingRenames((prev) => {
      const next = { ...prev };
      if (!to || to === from) {
        delete next[from];
      } else {
        next[from] = to;
      }
      return next;
    });
  }, []);

  const handleStripToggleRemoved = useCallback((name: string) => {
    setPendingRemovals((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const handleStripAddSpeaker = useCallback((name: string) => {
    setAddedSpeakers((prev) => (prev.includes(name) ? prev : [...prev, name]));
  }, []);

  const handleStripRemoveAdded = useCallback((name: string) => {
    setAddedSpeakers((prev) => prev.filter((n) => n !== name));
  }, []);

  // ── Selection ─────────────────────────────────────────────────────────────

  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(() => new Set());

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
  const [dismissedFlags, setDismissedFlags] = useState<Set<number>>(() => new Set());

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

  const editedSegmentsRef = useRef(editor.editedSegments);
  editedSegmentsRef.current = editor.editedSegments;
  const originalIndicesRef = useRef(editor.originalIndices);
  originalIndicesRef.current = editor.originalIndices;

  useEffect(() => {
    if (!isPlayingThisFile) {
      setActiveOrigIdx(null);
      return;
    }
    const interval = setInterval(() => {
      const t = useAudioStore.getState().currentTime;
      if (!useAudioStore.getState().isPlaying) return;
      const segs = editedSegmentsRef.current;
      const indices = originalIndicesRef.current;
      for (let e = segs.length - 1; e >= 0; e--) {
        const seg = segs[e];
        if (seg.start <= t && t < seg.end) {
          const origIdx = indices[e];
          setActiveOrigIdx((prev) => (prev === origIdx ? prev : origIdx));
          return;
        }
      }
      setActiveOrigIdx((prev) => (prev == null ? prev : null));
    }, 250);
    return () => clearInterval(interval);
  }, [isPlayingThisFile]);

  // Auto-scroll to active segment
  useEffect(() => {
    if (activeOrigIdx == null || !scrollRef.current) return;
    const el = scrollRef.current.querySelector(`[data-orig-idx="${activeOrigIdx}"]`);
    if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeOrigIdx]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────

  const searchRef = useRef<HTMLInputElement>(null);

  const isDirty = editor.isDirty || hasPendingStripChanges;

  useEffect(() => {
    if (!isDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      if (e.key === "s") {
        e.preventDefault();
        if (isDirty && !saveMutation.isPending) saveMutation.mutate();
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
  }, [isDirty, editor.canUndo, saveMutation]);

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

  const selectedVersionLabel = selectedVersion ? versionOption(selectedVersion) : "Latest";
  const refVersion = versions?.find((v) => v.id === refChoice) ?? null;
  const refSelectedLabel =
    refChoice === "none"
      ? "None"
      : refChoice === "default"
        ? referenceLabel ?? "Original"
        : refVersion
          ? versionOption(refVersion)
          : "None";

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 border-b border-border space-y-1">
        {versions && versions.length > 0 ? (
          <div className="flex items-center gap-2 flex-wrap">
            <SectionHeader className="shrink-0 w-20">Version</SectionHeader>
            <AutoWidthSelect
              value={selectedVersionId ?? ""}
              onChange={(v) => setSelectedVersionId(v || null)}
              selectedLabel={selectedVersionLabel}
            >
              <option value="">Latest</option>
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  {versionOption(v)}
                </option>
              ))}
            </AutoWidthSelect>
            {hasCompareOptions && (
              <>
                <span className="text-xs text-muted-foreground/60">vs</span>
                <AutoWidthSelect
                  value={refChoice}
                  onChange={handleRefChoiceChange}
                  selectedLabel={refSelectedLabel}
                >
                  <option value="none">None</option>
                  {referenceSegments && (
                    <option value="default">{referenceLabel}</option>
                  )}
                  {versions.map((v) => (
                    <option key={v.id} value={v.id}>
                      {versionOption(v)}
                    </option>
                  ))}
                </AutoWidthSelect>
              </>
            )}
            <div className="flex-1" />
            {infoItems.length > 0 && (
              <button
                onClick={() => setExpandedInfo(!expandedInfo)}
                className="text-xs text-muted-foreground/60 hover:text-muted-foreground transition shrink-0"
              >
                File details
              </button>
            )}
            {deleteVersion && (
              <button
                onClick={() => {
                  const targetId = selectedVersionId ?? versions[0].id;
                  const target = versions.find((v) => v.id === targetId);
                  if (!target) return;
                  if (!window.confirm(`Delete this version?\n\n${versionOption(target)}\n\nThis removes both the file and the database entry and cannot be undone.`)) return;
                  deleteVersionMutation.mutate(targetId);
                  setSelectedVersionId(null);
                }}
                className="p-1 rounded text-muted-foreground/60 hover:text-destructive hover:bg-destructive/10 transition shrink-0"
                title="Delete current version (file + db entry)"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            )}
          </div>
        ) : sourceLabel ? (
          <div className="flex items-center gap-2">
            <SectionHeader className="shrink-0 w-20">Version</SectionHeader>
            <span className="text-xs text-muted-foreground font-mono">{sourceLabel}</span>
          </div>
        ) : null}

        {hasCompareOptions && (!versions || versions.length === 0) && (
          <div className="flex items-center gap-2">
            <SectionHeader className="shrink-0 w-20">Compare</SectionHeader>
            <select
              value={refChoice}
              onChange={(e) => handleRefChoiceChange(e.target.value)}
              className={`${selectClass} text-xs`}
            >
              <option value="none">None</option>
              {referenceSegments && (
                <option value="default">{referenceLabel}</option>
              )}
            </select>
          </div>
        )}

        {saveSpeakerMap && (
          <SpeakerStrip
            segments={sourceSegments}
            pendingRenames={pendingRenames}
            pendingRemovals={pendingRemovals}
            addedSpeakers={addedSpeakers}
            showSpeakers={externalSpeakers}
            audioPath={audioPath}
            onRename={handleStripRename}
            onToggleRemoved={handleStripToggleRemoved}
            onAddSpeaker={handleStripAddSpeaker}
            onRemoveAdded={handleStripRemoveAdded}
          />
        )}

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
          {speakers.length > 1 && (
            <div
              className={`flex items-center gap-1 rounded border pl-1.5 transition ${
                filters.speakerFilter ? "border-primary/50 text-foreground" : "border-border text-muted-foreground"
              }`}
              title="Filter view by speaker"
            >
              <Filter className="w-3 h-3 shrink-0" />
              <select
                value={filters.speakerFilter}
                onChange={(e) => { filters.setSpeakerFilter(e.target.value); filters.setPage(0); }}
                className="bg-transparent outline-none text-xs py-0.5 pr-5"
              >
                <option value="">all</option>
                {speakers.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          )}
          {/* Flagged badge */}
          {showFlags && flaggedCount > 0 && (
            <button
              onClick={() => { filters.setShowFlaggedOnly(!filters.showFlaggedOnly); filters.setPage(0); }}
              className={`flex items-center gap-1 text-xs px-1.5 py-0.5 rounded border transition ${
                filters.showFlaggedOnly ? "border-warning/50 text-warning" : "border-border text-muted-foreground hover:text-foreground"
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
                <p className="text-2xs text-muted-foreground font-medium">Speech density thresholds</p>
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
            <ExportDropdown audioPath={audioPath} source={exportSource} filename={exportFilename} />
          )}
          <Button
            variant={isDirty ? "default" : "outline"}
            size="sm"
            className="text-xs h-7"
            onClick={() => saveMutation.mutate()}
            disabled={saveMutation.isPending || !isDirty}
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
                isPendingRemoval={pendingRemovals.has(segment.speaker)}
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
