import { useMemo, useState, useEffect, useCallback, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Segment, VersionInfo } from "@/api/types";
import { useSegments } from "@/hooks/useSegments";
import { useAppStore } from "@/store";
import EditorToolbar from "./EditorToolbar";
import SegmentRow from "./SegmentRow";
import Pagination from "./Pagination";

interface SegmentEditorProps {
  editorKey: string;
  audioPath?: string;
  episodeDuration?: number;
  loadSegments: () => Promise<Segment[]>;
  loadRawSegments?: () => Promise<Segment[]>;
  loadVersionInfo: () => Promise<VersionInfo>;
  saveSegments: (segments: Segment[]) => Promise<unknown>;
  showDelete?: boolean;
  showFlags?: boolean;
  showSpeaker?: boolean;
  referenceSegments?: Segment[];
  referenceLabel?: string;
  speakers?: string[];
}

/** Distribute timestamps proportionally across segments based on text length. */
function estimateTimestamps(segments: Segment[], totalDuration: number): Segment[] {
  const totalChars = segments.reduce((sum, s) => sum + Math.max(s.text.length, 1), 0);
  let cursor = 0;
  return segments.map((seg) => {
    const weight = Math.max(seg.text.length, 1) / totalChars;
    const duration = weight * totalDuration;
    const start = Math.round(cursor * 10) / 10;
    cursor += duration;
    const end = Math.round(cursor * 10) / 10;
    return { ...seg, start, end };
  });
}

export default function SegmentEditor({
  editorKey,
  audioPath,
  episodeDuration,
  loadSegments,
  loadRawSegments,
  loadVersionInfo,
  saveSegments,
  showDelete = true,
  showFlags = true,
  showSpeaker = true,
  referenceSegments,
  referenceLabel = "Original",
  speakers: externalSpeakers,
}: SegmentEditorProps) {
  const queryClient = useQueryClient();

  const { data: segments } = useQuery({
    queryKey: [editorKey, "segments", audioPath],
    queryFn: loadSegments,
  });

  const { data: versionInfo } = useQuery({
    queryKey: [editorKey, "version-info", audioPath],
    queryFn: loadVersionInfo,
  });

  const editor = useSegments(segments ?? []);

  // Reset editor when source segments change
  useEffect(() => {
    if (segments) editor.reset(segments);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segments]);

  const saveMutation = useMutation({
    mutationFn: () => saveSegments(editor.editedSegments),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [editorKey, "segments", audioPath] });
      queryClient.invalidateQueries({ queryKey: [editorKey, "version-info", audioPath] });
    },
  });

  // Merge dialog state: when merging segments with different speakers
  const [mergeDialog, setMergeDialog] = useState<{
    index: number;
    speakers: [string, string];
  } | null>(null);

  const handleMerge = useCallback((originalIndex: number, currentSpeaker: string) => {
    const next = editor.getNextSegment(originalIndex);
    if (!next) return;
    if (next.speaker === currentSpeaker) {
      editor.mergeWithNext(originalIndex);
    } else {
      setMergeDialog({ index: originalIndex, speakers: [currentSpeaker, next.speaker] });
    }
  }, [editor]);

  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(20);
  const [speakerFilter, setSpeakerFilter] = useState("");
  const [showFlaggedOnly, setShowFlaggedOnly] = useState(false);
  const [showChangedOnly, setShowChangedOnly] = useState(false);
  const [densityThreshold, setDensityThreshold] = useState(2);
  const [maxDensityThreshold, setMaxDensityThreshold] = useState(50);
  const [searchQuery, setSearchQuery] = useState("");

  // Build list of speakers
  const speakers = useMemo(() => {
    if (externalSpeakers && externalSpeakers.length > 0) return externalSpeakers;
    if (!segments) return [];
    const set = new Set<string>();
    for (const seg of segments) {
      if (seg.speaker && seg.speaker !== "[BREAK]") set.add(seg.speaker);
    }
    return Array.from(set).sort();
  }, [segments, externalSpeakers]);

  // Lowercase search for case-insensitive matching
  const searchLower = searchQuery.toLowerCase().trim();

  const UNKNOWN_SPEAKERS = new Set(["UNKNOWN", "UNK", "None", "none", ""]);

  // Compute flagged based on density threshold sliders
  const isFlaggedSeg = (seg: Segment): boolean => {
    if (seg.speaker === "[BREAK]") return false;
    if (UNKNOWN_SPEAKERS.has(seg.speaker)) return true;
    if (seg.speaker === "[remove]") return true;
    const dur = seg.end - seg.start;
    if (dur > 0) {
      const density = seg.text.length / dur;
      if (density < densityThreshold) return true;
      if (density > maxDensityThreshold) return true;
    }
    return false;
  };

  // Map originalIndex → position in editedSegments (for positional reference matching).
  // After deletes, positions shift so references stay aligned.
  const origToEditedIdx = useMemo(() => {
    const map = new Map<number, number>();
    for (let i = 0; i < editor.originalIndices.length; i++) {
      map.set(editor.originalIndices[i], i);
    }
    return map;
  }, [editor.originalIndices]);

  // Check if a segment differs from its reference (positional match)
  const isChanged = (seg: Segment, origIdx: number) => {
    if (!referenceSegments || seg.speaker === "[BREAK]") return false;
    const editedIdx = origToEditedIdx.get(origIdx) ?? origIdx;
    const ref = referenceSegments[editedIdx];
    return ref != null && ref.text !== seg.text;
  };

  // Count changed segments
  const changedCount = useMemo(() => {
    if (!referenceSegments) return 0;
    let count = 0;
    for (let e = 0; e < editor.editedSegments.length; e++) {
      if (isChanged(editor.editedSegments[e], editor.originalIndices[e])) count++;
    }
    return count;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor.editedSegments, editor.originalIndices, referenceSegments]);

  // Filter editedSegments for display
  const displaySegments = useMemo(() => {
    const result: { segment: Segment; originalIndex: number; displayIndex: number }[] = [];
    if (!segments) return result;

    let idx = 0;
    for (let e = 0; e < editor.editedSegments.length; e++) {
      const seg = editor.editedSegments[e];
      const origIdx = editor.originalIndices[e];

      // Speaker filter
      if (speakerFilter && seg.speaker !== speakerFilter && seg.speaker !== "[BREAK]") continue;

      // Flagged filter (use client-side density threshold)
      if (showFlaggedOnly && showFlags && !isFlaggedSeg(seg) && seg.speaker !== "[BREAK]") continue;

      // Changed filter
      if (showChangedOnly && !isChanged(seg, origIdx) && seg.speaker !== "[BREAK]") continue;

      // Search filter
      if (searchLower && !seg.text.toLowerCase().includes(searchLower) && seg.speaker !== "[BREAK]") continue;

      result.push({ segment: seg, originalIndex: origIdx, displayIndex: idx });
      idx++;
    }

    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor.editedSegments, editor.originalIndices, segments, speakerFilter, showFlaggedOnly, showChangedOnly, showFlags, searchLower, referenceSegments, densityThreshold, maxDensityThreshold]);

  const totalPages = Math.ceil(displaySegments.length / pageSize);
  const pageSegments = displaySegments.slice(page * pageSize, (page + 1) * pageSize);

  // Flagged count based on current density thresholds
  const flaggedCount = useMemo(() => {
    return editor.editedSegments.filter(isFlaggedSeg).length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor.editedSegments, densityThreshold, maxDensityThreshold]);

  // Reset page when filters change
  useEffect(() => setPage(0), [speakerFilter, showFlaggedOnly, showChangedOnly, searchQuery]);

  // Clamp page when total pages shrinks (e.g. after deleting segments)
  useEffect(() => {
    if (totalPages > 0 && page >= totalPages) setPage(totalPages - 1);
  }, [totalPages, page]);

  // Track which segment is currently playing
  const storeAudioPath = useAppStore((s) => s.audioPath);
  const isPlayingThisFile = audioPath != null && storeAudioPath === audioPath;

  const [activeOrigIdx, setActiveOrigIdx] = useState<number | null>(null);

  useEffect(() => {
    if (!isPlayingThisFile) {
      setActiveOrigIdx(null);
      return;
    }
    // Subscribe to currentTime changes at ~4Hz to find active segment
    const interval = setInterval(() => {
      const t = useAppStore.getState().currentTime;
      if (!useAppStore.getState().isPlaying) return;
      // Find segment containing current time
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

  // Detect if segments are missing timestamps
  const missingTimestamps = useMemo(() => {
    if (!segments || segments.length === 0) return false;
    return segments.every((s) => s.start === 0 && s.end === 0);
  }, [segments]);

  const handleEstimateTimestamps = () => {
    if (!episodeDuration || !segments) return;
    const withTimestamps = estimateTimestamps(editor.editedSegments, episodeDuration);
    editor.reset(withTimestamps);
  };

  const handleLoadOriginal = loadRawSegments
    ? async () => {
        const raw = await loadRawSegments();
        editor.reset(raw);
      }
    : undefined;

  const handleLoadEdits = async () => {
    const data = await loadSegments();
    editor.reset(data);
  };

  // Ref for search input focus
  const searchRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Warn before leaving with unsaved changes
  useEffect(() => {
    if (!editor.isDirty) return;
    const handler = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [editor.isDirty]);

  // Keyboard shortcuts: Ctrl+S save, Ctrl+Z undo, Ctrl+F search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      if (e.key === "s") {
        e.preventDefault();
        if (editor.isDirty && !saveMutation.isPending) saveMutation.mutate();
      } else if (e.key === "z" && !e.shiftKey) {
        // Only handle undo if focus is NOT in a textarea (let native undo work there)
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

  // Auto-scroll to active (playing) segment
  useEffect(() => {
    if (activeOrigIdx == null || !scrollRef.current) return;
    const el = scrollRef.current.querySelector(`[data-orig-idx="${activeOrigIdx}"]`);
    if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeOrigIdx]);

  if (!segments) {
    return <div className="p-6 text-muted-foreground text-sm">Loading segments...</div>;
  }

  if (segments.length === 0) {
    return <div className="p-6 text-muted-foreground text-sm">No segments available.</div>;
  }

  return (
    <div className="flex flex-col h-full">
      <EditorToolbar
        totalSegments={editor.editedSegments.length}
        visibleCount={displaySegments.length}
        isDirty={editor.isDirty}
        versionInfo={versionInfo ?? null}
        flaggedCount={flaggedCount}
        deletedCount={editor.deletedCount}
        speakers={speakers}
        speakerFilter={speakerFilter}
        onSpeakerFilterChange={setSpeakerFilter}
        showFlaggedOnly={showFlaggedOnly}
        onFlaggedFilterChange={setShowFlaggedOnly}
        showChangedOnly={showChangedOnly}
        onChangedFilterChange={setShowChangedOnly}
        hasReference={!!referenceSegments}
        changedCount={changedCount}
        densityThreshold={densityThreshold}
        onDensityChange={setDensityThreshold}
        maxDensityThreshold={maxDensityThreshold}
        onMaxDensityChange={setMaxDensityThreshold}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        searchRef={searchRef}
        onSave={() => saveMutation.mutate()}
        onLoadOriginal={handleLoadOriginal}
        onLoadEdits={versionInfo?.has_validated ? handleLoadEdits : undefined}
        onUndo={editor.canUndo ? editor.undo : undefined}
        onEstimateTimestamps={missingTimestamps && episodeDuration ? handleEstimateTimestamps : undefined}
        onDeleteFlagged={() => {
          const indices = editor.editedSegments
            .map((seg, i) => isFlaggedSeg(seg) ? editor.originalIndices[i] : -1)
            .filter(i => i >= 0);
          for (const idx of indices) editor.deleteSegment(idx);
        }}
        isSaving={saveMutation.isPending}
      />

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {pageSegments.map(({ segment, originalIndex, displayIndex }) => {
          const isBreak = segment.speaker === "[BREAK]";
          const editedIdx = origToEditedIdx.get(originalIndex) ?? originalIndex;
          const ref = referenceSegments?.[editedIdx];
          return (
            <div key={`${editorKey}-${originalIndex}`} data-orig-idx={originalIndex}>
            <SegmentRow
              segment={segment}
              index={displayIndex}
              totalCount={displaySegments.length}
              isEdited={editor.isDirty && (originalIndex >= segments.length || segment !== segments[originalIndex])}
              isFlagged={showFlags ? isFlaggedSeg(segment) : false}
              isChanged={isChanged(segment, originalIndex)}
              isActive={activeOrigIdx === originalIndex}
              isBreak={isBreak}
              audioPath={audioPath}
              referenceText={ref && !isBreak ? ref.text : undefined}
              referenceLabel={referenceLabel}
              speakers={speakers}
              showDelete={showDelete}
              showSpeaker={showSpeaker && !isBreak}
              onTextChange={(text) => editor.updateText(originalIndex, text)}
              onSpeakerChange={(speaker) => editor.updateSpeaker(originalIndex, speaker)}
              onTimestampChange={(field, value) => editor.updateTimestamp(originalIndex, field, value)}
              onDelete={() => editor.deleteSegment(originalIndex)}
              onInsertBefore={() => editor.insertAfter(originalIndex - 1, { speaker: segment.speaker, text: "", start: segment.start, end: segment.start })}
              onInsertAfter={() => editor.insertAfter(originalIndex, { speaker: segment.speaker, text: "", start: segment.end, end: segment.end })}
              onMergeNext={() => handleMerge(originalIndex, segment.speaker)}
              onSplit={(cursorPos) => editor.splitAt(originalIndex, cursorPos)}
            />
            </div>
          );
        })}
      </div>

      {totalPages > 1 && (
        <Pagination
          page={page}
          totalPages={totalPages}
          pageSize={pageSize}
          onPageChange={setPage}
          onPageSizeChange={(s) => {
            setPageSize(s);
            setPage(0);
          }}
        />
      )}

      {/* Merge speaker dialog */}
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
