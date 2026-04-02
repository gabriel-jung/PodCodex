import { useMemo, useState, useEffect, useCallback, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Segment, VersionEntry } from "@/api/types";
import { useSegments } from "@/hooks/useSegments";
import { useSegmentFiltering, useFilteredSegments, flagReason } from "@/hooks/useSegmentFiltering";
import { versionDate, versionLabel } from "@/lib/utils";
import { useAudioStore } from "@/stores";
import EditorToolbar from "./EditorToolbar";
import SegmentRow from "./SegmentRow";
import Pagination from "./Pagination";

interface SegmentEditorProps {
  editorKey: string;
  audioPath?: string;
  episodeDuration?: number;
  loadSegments: () => Promise<Segment[]>;
  saveSegments: (segments: Segment[]) => Promise<unknown>;
  showDelete?: boolean;
  showFlags?: boolean;
  showSpeaker?: boolean;
  referenceSegments?: Segment[];
  referenceLabel?: string;
  speakers?: string[];
  loadVersions?: () => Promise<VersionEntry[]>;
  loadVersion?: (id: string) => Promise<Segment[]>;
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
  saveSegments,
  showDelete = true,
  showFlags = true,
  showSpeaker = true,
  referenceSegments,
  referenceLabel = "Original",
  speakers: externalSpeakers,
  loadVersions,
  loadVersion,
}: SegmentEditorProps) {
  const queryClient = useQueryClient();

  const { data: segments } = useQuery({
    queryKey: [editorKey, "segments", audioPath],
    queryFn: loadSegments,
  });

  const { data: versions } = useQuery({
    queryKey: [editorKey, "versions", audioPath],
    queryFn: loadVersions!,
    enabled: !!loadVersions,
  });

  const editor = useSegments(segments ?? []);

  // Reset editor when source segments change
  useEffect(() => {
    if (segments) {
      editor.reset(segments);
      // Push to store so AudioBar can show current segment text
      if (audioPath) {
        useAudioStore.getState().setAudioSegments(audioPath, segments.map((s) => ({
          start: s.start, end: s.end, speaker: s.speaker, text: s.text,
        })));
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segments]);

  const saveMutation = useMutation({
    mutationFn: () => saveSegments(editor.editedSegments),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [editorKey, "segments", audioPath] });
      queryClient.invalidateQueries({ queryKey: [editorKey, "versions", audioPath] });
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

  const filters = useSegmentFiltering();

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

  // Set of original indices whose flags have been manually dismissed
  const [dismissedFlags, setDismissedFlags] = useState<Set<number>>(new Set());

  const isFlaggedSeg = (seg: Segment, origIdx?: number): boolean => {
    if (origIdx != null && dismissedFlags.has(origIdx)) return false;
    return flagReason(seg, filters.densityThreshold, filters.maxDensityThreshold) !== null;
  };

  const dismissFlag = (origIdx: number) => {
    setDismissedFlags((prev) => new Set(prev).add(origIdx));
  };

  // Map originalIndex → position in editedSegments (for positional reference matching).
  const origToEditedIdx = useMemo(() => {
    const map = new Map<number, number>();
    for (let i = 0; i < editor.originalIndices.length; i++) {
      map.set(editor.originalIndices[i], i);
    }
    return map;
  }, [editor.originalIndices]);

  // ── Compare-with selector ────────────────────────────
  // "default" = parent-supplied referenceSegments, "none" = no diff, version id = load that version
  const [refChoice, setRefChoice] = useState<"default" | "none" | string>(
    referenceSegments ? "default" : "none",
  );
  // Reset choice when the parent-supplied reference changes
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
    // Load a version as reference
    if (loadVersion) {
      try {
        const data = await loadVersion(choice);
        setVersionRefSegments(data);
        const v = versions?.find((ver) => ver.id === choice);
        setVersionRefLabel(v ? `Version ${new Date(v.timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric" })}` : "Version");
      } catch {
        setVersionRefSegments(null);
        setVersionRefLabel("");
        setRefChoice("none");
      }
    }
  };

  const effectiveReference = refChoice === "default" ? referenceSegments
    : refChoice === "none" ? undefined
    : versionRefSegments ?? undefined;
  const effectiveRefLabel = refChoice === "default" ? referenceLabel
    : refChoice === "none" ? ""
    : versionRefLabel;

  // Build adjusted reference lookup.
  // Both edited and reference segments started as parallel arrays. When an
  // edited segment is deleted, the corresponding reference position should
  // also be consumed (auto-skipped).
  //
  // Strategy: build a list of "active" (non-deleted) reference
  // indices, then assign them 1:1 to edited segments.
  const refMapping = useMemo(() => {
    if (!effectiveReference) return null;

    // Detect which original positions were deleted by finding gaps in originalIndices
    const deletedOriginals = new Set<number>();
    const survivingSet = new Set(editor.originalIndices);
    const maxOrig = editor.originalIndices.length > 0
      ? Math.max(...editor.originalIndices, effectiveReference.length - 1)
      : effectiveReference.length - 1;
    for (let i = 0; i <= maxOrig; i++) {
      if (!survivingSet.has(i)) deletedOriginals.add(i);
    }

    // Active reference indices: skip auto-skipped (deleted)
    const activeRefIndices: number[] = [];
    for (let i = 0; i < effectiveReference.length; i++) {
      if (!deletedOriginals.has(i)) {
        activeRefIndices.push(i);
      }
    }

    // Map editedIdx → refIdx (1:1 positional)
    const map = new Map<number, number>();
    for (let e = 0; e < editor.editedSegments.length && e < activeRefIndices.length; e++) {
      map.set(e, activeRefIndices[e]);
    }
    return map;
  }, [effectiveReference, editor.editedSegments, editor.originalIndices]);

  // Get reference segment for a given edited position
  const getRef = (editedIdx: number) => {
    if (!effectiveReference || !refMapping) return undefined;
    const refIdx = refMapping.get(editedIdx);
    return refIdx != null ? effectiveReference[refIdx] : undefined;
  };

  const getRefIdx = (editedIdx: number) => {
    return refMapping?.get(editedIdx);
  };

  // Check if a segment differs from its reference (positional — shifts with deletes)
  const isChanged = (seg: Segment, origIdx: number) => {
    if (!effectiveReference || seg.speaker === "[BREAK]") return false;
    const editedIdx = origToEditedIdx.get(origIdx) ?? origIdx;
    const ref = getRef(editedIdx);
    return ref != null && ref.text !== seg.text;
  };

  // Count changed segments
  const changedCount = useMemo(() => {
    if (!effectiveReference) return 0;
    let count = 0;
    for (let e = 0; e < editor.editedSegments.length; e++) {
      if (isChanged(editor.editedSegments[e], editor.originalIndices[e])) count++;
    }
    return count;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor.editedSegments, editor.originalIndices, effectiveReference]);

  const { displaySegments, pageSegments, totalPages, flaggedCount } = useFilteredSegments(
    editor.editedSegments,
    editor.originalIndices,
    filters,
    { dismissedFlags, isChanged },
  );

  // Keep anchor in sync with what's currently visible
  useEffect(() => {
    if (pageSegments.length > 0) {
      filters.setAnchorOrigIdx(pageSegments[0].originalIndex);
    }
  }, [pageSegments]); // eslint-disable-line react-hooks/exhaustive-deps

  // Scroll to top of list when page changes
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
  }, [filters.page]);

  // Track which segment is currently playing
  const storeAudioPath = useAudioStore((s) => s.audioPath);
  const isPlayingThisFile = audioPath != null && storeAudioPath === audioPath;

  const [activeOrigIdx, setActiveOrigIdx] = useState<number | null>(null);

  useEffect(() => {
    if (!isPlayingThisFile) {
      setActiveOrigIdx(null);
      return;
    }
    // Subscribe to currentTime changes at ~4Hz to find active segment
    const interval = setInterval(() => {
      const t = useAudioStore.getState().currentTime;
      if (!useAudioStore.getState().isPlaying) return;
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

  const handleLoadVersion = loadVersion
    ? async (id: string) => {
        const data = await loadVersion(id);
        editor.reset(data);
      }
    : undefined;

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
        flaggedCount={flaggedCount}
        deletedCount={editor.deletedCount}
        speakers={speakers}
        speakerFilter={filters.speakerFilter}
        onSpeakerFilterChange={filters.setSpeakerFilter}
        showFlaggedOnly={filters.showFlaggedOnly}
        onFlaggedFilterChange={filters.setShowFlaggedOnly}
        showChangedOnly={filters.showChangedOnly}
        onChangedFilterChange={filters.setShowChangedOnly}
        hasReference={!!effectiveReference}
        changedCount={changedCount}
        densityThreshold={filters.densityThreshold}
        onDensityChange={filters.setDensityThreshold}
        maxDensityThreshold={filters.maxDensityThreshold}
        onMaxDensityChange={filters.setMaxDensityThreshold}
        searchQuery={filters.searchQuery}
        onSearchChange={filters.setSearchQuery}
        searchRef={searchRef}
        onSave={() => saveMutation.mutate()}
        onUndo={editor.canUndo ? editor.undo : undefined}
        onEstimateTimestamps={missingTimestamps && episodeDuration ? handleEstimateTimestamps : undefined}
        onDeleteFlagged={() => {
          const indices = editor.editedSegments
            .map((seg, i) => isFlaggedSeg(seg, editor.originalIndices[i]) ? editor.originalIndices[i] : -1)
            .filter(i => i >= 0);
          for (const idx of indices) editor.deleteSegment(idx);
        }}
        isSaving={saveMutation.isPending}
        versions={versions}
        onLoadVersion={handleLoadVersion}
        audioPath={audioPath}
        exportSource={
          editorKey === "transcript" ? "transcript"
            : editorKey === "polish" ? "polished"
            : editorKey.startsWith("translate-") ? editorKey.replace("translate-", "")
            : undefined
        }
      />

      {/* Compare-with selector */}
      {(referenceSegments || (versions && versions.length > 0)) && (
        <div className="px-4 py-1.5 border-b border-border/50 flex items-center gap-2 text-xs">
          <span className="text-muted-foreground">Compare with:</span>
          <select
            value={refChoice}
            onChange={(e) => handleRefChoiceChange(e.target.value)}
            className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-xs"
          >
            <option value="none">None</option>
            {referenceSegments && (
              <option value="default">{referenceLabel}</option>
            )}
            {versions && versions.map((v) => (
              <option key={v.id} value={v.id}>
                {versionDate(v)} — {versionLabel(v)} ({v.segment_count} seg)
              </option>
            ))}
          </select>
        </div>
      )}

      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {pageSegments.map(({ segment, originalIndex, displayIndex }) => {
          const isBreak = segment.speaker === "[BREAK]";
          const editedIdx = origToEditedIdx.get(originalIndex) ?? originalIndex;
          const ref = getRef(editedIdx);
          const refIdx = getRefIdx(editedIdx);
          return (
            <div key={`${editorKey}-${originalIndex}`} data-orig-idx={originalIndex}>
            <SegmentRow
              segment={segment}
              index={originalIndex}
              totalCount={editor.editedSegments.length}
              isEdited={editor.isDirty && (originalIndex >= segments.length || segment !== segments[originalIndex])}
              isFlagged={showFlags ? isFlaggedSeg(segment, originalIndex) : false}
              flagReason={showFlags ? flagReason(segment, filters.densityThreshold, filters.maxDensityThreshold) : null}
              onDismissFlag={showFlags ? () => dismissFlag(originalIndex) : undefined}
              isChanged={isChanged(segment, originalIndex)}
              isActive={activeOrigIdx === originalIndex}
              isBreak={isBreak}
              audioPath={audioPath}
              referenceText={ref && !isBreak ? ref.text : undefined}
              referenceLabel={effectiveRefLabel}
              referenceIndex={refIdx}
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
