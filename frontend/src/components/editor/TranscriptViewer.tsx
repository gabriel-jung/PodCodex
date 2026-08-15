/**
 * TranscriptViewer — unified editable transcript.
 *
 * Combines the clean read-only layout (timestamp | speaker | text per row) with
 * inline editing. Every segment is directly editable; there is no separate
 * read/edit mode toggle.
 */

import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import type { Segment, VersionEntry } from "@/api/types";
import { saveExportFile } from "@/api/client";
import { usePlatform } from "@/platform";
import { invalidateSpeakerViews } from "@/api/cacheInvalidation";
import { queryKeys } from "@/api/queryKeys";
import { useAudioStore } from "@/stores";
import { useSegments } from "@/hooks/useSegments";
import { useSegmentFiltering, useFilteredSegments } from "@/hooks/useSegmentFiltering";
import { useFlagPatternsStore } from "@/stores/flagPatternsStore";
import { versionInfo, isEdited } from "@/lib/utils";

/** Sentinel values for the compare ("vs") picker. `REF_NONE` = no diff,
 *  `REF_DEFAULT` = original/reference segments passed by the parent panel.
 *  Anything else is a version id. */
const REF_NONE = "none";
const REF_DEFAULT = "default";
type RefChoice = typeof REF_NONE | typeof REF_DEFAULT | string;
import { BREAK_SPEAKER, isSoloDefaultSpeaker } from "@/lib/speakers";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import Pagination from "./Pagination";
import { PAGE_SIZE_ALL } from "@/lib/pagination";
import SpeakerStrip from "./SpeakerStrip";
import SegmentList, { type SegmentListHandle } from "./SegmentList";
import EditorToolbar from "./EditorToolbar";
import VersionControlBar from "./VersionControlBar";
import { Download, Save, CheckCheck } from "lucide-react";

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
  const platform = usePlatform();

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const formats: { label: string; ext: "txt" | "srt" | "vtt" }[] = [
    { label: "Plain Text", ext: "txt" },
    { label: "SRT Subtitles", ext: "srt" },
    { label: "WebVTT Subtitles", ext: "vtt" },
  ];

  const handleExport = (ext: "txt" | "srt" | "vtt") => {
    setOpen(false);
    return saveExportFile(platform, {
      audioPath,
      source,
      format: ext,
      defaultName: `${filename || "export"}.${ext}`,
    });
  };

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
          {formats.map(({ label, ext }) => (
            <button
              key={ext}
              type="button"
              onClick={() => handleExport(ext)}
              className="block w-full text-left px-3 py-1.5 text-xs hover:bg-accent transition"
            >
              {label}
            </button>
          ))}
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
  /** Optional broader list shown only in the compare ("vs") picker. Defaults
   *  to `loadVersions` when omitted. CorrectPanel uses this so the primary
   *  picker stays scoped to corrected versions while the compare picker can
   *  also show upstream transcript versions. */
  loadCompareVersions?: () => Promise<VersionEntry[]>;
  loadVersion?: (id: string, version?: VersionEntry) => Promise<Segment[]>;
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
  /** Initial state for the diff highlight toggle. Defaults to true (red/green
   *  word-level diff visible). Translate panels pass false so source-vs-target
   *  doesn't paint every line as "changed". */
  defaultShowDiff?: boolean;
  // Source label fallback when no versions
  sourceLabel?: string;
  /** Fires whenever the set of row-checkbox-selected segments changes.
   *  Emits the edited segments (post-rename, post-edit) corresponding to the
   *  currently checked rows, excluding [BREAK] markers. Callers use this to
   *  drive downstream scope filters (e.g. synthesis). */
  onSelectionChange?: (selected: Segment[]) => void;
  /** Fires after a version is saved (a real edit, a "Mark reviewed", or a
   *  "Save as latest"). Lets the panel react — e.g. offer to clear recorded
   *  LLM batch failures now that a new version supersedes them. */
  onSaved?: () => void;
  /** Forwarded to ``VersionControlBar`` to surface the verified-version star
   *  toggle in the bar above the segment list. See its prop docs. */
  verifiableStep?: "transcript" | "corrected";
  verifiedVersionId?: string | null;
  verifiedStepMatches?: boolean;
  onToggleVerified?: (targetId: string, isCurrentlyVerified: boolean) => void;
}

export default function TranscriptViewer({
  editorKey,
  audioPath,
  loadSegments,
  saveSegments,
  saveSpeakerMap,
  loadVersions,
  loadCompareVersions,
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
  defaultShowDiff = true,
  sourceLabel,
  onSelectionChange,
  onSaved,
  verifiableStep,
  verifiedVersionId,
  verifiedStepMatches,
  onToggleVerified,
}: TranscriptViewerProps) {
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

  const { data: compareVersionsExtra } = useQuery({
    queryKey: queryKeys.stepVersions(`${editorKey}__compare`, audioPath),
    queryFn: loadCompareVersions!,
    enabled: !!loadCompareVersions,
  });
  const compareVersions = compareVersionsExtra ?? versions ?? [];

  // ── Version selector state ────────────────────────────────────────────────

  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [expandedInfo, setExpandedInfo] = useState(false);
  const [showFlagSettings, setShowFlagSettings] = useState(false);
  const [showDiff, setShowDiff] = useState(defaultShowDiff);
  const customPatterns = useFlagPatternsStore((s) => s.patterns);
  const setFlagPatterns = useFlagPatternsStore((s) => s.setPatterns);
  const [patternDraft, setPatternDraft] = useState(customPatterns.join("\n"));
  // Re-sync draft from store when opening the popover, so external edits
  // (e.g. from SettingsPage) are reflected — but never overwrite an in-progress draft.
  useEffect(() => {
    if (showFlagSettings) setPatternDraft(useFlagPatternsStore.getState().patterns.join("\n"));
  }, [showFlagSettings]);

  const { data: versionSegments } = useQuery({
    queryKey: queryKeys.stepVersionSegments(editorKey, audioPath, selectedVersionId),
    queryFn: () => {
      const v = versions?.find((x) => x.id === selectedVersionId);
      return loadVersion!(selectedVersionId!, v);
    },
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
      // Row ids are reassigned by `editor.reset`; clear id-keyed UI state so
      // stale entries can't bleed into the fresh transcript.
      setSelectedIds(new Set());
      setDismissedFlags(new Set());
      setRecentlyEdited(new Set());
      // Row list is virtualized, so mount cost is O(visible rows) regardless
      // of total. Default to "All" and only fall back to a cap for truly
      // oversized transcripts (pagination still useful as a navigation aid).
      filters.setPageSize(sourceSegments.length < 2000 ? PAGE_SIZE_ALL : 500);
      filters.setPage(0);
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
      // Always write the segments: a dirty save persists the edits, a clean
      // save ("Mark reviewed") persists an unchanged copy whose provenance is
      // flagged manual_edit so the version reads as reviewed.
      const finalSegments = editor.editedSegments
        .filter((seg) => !pendingRemovals.has(seg.speaker))
        .map((seg) => {
          const renamed = pendingRenames[seg.speaker];
          return renamed && renamed !== seg.speaker ? { ...seg, speaker: renamed } : seg;
        });
      writes.push(saveSegments(finalSegments));
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
    // Saving a long transcript outlives the editor if the user navigates away
    // mid-save, so the invalidation runs at the cache level. Deliberately no
    // ["search"] / ["index"]: both read indexed chunks out of LanceDB, which
    // an editor save does not touch. They only change on a reindex, and the
    // stale-vs-index state already surfaces as the episode's `outdated` mark.
    meta: {
      invalidates: [
        queryKeys.stepSegments(editorKey, audioPath),
        queryKeys.stepVersions(editorKey, audioPath),
        queryKeys.episodesAll(),
        // An edited segment fans out to every cross-step view of this episode.
        queryKeys.allVersions(audioPath),
        queryKeys.bestSourceSegments(audioPath),
        queryKeys.speakerMap(audioPath),
        invalidateSpeakerViews,
      ],
    },
    onSuccess: () => {
      // The save is the new latest version — snap the picker back to "Latest"
      // so the editor tracks it (matters when an older version was promoted).
      setSelectedVersionId(null);
      onSaved?.();
    },
  });

  const deleteVersionMutation = useMutation({
    mutationFn: (id: string) => deleteVersion!(id),
    meta: {
      invalidates: [
        queryKeys.stepVersions(editorKey, audioPath),
        queryKeys.stepSegments(editorKey, audioPath),
        queryKeys.episodesAll(),
        queryKeys.allVersions(audioPath),
        queryKeys.bestSourceSegments(audioPath),
        // Deleting the canonical version shifts what the speaker views resolve.
        invalidateSpeakerViews,
      ],
    },
  });

  // ── Speaker list ──────────────────────────────────────────────────────────

  const speakers = useMemo(() => {
    const set = new Set<string>(externalSpeakers ?? []);
    for (const seg of sourceSegments ?? []) {
      if (seg.speaker && seg.speaker !== BREAK_SPEAKER) set.add(seg.speaker);
    }
    for (const name of addedSpeakers) set.add(name);
    for (const target of Object.values(pendingRenames)) set.add(target);
    for (const removed of pendingRemovals) set.delete(removed);
    return Array.from(set).sort();
  }, [sourceSegments, externalSpeakers, addedSpeakers, pendingRenames, pendingRemovals]);

  // A transcript with nothing but the default narrator (no diarization, or a
  // subtitle import without <v Speaker> tags) repeats one meaningless label on
  // every row. Fade it out at rest; the row still reveals it on hover so the
  // speaker can be named, which is how it stops being the default.
  //
  // Derived from the segments alone, not from `speakers`: that list also
  // carries the show's known speakers, which are picker options rather than
  // anyone actually speaking in this episode.
  const speakerMuted = useMemo(() => {
    const present = new Set<string>();
    for (const seg of sourceSegments ?? []) {
      if (seg.speaker !== BREAK_SPEAKER) present.add(seg.speaker);
    }
    return isSoloDefaultSpeaker([...present]);
  }, [sourceSegments]);

  // ── Merge dialog (when speakers differ) ──────────────────────────────────

  const [mergeDialog, setMergeDialog] = useState<{
    id: number;
    speakers: [string, string];
  } | null>(null);

  const editorGetNextSegment = editor.getNextSegment;
  const editorMergeWithNext = editor.mergeWithNext;
  const handleMerge = useCallback(
    (id: number, currentSpeaker: string) => {
      const next = editorGetNextSegment(id);
      if (!next) return;
      if (next.speaker === currentSpeaker) {
        editorMergeWithNext(id);
      } else {
        setMergeDialog({ id, speakers: [currentSpeaker, next.speaker] });
      }
    },
    [editorGetNextSegment, editorMergeWithNext],
  );

  // ── Compare-with selector ─────────────────────────────────────────────────

  const [refChoice, setRefChoice] = useState<RefChoice>(
    referenceSegments ? REF_DEFAULT : REF_NONE,
  );

  useEffect(() => {
    setRefChoice(referenceSegments ? REF_DEFAULT : REF_NONE);
  }, [referenceSegments]);

  const [versionRefSegments, setVersionRefSegments] = useState<Segment[] | null>(null);

  const handleRefChoiceChange = async (choice: RefChoice) => {
    setRefChoice(choice);
    if (choice === REF_DEFAULT || choice === REF_NONE) {
      setVersionRefSegments(null);
      return;
    }
    if (loadVersion) {
      try {
        const v = versions?.find((x) => x.id === choice);
        const data = await loadVersion(choice, v);
        setVersionRefSegments(data);
      } catch {
        setVersionRefSegments(null);
        setRefChoice(REF_NONE);
      }
    }
  };

  const effectiveReference =
    refChoice === REF_DEFAULT ? referenceSegments :
    refChoice === REF_NONE ? undefined :
    versionRefSegments ?? undefined;

  // Per-row reference index, keyed by row id. null = inserted this session
  // (no diff-reference counterpart). undefined = unknown row.
  const sourceIndexById = useMemo(() => {
    const map = new Map<number, number | null>();
    for (let i = 0; i < editor.allIds.length; i++) {
      map.set(editor.allIds[i], editor.allSourceIndices[i]);
    }
    return map;
  }, [editor.allIds, editor.allSourceIndices]);

  /** Reference segment for a row id, or undefined if the row is inserted
   *  (no counterpart in the diff reference) or no reference is selected. */
  const getRef = (id: number): Segment | undefined => {
    if (!effectiveReference) return undefined;
    const srcIdx = sourceIndexById.get(id);
    return srcIdx == null ? undefined : effectiveReference[srcIdx];
  };

  const isChanged = (seg: Segment, id: number) => {
    if (!effectiveReference || seg.speaker === BREAK_SPEAKER) return false;
    // Rows inserted this session (split / manual insert) have no source
    // counterpart. Count them as changed so the "show changed only" filter
    // keeps them visible.
    const srcIdx = sourceIndexById.get(id);
    if (srcIdx == null) return true;
    const ref = effectiveReference[srcIdx];
    return ref != null && ref.text !== seg.text;
  };

  const changedCount = useMemo(() => {
    if (!effectiveReference) return 0;
    let count = 0;
    for (let e = 0; e < editor.editedSegments.length; e++) {
      if (isChanged(editor.editedSegments[e], editor.ids[e])) count++;
    }
    return count;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor.editedSegments, editor.ids, effectiveReference]);

  const hasCompareOptions = !!referenceSegments || !!(versions && versions.length > 0);

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

  // Keyed by stable row id. Insert/split do not invalidate these entries.
  const [selectedIds, setSelectedIds] = useState<Set<number>>(() => new Set());

  const toggleSelect = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);

  // Per-id segment lookup. Built from the current edited list (non-deleted) so
  // bulkMerge can find positional adjacency and the selection-emit effect can
  // dereference each selected id.
  const editedSegments = editor.editedSegments;
  const editorIds = editor.ids;
  const segmentById = useMemo(() => {
    const map = new Map<number, { segment: Segment; position: number }>();
    for (let i = 0; i < editorIds.length; i++) {
      map.set(editorIds[i], { segment: editedSegments[i], position: i });
    }
    return map;
  }, [editedSegments, editorIds]);

  // Emit selection changes — downstream consumers (e.g. synthesis scope
  // filter) work on the edited segment payloads, not raw indices, so their
  // state survives version switches and pagination. Signature guard keeps
  // per-keystroke edits from re-emitting when the selection hasn't changed.
  const lastSelectionSigRef = useRef<string>("");
  useEffect(() => {
    if (!onSelectionChange) return;
    const out: Segment[] = [];
    const sigParts: string[] = [];
    for (const id of selectedIds) {
      const hit = segmentById.get(id);
      if (!hit || hit.segment.speaker === BREAK_SPEAKER) continue;
      out.push(hit.segment);
      sigParts.push(`${hit.segment.speaker}:${hit.segment.start}:${hit.segment.end}`);
    }
    const sig = sigParts.join("|");
    if (sig === lastSelectionSigRef.current) return;
    lastSelectionSigRef.current = sig;
    onSelectionChange(out);
  }, [selectedIds, segmentById, onSelectionChange]);

  const bulkDelete = useCallback(() => {
    // Order doesn't matter — ids stay valid across deletes.
    for (const id of selectedIds) editor.deleteSegment(id);
    clearSelection();
  }, [selectedIds, editor, clearSelection]);

  const bulkSpeaker = useCallback((speaker: string) => {
    for (const id of selectedIds) editor.updateSpeaker(id, speaker);
    setRecentlyEdited((prev) => {
      const next = new Set(prev);
      for (const id of selectedIds) next.add(id);
      return next;
    });
    clearSelection();
  }, [selectedIds, editor, clearSelection]);

  const bulkMerge = useCallback(() => {
    // Adjacency by current position, not by id (ids are not contiguous after
    // inserts). Sort selected ids by their position, then merge from the
    // bottom up so the merge index doesn't shift the rest.
    const positions = Array.from(selectedIds)
      .map((id) => ({ id, position: segmentById.get(id)?.position ?? -1 }))
      .filter((p) => p.position >= 0)
      .sort((a, b) => a.position - b.position);
    if (positions.length < 2) return;
    for (let i = positions.length - 1; i > 0; i--) {
      if (positions[i].position === positions[i - 1].position + 1) {
        editor.mergeWithNext(positions[i - 1].id);
      }
    }
    clearSelection();
  }, [selectedIds, segmentById, editor, clearSelection]);

  // ── Filtering / pagination ────────────────────────────────────────────────

  const filters = useSegmentFiltering();
  // Keyed by stable row id — survive insert/split unchanged.
  const [dismissedFlags, setDismissedFlags] = useState<Set<number>>(() => new Set());
  const [recentlyEdited, setRecentlyEdited] = useState<Set<number>>(() => new Set());

  const dismissFlag = useCallback((id: number) => {
    setDismissedFlags((prev) => new Set(prev).add(id));
  }, []);

  const setAnchorId = filters.setAnchorId;
  const markEdited = useCallback((id: number) => {
    setRecentlyEdited((prev) => {
      if (prev.has(id)) return prev;
      const next = new Set(prev);
      next.add(id);
      return next;
    });
    setAnchorId(id);
  }, [setAnchorId]);

  // Clear recentlyEdited on any view change — fresh filter pass should not
  // be subverted by stale sticky entries. With pageSize=All, page never
  // changes, so we also depend on filter toggles to ensure a clear happens.
  useEffect(() => {
    setRecentlyEdited((prev) => (prev.size === 0 ? prev : new Set()));
  }, [filters.page, filters.showFlaggedOnly, filters.showChangedOnly, filters.showRemovedOnly, filters.speakerFilter, filters.searchQuery]);

  const deletedSet = editor.deletedSet;
  const isPendingRemovalSeg = useCallback(
    (seg: Segment, id: number) =>
      pendingRemovals.has(seg.speaker) || deletedSet.has(id),
    [pendingRemovals, deletedSet],
  );

  const pendingRemovalCount = useMemo(() => {
    let n = 0;
    for (let i = 0; i < editor.allEditedSegments.length; i++) {
      if (isPendingRemovalSeg(editor.allEditedSegments[i], editor.allIds[i])) n++;
    }
    return n;
  }, [editor.allEditedSegments, editor.allIds, isPendingRemovalSeg]);

  // Stable row callbacks — keyed by id so each row's props stay reference-equal
  // across renders. Paired with React.memo on SegmentViewRow this drops
  // per-keystroke cost from O(visible rows) to O(1). Deps list the individual
  // editor methods (each wrapped in useCallback inside useSegments) so a fresh
  // editor return object doesn't invalidate every row.
  const editorUpdateText = editor.updateText;
  const editorUpdateSpeaker = editor.updateSpeaker;
  const editorUpdateTimestamp = editor.updateTimestamp;
  const editorDeleteSegment = editor.deleteSegment;
  const editorRestoreSegment = editor.restoreSegment;
  const editorInsertAfter = editor.insertAfter;
  const editorSplitAt = editor.splitAt;
  // insertBefore needs to know the id of the row preceding the target so it
  // can re-use insertAfter. We need a stale-free lookup (live rows, not a
  // snapshot in deps) so the callback identity stays stable.
  const segmentByIdRef = useRef(segmentById);
  const editorIdsRef = useRef(editor.ids);
  segmentByIdRef.current = segmentById;
  editorIdsRef.current = editor.ids;
  const handleRowTextChange = useCallback((id: number, text: string) => {
    editorUpdateText(id, text);
    markEdited(id);
  }, [editorUpdateText, markEdited]);
  const handleRowSpeakerChange = useCallback((id: number, speaker: string) => {
    editorUpdateSpeaker(id, speaker);
    markEdited(id);
  }, [editorUpdateSpeaker, markEdited]);
  const handleRowTimestampChange = useCallback(
    (id: number, field: "start" | "end", value: number) => {
      editorUpdateTimestamp(id, field, value);
      markEdited(id);
    },
    [editorUpdateTimestamp, markEdited],
  );
  const handleRowDelete = useCallback((id: number) => {
    editorDeleteSegment(id);
  }, [editorDeleteSegment]);
  const handleRowRestore = useCallback((id: number) => {
    editorRestoreSegment(id);
  }, [editorRestoreSegment]);
  const handleRowInsertBefore = useCallback((id: number, seg: Segment) => {
    // Find the row preceding `id`. If it's the first row, fall back to
    // inserting after itself with a duplicated seed and immediately swap
    // would be ugly — accept that "insert before the first row" inserts
    // after the first instead; the visible position difference is one slot.
    const ids = editorIdsRef.current;
    const pos = segmentByIdRef.current.get(id)?.position ?? -1;
    const beforeId = pos > 0 ? ids[pos - 1] : id;
    editorInsertAfter(beforeId, {
      speaker: seg.speaker,
      text: "",
      start: seg.start,
      end: seg.start,
      flagged: false,
    });
  }, [editorInsertAfter]);
  const handleRowInsertAfter = useCallback((id: number, seg: Segment) => {
    editorInsertAfter(id, {
      speaker: seg.speaker,
      text: "",
      start: seg.end,
      end: seg.end,
      flagged: false,
    });
  }, [editorInsertAfter]);
  // Resolve a staged SpeakerStrip rename so the split's new row inherits the
  // currently-displayed speaker name, not the raw base name. Without this,
  // splitting a row after a chip rename leaves the new row's base.speaker
  // pointing at the pre-rename name; any later chip change desyncs the halves.
  const pendingRenamesRef = useRef(pendingRenames);
  pendingRenamesRef.current = pendingRenames;
  const handleRowSplit = useCallback(
    (id: number, cursorPos: number, t?: number) => {
      const hit = segmentByIdRef.current.get(id);
      const baseSpeaker = hit?.segment.speaker;
      const renamed = baseSpeaker ? pendingRenamesRef.current[baseSpeaker] : undefined;
      const resolved = renamed && renamed !== baseSpeaker ? renamed : undefined;
      editorSplitAt(id, cursorPos, t, resolved);
    },
    [editorSplitAt],
  );

  const { displaySegments, pageSegments, totalPages, flaggedCount } = useFilteredSegments(
    editor.allEditedSegments,
    editor.allIds,
    filters,
    { dismissedFlags, isChanged, recentlyEdited, customPatterns, isPendingRemoval: isPendingRemovalSeg },
  );

  useEffect(() => {
    if (pageSegments.length > 0) {
      filters.setAnchorId(pageSegments[0].id);
    }
  }, [pageSegments]); // eslint-disable-line react-hooks/exhaustive-deps

  // SegmentList owns its scroll container and the virtualizer; this ref
  // exposes scrollToId so the parent can drive jumps without holding the
  // virtualizer itself.
  const listRef = useRef<SegmentListHandle | null>(null);

  // ── Active segment tracking ───────────────────────────────────────────────

  const storeAudioPath = useAudioStore((s) => s.audioPath);
  const storeIsPlaying = useAudioStore((s) => s.isPlaying);
  const isPlayingThisFile = audioPath != null && storeAudioPath === audioPath;
  const [activeId, setActiveId] = useState<number | null>(null);

  const editedSegmentsRef = useRef(editor.editedSegments);
  editedSegmentsRef.current = editor.editedSegments;
  const idsRef = useRef(editor.ids);
  idsRef.current = editor.ids;

  // Drop activeId only when the player jumps to a different file. While the
  // current track is paused we keep the last activeId so the Now-playing
  // toolbar button still has somewhere to re-center to.
  useEffect(() => {
    if (audioPath == null || storeAudioPath !== audioPath) {
      setActiveId(null);
    }
  }, [audioPath, storeAudioPath]);

  useEffect(() => {
    if (!isPlayingThisFile) return;
    const interval = setInterval(() => {
      const t = useAudioStore.getState().currentTime;
      if (!useAudioStore.getState().isPlaying) return;
      const segs = editedSegmentsRef.current;
      const ids = idsRef.current;
      for (let e = segs.length - 1; e >= 0; e--) {
        const seg = segs[e];
        if (seg.start <= t && t < seg.end) {
          const id = ids[e];
          setActiveId((prev) => (prev === id ? prev : id));
          return;
        }
      }
      setActiveId((prev) => (prev == null ? prev : null));
    }, 250);
    return () => clearInterval(interval);
  }, [isPlayingThisFile]);

  const scrollToId = useCallback(
    (id: number, behavior: ScrollBehavior = "smooth") => {
      return listRef.current?.scrollToId(id, behavior) ?? false;
    },
    [],
  );

  // Auto-follow: while ON, the list scrolls to the active segment as
  // playback advances. Any user-initiated scroll (wheel/touchmove) flips it
  // OFF so the reader can browse without being yanked back. "Now playing"
  // toolbar button flips it back ON and re-centers.
  const [followMode, setFollowMode] = useState(true);
  const prevAudioPathRef = useRef(audioPath);
  if (prevAudioPathRef.current !== audioPath) {
    prevAudioPathRef.current = audioPath;
    // Re-engage follow on track change so the new transcript opens aligned.
    setFollowMode(true);
  }
  // Set by jumpToActive so the follow effect's first run after the toggle
  // doesn't re-fire scrollToId on top of the explicit call.
  const suppressNextFollowScrollRef = useRef(false);
  useEffect(() => {
    if (!followMode || activeId == null) return;
    if (suppressNextFollowScrollRef.current) {
      suppressNextFollowScrollRef.current = false;
      return;
    }
    // Skip when the user is typing inside the list: an active textarea/input
    // means they're editing this row (or a nearby one) and the smooth-scroll
    // would yank the caret offscreen mid-keystroke.
    const focusTag = (document.activeElement as HTMLElement | null)?.tagName;
    const focusEditable =
      focusTag === "TEXTAREA" ||
      focusTag === "INPUT" ||
      (document.activeElement as HTMLElement | null)?.isContentEditable;
    if (focusEditable) return;
    // 'auto' avoids stacked smooth-scroll animations on the 250ms activeId
    // polling tick. 'smooth' is reserved for the explicit jumpToActive click.
    scrollToId(activeId, "auto");
  }, [followMode, activeId, scrollToId]);

  const handleUserScroll = useCallback(() => {
    setFollowMode((cur) => (cur ? false : cur));
  }, []);

  const jumpToActive = () => {
    if (!followMode) {
      // Effect would scroll on its own once followMode commits — but we want
      // 'smooth' here (explicit user gesture), so do it manually and ask the
      // effect to skip its first post-toggle run.
      suppressNextFollowScrollRef.current = true;
      setFollowMode(true);
    }
    if (activeId != null) scrollToId(activeId, "smooth");
  };

  // Cross-page jump: SpeakerStrip excerpts ask to locate a segment in the
  // editor. If a page flip is needed, the effect picks up the scroll once
  // pageSegments contains the target.
  const pendingJumpRef = useRef<number | null>(null);
  const jumpToSegmentById = useCallback((id: number) => {
    const pos = displaySegments.findIndex((d) => d.id === id);
    if (pos < 0) return;
    const targetPage = Math.floor(pos / filters.pageSize);
    filters.setAnchorId(id);
    if (targetPage !== filters.page) {
      filters.setPage(targetPage);
      pendingJumpRef.current = id;
    } else {
      scrollToId(id);
    }
  }, [displaySegments, filters, scrollToId]);

  useEffect(() => {
    const target = pendingJumpRef.current;
    if (target == null) return;
    // One-shot: clear regardless so a missing target (filtered out, etc.)
    // can't trigger a stale jump on a later unrelated pageSegments change.
    pendingJumpRef.current = null;
    scrollToId(target);
  }, [pageSegments, scrollToId]);

  // When search empties, the visible list expands but the scroll container's
  // pixel offset is preserved by the DOM — so the user lands at the top of
  // the now-bigger list instead of next to the segment they were viewing.
  // Pin to whichever segment was the anchor when search cleared.
  const prevSearchRef = useRef(filters.searchQuery);
  useEffect(() => {
    const prev = prevSearchRef.current;
    prevSearchRef.current = filters.searchQuery;
    if (prev !== "" && filters.searchQuery === "") {
      // filters.anchorId in this effect is the pre-clear value; the per-page
      // anchor reset for the expanded list is queued for the next render.
      const id = filters.anchorId;
      if (id != null) scrollToId(id, "auto");
    }
  }, [filters.searchQuery, filters.anchorId, scrollToId]);

  // SpeakerStrip excerpts identify segments by their position in
  // `sourceSegments` — the canonical version that drives speaker computation.
  // Map that source position back to the live row id so jumps work after
  // edits/inserts.
  const idBySourceIndex = useMemo(() => {
    const map = new Map<number, number>();
    for (let i = 0; i < editor.allIds.length; i++) {
      const src = editor.allSourceIndices[i];
      if (src != null) map.set(src, editor.allIds[i]);
    }
    return map;
  }, [editor.allIds, editor.allSourceIndices]);

  const handleStripJump = useCallback((sourceIdx: number) => {
    const id = idBySourceIndex.get(sourceIdx);
    if (id != null) jumpToSegmentById(id);
  }, [idBySourceIndex, jumpToSegmentById]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────

  const searchRef = useRef<HTMLInputElement>(null);

  const isDirty = editor.isDirty || hasPendingStripChanges;
  // When nothing is dirty the Save button instead re-saves an unchanged copy.
  // Showing the latest version → "Mark reviewed" (flag it edited); showing an
  // older version → "Save as latest" (re-stamps it as the newest version, the
  // way the user promotes a version by hand). Disabled only when the latest
  // is already on screen and already edited — nothing left to do.
  const latest = versions && versions.length > 0 ? versions[0] : undefined;
  const showingLatest = selectedVersionId == null || selectedVersionId === latest?.id;
  const canMarkReviewed = !isDirty && !(showingLatest && !!latest && isEdited(latest));
  const saveButton = isDirty
    ? { label: "Save", title: "Save (Cmd+S)", Icon: Save }
    : canMarkReviewed
      ? {
          label: showingLatest ? "Mark reviewed" : "Save as latest",
          title: showingLatest
            ? "Mark this version as reviewed without changes (Cmd+S)"
            : "Re-save this version so it becomes the latest (Cmd+S)",
          Icon: CheckCheck,
        }
      : { label: "Reviewed", title: "Latest version is already marked as reviewed", Icon: CheckCheck };

  useEffect(() => {
    if (!isDirty) return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [isDirty]);

  const canUndo = editor.canUndo;
  const undo = editor.undo;
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      if (e.key === "s") {
        e.preventDefault();
        if ((isDirty || canMarkReviewed) && !saveMutation.isPending) saveMutation.mutate();
      } else if (e.key === "z" && !e.shiftKey) {
        if ((e.target as HTMLElement)?.tagName !== "TEXTAREA" && canUndo) {
          e.preventDefault();
          undo();
        }
      } else if (e.key === "f") {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isDirty, canMarkReviewed, canUndo, undo, saveMutation]);

  const compareExtras = useMemo(
    () => [
      { value: REF_NONE, label: "None" },
      ...(referenceSegments ? [{ value: REF_DEFAULT, label: referenceLabel ?? "Original" }] : []),
    ],
    [referenceSegments, referenceLabel],
  );

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
      <div className="px-4 py-2 border-b border-border space-y-1">
        <VersionControlBar
          versions={versions}
          selectedVersionId={selectedVersionId}
          onSelectVersion={setSelectedVersionId}
          hasCompareOptions={hasCompareOptions}
          compareVersions={compareVersions}
          refChoice={refChoice}
          onRefChoiceChange={handleRefChoiceChange}
          compareExtras={compareExtras}
          refNoneSentinel={REF_NONE}
          sourceLabel={sourceLabel}
          infoItems={infoItems}
          expandedInfo={expandedInfo}
          setExpandedInfo={setExpandedInfo}
          onDeleteVersion={deleteVersion
            ? (id) => {
                deleteVersionMutation.mutate(id);
                setSelectedVersionId(null);
              }
            : undefined}
          verifiableStep={verifiableStep}
          verifiedVersionId={verifiedVersionId}
          verifiedStepMatches={verifiedStepMatches}
          onToggleVerified={onToggleVerified}
        />

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
            onJumpToSegment={handleStripJump}
          />
        )}
      </div>

      <EditorToolbar
        filters={filters}
        searchRef={searchRef}
        selectedIds={selectedIds}
        setSelectedIds={setSelectedIds}
        clearSelection={clearSelection}
        bulkSpeaker={bulkSpeaker}
        bulkMerge={bulkMerge}
        bulkDelete={bulkDelete}
        pageSegments={pageSegments}
        speakers={speakers}
        flaggedCount={flaggedCount}
        pendingRemovalCount={pendingRemovalCount}
        changedCount={changedCount}
        showFlags={showFlags}
        showFlagSettings={showFlagSettings}
        setShowFlagSettings={setShowFlagSettings}
        patternDraft={patternDraft}
        setPatternDraft={setPatternDraft}
        setFlagPatterns={setFlagPatterns}
        hasReference={effectiveReference != null}
        showDiff={showDiff}
        setShowDiff={setShowDiff}
        isPlayingThisFile={isPlayingThisFile}
        activeId={activeId}
        jumpToActive={jumpToActive}
        followMode={followMode}
        canUndo={editor.canUndo}
        undo={editor.undo}
        exportSlot={exportSource && audioPath
          ? <ExportDropdown audioPath={audioPath} source={exportSource} filename={exportFilename} />
          : null}
        isDirty={isDirty}
        canMarkReviewed={canMarkReviewed}
        saveButton={saveButton}
        saveMutation={saveMutation}
      />

      <SegmentList
        ref={listRef}
        editorKey={editorKey}
        pageSegments={pageSegments}
        filterActive={
          !!filters.speakerFilter ||
          filters.showFlaggedOnly ||
          filters.showChangedOnly ||
          filters.showRemovedOnly ||
          filters.searchQuery.trim() !== ""
        }
        speakers={speakers}
        audioPath={audioPath}
        showFlags={showFlags}
        showSpeaker={showSpeaker}
        speakerMuted={speakerMuted}
        showDelete={showDelete}
        showDiff={showDiff}
        densityThreshold={filters.densityThreshold}
        maxDensityThreshold={filters.maxDensityThreshold}
        customPatterns={customPatterns}
        activeId={activeId}
        storeIsPlaying={storeIsPlaying}
        pendingRenames={pendingRenames}
        pendingRemovals={pendingRemovals}
        dismissedFlags={dismissedFlags}
        deletedSet={deletedSet}
        textEditedIds={editor.textEditedIds}
        selectedIds={selectedIds}
        getRef={getRef}
        isChanged={isChanged}
        onToggleSelect={toggleSelect}
        onTextChange={handleRowTextChange}
        onSpeakerChange={handleRowSpeakerChange}
        onTimestampChange={handleRowTimestampChange}
        onDelete={handleRowDelete}
        onRestore={handleRowRestore}
        onDismissFlag={dismissFlag}
        onInsertBefore={handleRowInsertBefore}
        onInsertAfter={handleRowInsertAfter}
        onMergeNext={handleMerge}
        onSplit={handleRowSplit}
        onUserScroll={handleUserScroll}
      />

      {/* ── Pagination ── */}
      {displaySegments.length > 10 && (
        <Pagination
          page={filters.page}
          totalPages={totalPages}
          pageSize={filters.pageSize}
          onPageChange={filters.setPage}
          onPageSizeChange={(s) => {
            const anchor = filters.anchorId;
            filters.setPageSize(s);
            if (anchor != null) {
              const pos = displaySegments.findIndex((d) => d.id === anchor);
              filters.setPage(pos >= 0 ? Math.floor(pos / s) : 0);
            } else {
              filters.setPage(0);
            }
          }}
        />
      )}

      {/* ── Merge speaker dialog ── */}
      {mergeDialog && (
        <Dialog open onOpenChange={(o) => { if (!o) setMergeDialog(null); }}>
          <DialogContent className="sm:max-w-xs bg-popover">
            <DialogHeader>
              <DialogTitle>Merge segments</DialogTitle>
              <DialogDescription>Which speaker for the merged segment?</DialogDescription>
            </DialogHeader>
            <div className="flex flex-col gap-2">
              {mergeDialog.speakers.map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    editor.mergeWithNext(mergeDialog.id, s);
                    setMergeDialog(null);
                  }}
                  className="px-3 py-2 text-sm rounded-md border border-border bg-secondary hover:bg-accent transition text-left"
                >
                  {s}
                </button>
              ))}
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
