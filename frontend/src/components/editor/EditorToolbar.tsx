/**
 * EditorToolbar — action bar above the segment list.
 *
 * Renders: select-all, search, speaker filter, flag chip + settings popover,
 * pending-removal chip, comparison group (diff toggle + changed chip),
 * now-playing jump, undo, export, save; plus the selection action bar
 * (set-speaker / merge / delete) when rows are selected.
 *
 * State lives in the parent TranscriptViewer; this component is a thin
 * presentational layer driven by props.
 */

import { type RefObject, useMemo } from "react";
import type { UseMutationResult } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import type { FilterState } from "@/hooks/useSegmentFiltering";
import type { SegmentListItem } from "./SegmentList";
import { BREAK_SPEAKER } from "@/lib/speakers";
import { selectClass } from "@/lib/utils";
import {
  Undo2,
  Trash2,
  Merge,
  Search,
  X,
  Diff,
  Filter,
  AlertTriangle,
  SlidersHorizontal,
  Locate,
  Eye,
  EyeOff,
  type LucideIcon,
} from "lucide-react";

// Active classes are full literal strings so Tailwind keeps them; a templated
// `bg-${color}/15` would be purged at build.
const CHIP_ACTIVE: Record<"warning" | "info" | "destructive", string> = {
  warning: "bg-warning/15 text-warning",
  info: "bg-info/15 text-info",
  destructive: "bg-destructive/15 text-destructive",
};

/** Toolbar toggle showing an icon + count. Active = filled tint, inactive =
 *  quiet ghost. Used for flagged / changed / pending-removal filters. */
function FilterChip({
  active,
  color,
  icon: Icon,
  count,
  title,
  onClick,
}: {
  active: boolean;
  color: "warning" | "info" | "destructive";
  icon: LucideIcon;
  count: number;
  title: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`flex items-center gap-1 text-xs px-1.5 py-0.5 rounded transition ${
        active ? CHIP_ACTIVE[color] : "text-muted-foreground hover:text-foreground hover:bg-accent"
      }`}
    >
      <Icon className="w-3 h-3" />
      {count}
    </button>
  );
}

export interface SaveButtonSpec {
  label: string;
  title: string;
  Icon: LucideIcon;
}

export interface EditorToolbarProps {
  // Filter / pagination
  filters: FilterState;
  searchRef: RefObject<HTMLInputElement | null>;
  // Selection
  selectedIds: Set<number>;
  setSelectedIds: (s: Set<number>) => void;
  clearSelection: () => void;
  bulkSpeaker: (speaker: string) => void;
  bulkMerge: () => void;
  bulkDelete: () => void;
  // Source / counts
  pageSegments: SegmentListItem[];
  speakers: string[];
  flaggedCount: number;
  pendingRemovalCount: number;
  changedCount: number;
  // Flag rules
  showFlags: boolean;
  showFlagSettings: boolean;
  setShowFlagSettings: (v: boolean) => void;
  patternDraft: string;
  setPatternDraft: (s: string) => void;
  setFlagPatterns: (patterns: string[]) => void;
  // Diff comparison
  hasReference: boolean;
  showDiff: boolean;
  setShowDiff: (fn: (v: boolean) => boolean) => void;
  // Playback / scroll
  isPlayingThisFile: boolean;
  activeId: number | null;
  jumpToActive: () => void;
  /** True when the editor is auto-scrolling to the active segment. The
   *  Now-playing button shows filled when ON, ghost when OFF. */
  followMode: boolean;
  // Editor state
  canUndo: boolean;
  undo: () => void;
  // Export
  exportSlot: React.ReactNode;
  // Save
  isDirty: boolean;
  canMarkReviewed: boolean;
  saveButton: SaveButtonSpec;
  saveMutation: UseMutationResult<unknown, Error, void, unknown>;
}

export default function EditorToolbar({
  filters,
  searchRef,
  selectedIds,
  setSelectedIds,
  clearSelection,
  bulkSpeaker,
  bulkMerge,
  bulkDelete,
  pageSegments,
  speakers,
  flaggedCount,
  pendingRemovalCount,
  changedCount,
  showFlags,
  showFlagSettings,
  setShowFlagSettings,
  patternDraft,
  setPatternDraft,
  setFlagPatterns,
  hasReference,
  showDiff,
  setShowDiff,
  isPlayingThisFile,
  activeId,
  jumpToActive,
  followMode,
  canUndo,
  undo,
  exportSlot,
  isDirty,
  canMarkReviewed,
  saveButton,
  saveMutation,
}: EditorToolbarProps) {
  const nonBreakPageSegments = useMemo(
    () => pageSegments.filter((s) => s.segment.speaker !== BREAK_SPEAKER),
    [pageSegments],
  );

  return (
    <div className="px-4 py-1.5 border-b border-border space-y-1">
      <div className="flex items-center gap-2 flex-wrap">
        <input
          type="checkbox"
          checked={selectedIds.size > 0 && selectedIds.size === nonBreakPageSegments.length}
          ref={(el) => {
            if (el) {
              el.indeterminate = selectedIds.size > 0 && selectedIds.size < nonBreakPageSegments.length;
            }
          }}
          onChange={() => {
            if (selectedIds.size === nonBreakPageSegments.length) {
              clearSelection();
            } else {
              setSelectedIds(new Set(nonBreakPageSegments.map((s) => s.id)));
            }
          }}
          className="w-3 h-3 accent-primary cursor-pointer"
          title="Select all"
        />
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
              aria-label="Clear search"
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
        {showFlags && (
          <FilterChip
            active={filters.showFlaggedOnly}
            color="warning"
            icon={AlertTriangle}
            count={flaggedCount}
            title="Show flagged segments"
            onClick={() => { filters.setShowFlaggedOnly(!filters.showFlaggedOnly); filters.setPage(0); }}
          />
        )}
        <div className="relative">
          <button
            onClick={() => setShowFlagSettings(!showFlagSettings)}
            className={`p-1 rounded transition ${
              showFlagSettings ? "bg-accent text-foreground" : "text-muted-foreground hover:text-foreground hover:bg-accent"
            }`}
            aria-label="Flag rules"
            title="Flag rules: speech speed and word list"
          >
            <SlidersHorizontal className="w-3 h-3" />
          </button>
          {showFlagSettings && (
            <div className="absolute left-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg p-3 space-y-3 w-80 text-xs">
              <div className="space-y-2">
                <p className="text-xs font-medium">Flagged segments</p>
                <p className="text-2xs text-muted-foreground">
                  Segments that may need a look are flagged automatically: odd
                  speech speed, or words you list below.
                </p>
                <p className="text-2xs text-muted-foreground pt-0.5">Flag speech that is:</p>
                <label className="flex items-center gap-2">
                  <span className="shrink-0 w-16">Too sparse</span>
                  <input
                    type="range"
                    min={1}
                    max={40}
                    value={filters.densityThreshold}
                    onChange={(e) => filters.setDensityThreshold(Number(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-muted-foreground w-24 text-right tabular-nums">
                    under {filters.densityThreshold} char/s
                  </span>
                </label>
                <label className="flex items-center gap-2">
                  <span className="shrink-0 w-16">Too dense</span>
                  <input
                    type="range"
                    min={20}
                    max={150}
                    value={filters.maxDensityThreshold}
                    onChange={(e) => filters.setMaxDensityThreshold(Number(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-muted-foreground w-24 text-right tabular-nums">
                    over {filters.maxDensityThreshold} char/s
                  </span>
                </label>
              </div>
              <div className="space-y-1 pt-1.5 border-t border-border/50">
                <p className="text-2xs text-muted-foreground">
                  Flag text containing these words (one per line):
                </p>
                <textarea
                  value={patternDraft}
                  onChange={(e) => setPatternDraft(e.target.value)}
                  onBlur={() => {
                    const list = patternDraft.split("\n").map((p) => p.trim()).filter(Boolean);
                    setFlagPatterns(list);
                    setPatternDraft(list.join("\n"));
                  }}
                  rows={4}
                  placeholder={"um\neuh"}
                  className="w-full bg-secondary border border-border rounded px-1.5 py-1 outline-none focus:border-primary/50 font-mono"
                />
              </div>
            </div>
          )}
        </div>
        {pendingRemovalCount > 0 && (
          <FilterChip
            active={filters.showRemovedOnly}
            color="destructive"
            icon={Trash2}
            count={pendingRemovalCount}
            title="Show segments pending removal; review before saving"
            onClick={() => { filters.setShowRemovedOnly(!filters.showRemovedOnly); filters.setPage(0); }}
          />
        )}
        {hasReference && (
          <>
            <span className="h-4 w-px bg-border mx-0.5" aria-hidden />
            <button
              onClick={() => setShowDiff((v) => !v)}
              className={`flex items-center gap-1 text-xs px-1.5 py-0.5 rounded transition hover:bg-accent ${
                showDiff ? "text-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
              title={showDiff ? "Hide word-level diff" : "Show word-level diff"}
            >
              {showDiff ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
              Diff
            </button>
            {changedCount > 0 && (
              <FilterChip
                active={filters.showChangedOnly}
                color="info"
                icon={Diff}
                count={changedCount}
                title="Show changed only"
                onClick={() => { filters.setShowChangedOnly(!filters.showChangedOnly); filters.setPage(0); }}
              />
            )}
          </>
        )}
        <div className="flex-1" />
        {activeId != null && (
          <Button
            variant="ghost"
            size="sm"
            className={`text-xs h-7 ${
              followMode && isPlayingThisFile
                ? "bg-primary/15 text-primary hover:bg-primary/20"
                : ""
            }`}
            onClick={jumpToActive}
            title={
              !isPlayingThisFile
                ? "Re-engage follow (auto-scroll resumes when you press play)"
                : followMode
                  ? "Following playback. Click to re-center."
                  : "Click to follow playback (auto-scroll resumes)"
            }
          >
            <Locate className="w-3 h-3 mr-1" />
            Now playing
          </Button>
        )}
        {canUndo && (
          <Button
            variant="ghost"
            size="sm"
            className="text-xs h-7"
            onClick={undo}
            title="Undo (Cmd+Z)"
          >
            <Undo2 className="w-3 h-3 mr-1" />
            Undo
          </Button>
        )}
        {exportSlot}
        <Button
          variant={isDirty || canMarkReviewed ? "default" : "outline"}
          size="sm"
          className="text-xs h-7"
          onClick={() => saveMutation.mutate()}
          disabled={saveMutation.isPending || (!isDirty && !canMarkReviewed)}
          title={saveButton.title}
        >
          <saveButton.Icon className="w-3 h-3 mr-1" />
          {saveMutation.isPending ? "Saving..." : saveButton.label}
        </Button>
      </div>

      {selectedIds.size > 0 && (
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-muted-foreground">{selectedIds.size} selected</span>
          <select
            value=""
            onChange={(e) => { if (e.target.value) { bulkSpeaker(e.target.value); } }}
            className={`${selectClass} text-xs`}
          >
            <option value="">Set speaker…</option>
            {speakers.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          {selectedIds.size >= 2 && (
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
  );
}
