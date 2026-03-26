import { useState, type RefObject } from "react";
import { Button } from "@/components/ui/button";
import { Search, X, SlidersHorizontal, Clock, Undo2, HelpCircle } from "lucide-react";
import type { VersionInfo } from "@/api/types";

function Tip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="text-muted-foreground/40 hover:text-muted-foreground transition"
      >
        <HelpCircle className="w-3 h-3" />
      </button>
      {show && (
        <div className="absolute left-0 top-full mt-1 z-50 bg-popover text-popover-foreground text-xs rounded-md border border-border shadow-lg px-2.5 py-1.5 max-w-64 whitespace-normal">
          {text}
        </div>
      )}
    </span>
  );
}

interface EditorToolbarProps {
  totalSegments: number;
  visibleCount: number;
  isDirty: boolean;
  versionInfo: VersionInfo | null;
  flaggedCount: number;
  deletedCount: number;
  speakers: string[];
  speakerFilter: string;
  onSpeakerFilterChange: (speaker: string) => void;
  showFlaggedOnly: boolean;
  onFlaggedFilterChange: (show: boolean) => void;
  showChangedOnly: boolean;
  onChangedFilterChange: (show: boolean) => void;
  hasReference: boolean;
  changedCount: number;
  densityThreshold: number;
  onDensityChange: (value: number) => void;
  maxDensityThreshold: number;
  onMaxDensityChange: (value: number) => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  searchRef?: RefObject<HTMLInputElement | null>;
  onSave: () => void;
  onUndo?: () => void;
  onLoadOriginal?: () => void;
  onLoadEdits?: () => void;
  onDeleteFlagged: () => void;
  onEstimateTimestamps?: () => void;
  isSaving: boolean;
}

export default function EditorToolbar({
  totalSegments,
  visibleCount,
  isDirty,
  versionInfo,
  flaggedCount,
  deletedCount,
  speakers,
  speakerFilter,
  onSpeakerFilterChange,
  showFlaggedOnly,
  onFlaggedFilterChange,
  showChangedOnly,
  onChangedFilterChange,
  hasReference,
  changedCount,
  densityThreshold,
  onDensityChange,
  maxDensityThreshold,
  onMaxDensityChange,
  searchQuery,
  onSearchChange,
  searchRef,
  onSave,
  onUndo,
  onLoadOriginal,
  onLoadEdits,
  onDeleteFlagged,
  onEstimateTimestamps,
  isSaving,
}: EditorToolbarProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const hasRaw = versionInfo?.has_raw ?? false;
  const hasValidated = versionInfo?.has_validated ?? false;

  return (
    <div className="px-4 py-2 border-b border-border flex items-center gap-3 flex-wrap text-xs">
      {/* Segment count */}
      <span className="text-muted-foreground">
        {visibleCount} of {totalSegments} segment{totalSegments !== 1 ? "s" : ""}
        {deletedCount > 0 && ` (${deletedCount} deleted)`}
      </span>

      {/* Version badge */}
      {(hasValidated || hasRaw) && (
        <span
          className={`px-2 py-0.5 rounded-full ${
            hasValidated && !isDirty
              ? "bg-green-900/40 text-green-400"
              : "bg-yellow-900/40 text-yellow-400"
          }`}
        >
          {hasValidated && !isDirty ? "Saved" : "Unsaved"}
        </span>
      )}

      {/* Speaker filter */}
      {speakers.length > 1 && (
        <select
          value={speakerFilter}
          onChange={(e) => onSpeakerFilterChange(e.target.value)}
          className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border"
        >
          <option value="">All speakers</option>
          {speakers.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      )}

      {/* Flagged filter */}
      {flaggedCount > 0 && (
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={showFlaggedOnly}
            onChange={(e) => onFlaggedFilterChange(e.target.checked)}
            className="accent-yellow-400"
          />
          <span className="text-yellow-400">{flaggedCount} flagged</span>
          <Tip text="Segments that may need attention: unidentified speakers, suspiciously little text for their duration (could be silence or bad timing), or way too much text (could be repeated/hallucinated content). Click the sliders icon to adjust sensitivity." />
        </label>
      )}

      {/* Changed filter */}
      {hasReference && changedCount > 0 && (
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={showChangedOnly}
            onChange={(e) => onChangedFilterChange(e.target.checked)}
            className="accent-blue-400"
          />
          <span className="text-blue-400">{changedCount} changed</span>
          <Tip text="Segments where the current text differs from the previous step's version — for example, corrections made during polishing, or translated text. Useful for reviewing what the AI changed." />
        </label>
      )}

      {/* Advanced (density) toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className={`p-1 rounded hover:bg-secondary transition ${showAdvanced ? "text-foreground" : "text-muted-foreground"}`}
        title="Adjust text density thresholds for flagging"
      >
        <SlidersHorizontal className="w-3.5 h-3.5" />
      </button>

      {/* Search */}
      <div className="relative flex items-center">
        <Search className="absolute left-1.5 w-3 h-3 text-muted-foreground pointer-events-none" />
        <input
          ref={searchRef}
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search..."
          className="bg-secondary text-secondary-foreground rounded pl-6 pr-6 py-1 border border-border w-36 text-xs"
        />
        {searchQuery && (
          <button
            onClick={() => onSearchChange("")}
            className="absolute right-1.5 text-muted-foreground hover:text-foreground"
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>

      <div className="flex-1" />

      {/* Actions */}
      {onUndo && (
        <Button
          onClick={onUndo}
          variant="ghost"
          size="sm"
          className="h-6"
          title="Undo last action"
        >
          <Undo2 className="w-3 h-3 mr-1" />
          Undo
        </Button>
      )}
      {onEstimateTimestamps && (
        <Button
          onClick={onEstimateTimestamps}
          variant="outline"
          size="sm"
          className="h-6"
          title="Recalculate timestamps based on text length (for uploaded transcripts without timing)"
        >
          <Clock className="w-3 h-3 mr-1" />
          Estimate timestamps
        </Button>
      )}
      {flaggedCount > 0 && (
        <Button
          onClick={onDeleteFlagged}
          variant="ghost"
          size="sm"
          className="text-yellow-400 h-6"
        >
          Delete flagged
        </Button>
      )}
      {onLoadOriginal && (
        <Button
          onClick={onLoadOriginal}
          disabled={!hasRaw}
          variant="ghost"
          size="sm"
          className="h-6"
        >
          Load original
        </Button>
      )}
      {onLoadEdits && (
        <Button
          onClick={onLoadEdits}
          disabled={!hasValidated}
          variant="ghost"
          size="sm"
          className="h-6"
        >
          Load edits
        </Button>
      )}
      <Button onClick={onSave} disabled={!isDirty || isSaving} size="sm" className="h-6">
        {isSaving ? "Saving..." : "Save"}
      </Button>

      {/* Density sliders — full width row below */}
      {showAdvanced && (
        <div className="w-full flex flex-col gap-1.5 pt-1 border-t border-border/50 mt-1">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground whitespace-nowrap w-24 flex items-center gap-1">
              Too sparse &lt;
              <Tip text="Flag segments where there's too little text for the time span. Usually means silence, a long pause, or incorrect timestamps. For reference, normal speech has about 10-15 characters per second." />
            </span>
            <input
              type="range"
              min={0}
              max={10}
              step={0.5}
              value={densityThreshold}
              onChange={(e) => onDensityChange(Number(e.target.value))}
              className="flex-1 max-w-48 accent-yellow-400 h-1"
            />
            <span className="text-muted-foreground tabular-nums w-14">{densityThreshold} char/s</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground whitespace-nowrap w-24 flex items-center gap-1">
              Too dense &gt;
              <Tip text="Flag segments where there's too much text for the time span. Usually means the transcription repeated itself, or the timestamps are wrong. Normal speech has about 10-15 characters per second." />
            </span>
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={maxDensityThreshold}
              onChange={(e) => onMaxDensityChange(Number(e.target.value))}
              className="flex-1 max-w-48 accent-yellow-400 h-1"
            />
            <span className="text-muted-foreground tabular-nums w-14">{maxDensityThreshold} char/s</span>
          </div>
        </div>
      )}
    </div>
  );
}
