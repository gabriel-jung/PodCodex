import { useRef, useEffect, useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Play, Pause, Trash2, ChevronDown, ChevronRight, Diff, Merge, Scissors, X, AlertTriangle } from "lucide-react";
import type { Segment } from "@/api/types";
import { formatTime } from "@/lib/utils";
import { computeWordDiff } from "@/lib/diffUtils";
import { useAudioStore } from "@/stores";

interface SegmentRowProps {
  segment: Segment;
  index: number;
  totalCount: number;
  isEdited: boolean;
  isFlagged: boolean;
  flagReason?: string | null;
  onDismissFlag?: () => void;
  isChanged: boolean;
  isActive: boolean;
  isBreak: boolean;
  audioPath?: string;
  referenceText?: string;
  referenceLabel?: string;
  referenceIndex?: number;
  speakers: string[];
  showDelete: boolean;
  showSpeaker: boolean;
  onTextChange: (text: string) => void;
  onSpeakerChange: (speaker: string) => void;
  onTimestampChange: (field: "start" | "end", value: number) => void;
  onDelete: () => void;
  onInsertBefore?: () => void;
  onInsertAfter?: () => void;
  onMergeNext?: () => void;
  onSplit?: (cursorPos: number) => void;
}

function InsertBeforeIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" className={className}>
      <line x1="4" y1="4" x2="12" y2="4" />
      <line x1="8" y1="8" x2="8" y2="14" />
      <line x1="5" y1="11" x2="11" y2="11" />
      <polyline points="5.5,10 8,7.5 10.5,10" />
    </svg>
  );
}

function InsertAfterIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" className={className}>
      <line x1="4" y1="12" x2="12" y2="12" />
      <line x1="8" y1="2" x2="8" y2="8" />
      <line x1="5" y1="5" x2="11" y2="5" />
      <polyline points="5.5,6 8,8.5 10.5,6" />
    </svg>
  );
}

/** Simple word-level diff — highlights removed (red) and added (green) words. */
function DiffView({ original, current }: { original: string; current: string }) {
  const diff = useMemo(() => computeWordDiff(original, current), [original, current]);
  return (
    <div className="text-xs leading-relaxed pr-4 py-1">
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

function hasValidTime(seg: Segment): boolean {
  return isFinite(seg.start) && isFinite(seg.end) && (seg.start > 0 || seg.end > 0);
}

export default function SegmentRow({
  segment,
  index,
  totalCount,
  isEdited,
  isFlagged,
  flagReason,
  onDismissFlag,
  isChanged,
  isActive,
  isBreak,
  audioPath,
  referenceText,
  referenceLabel,
  referenceIndex,
  speakers,
  showDelete,
  showSpeaker,
  onTextChange,
  onSpeakerChange,
  onTimestampChange,
  onDelete,
  onInsertBefore,
  onInsertAfter,
  onMergeNext,
  onSplit,
}: SegmentRowProps) {
  const seekTo = useAudioStore((s) => s.seekTo);
  const pauseAudio = useAudioStore((s) => s.pauseAudio);
  const isPlaying = useAudioStore((s) => s.isPlaying);
  const getAudioTime = () => useAudioStore.getState().currentTime;
  const [tsExpanded, setTsExpanded] = useState(false);
  const [refExpanded, setRefExpanded] = useState(true);
  const textRef = useRef<HTMLTextAreaElement>(null);

  const validTime = hasValidTime(segment);

  // Whether reference text exists and whether it differs from current
  const hasRef = referenceText != null;
  const hasDiff = hasRef && referenceText !== segment.text;

  // Auto-resize textarea to fit content
  useEffect(() => {
    const el = textRef.current;
    if (!el) return;
    el.style.height = "0";
    el.style.height = el.scrollHeight + "px";
  }, [segment.text]);

  // [BREAK] divider
  if (isBreak) {
    const gap = segment.end - segment.start;
    return (
      <div className="flex items-center gap-3 px-4 py-1.5 text-muted-foreground/40">
        <div className="flex-1 border-t border-dashed border-border" />
        <span className="text-[10px] select-none">
          {gap > 0 ? `${gap.toFixed(0)}s pause` : "break"}
        </span>
        <div className="flex-1 border-t border-dashed border-border" />
      </div>
    );
  }

  const handlePlay = () => {
    if (!audioPath || !validTime) return;
    seekTo(audioPath, segment.start);
  };

  return (
    <div
      className={`px-4 py-1 border-b border-border/50 transition-colors ${
        isActive ? "border-l-2 border-l-green-500 bg-green-500/10" :
        isFlagged ? "border-l-2 border-l-yellow-500" : isChanged ? "border-l-2 border-l-blue-500/50" : ""
      } ${isEdited && !isActive ? "bg-accent/20" : ""}`}
    >
      {/* Header row */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="w-12 text-right shrink-0 tabular-nums">{index + 1}/{totalCount}</span>

        {showSpeaker && (
          <select
            value={segment.speaker}
            onChange={(e) => onSpeakerChange(e.target.value)}
            className="bg-secondary text-secondary-foreground rounded px-1.5 py-0.5 border border-border text-xs max-w-[140px]"
          >
            {speakers.includes(segment.speaker) ? null : (
              <option value={segment.speaker}>{segment.speaker}</option>
            )}
            {speakers.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        )}

        {validTime && (
          <>
            <button
              onClick={() => setTsExpanded(!tsExpanded)}
              className="tabular-nums hover:text-foreground transition"
              title="Edit timestamps"
            >
              {formatTime(segment.start, false)} – {formatTime(segment.end, false)}
            </button>

            {audioPath && (
              isPlaying ? (
                <button onClick={pauseAudio} className="text-muted-foreground hover:text-foreground" title="Pause">
                  <Pause className="w-3 h-3" />
                </button>
              ) : (
                <button onClick={handlePlay} className="text-muted-foreground hover:text-foreground" title="Play from here">
                  <Play className="w-3 h-3" />
                </button>
              )
            )}
          </>
        )}

        {/* Flag reason + dismiss */}
        {isFlagged && flagReason && (
          <span className="flex items-center gap-1 text-yellow-500 text-[10px]">
            <AlertTriangle className="w-3 h-3" />
            {flagReason}
            {onDismissFlag && (
              <button
                onClick={onDismissFlag}
                className="hover:text-yellow-300 transition ml-0.5"
                title="Dismiss this flag"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </span>
        )}

        <div className="flex-1" />

        {isEdited && <span className="text-primary text-[10px]">edited</span>}

        {onInsertBefore && (
          <Button
            onClick={onInsertBefore}
            variant="ghost"
            size="icon"
            className="h-5 w-5 text-muted-foreground hover:text-foreground"
            title="Insert segment before"
          >
            <InsertBeforeIcon className="w-3.5 h-3.5" />
          </Button>
        )}

        {onMergeNext && (
          <Button
            onClick={onMergeNext}
            variant="ghost"
            size="icon"
            className="h-5 w-5 text-muted-foreground hover:text-foreground"
            title="Merge with next segment"
          >
            <Merge className="w-3 h-3" />
          </Button>
        )}

        {showDelete && (
          <Button
            onClick={onDelete}
            variant="ghost"
            size="icon"
            className="h-5 w-5 text-muted-foreground hover:text-destructive"
            title="Delete segment"
          >
            <Trash2 className="w-3 h-3" />
          </Button>
        )}

        {onInsertAfter && (
          <Button
            onClick={onInsertAfter}
            variant="ghost"
            size="icon"
            className="h-5 w-5 text-muted-foreground hover:text-foreground"
            title="Insert segment after"
          >
            <InsertAfterIcon className="w-3.5 h-3.5" />
          </Button>
        )}
      </div>

      {/* Timestamps (collapsible) */}
      {tsExpanded && (
        <div className="flex items-center gap-3 ml-10 text-xs">
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
              className="text-[10px] text-muted-foreground hover:text-foreground bg-secondary rounded px-1 py-0.5 border border-border"
              title="Set to current playback position"
            >
              ← current time
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
              className="text-[10px] text-muted-foreground hover:text-foreground bg-secondary rounded px-1 py-0.5 border border-border"
              title="Set to current playback position"
            >
              ← current time
            </button>
          </label>
        </div>
      )}

      {/* Text */}
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
        className="w-full bg-transparent text-sm leading-relaxed resize-none outline-none pr-4 overflow-hidden rounded border border-transparent hover:border-border focus:border-primary/50 focus:bg-accent/10 px-1.5 py-0 transition"
        style={{ marginLeft: "calc(2.5rem - 6px)" }}
        rows={1}
      />

      {/* Reference text — always shown when available, diff highlighted when different */}
      {hasRef && (
        <div className="ml-10">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setRefExpanded(!refExpanded)}
              className="flex items-center gap-1 text-[10px] text-muted-foreground/60 hover:text-muted-foreground transition"
            >
              <Diff className="w-3 h-3" />
              <span>{referenceLabel}{!hasDiff && " ✓"}</span>
              {refExpanded ? <ChevronDown className="w-2.5 h-2.5" /> : <ChevronRight className="w-2.5 h-2.5" />}
            </button>
          </div>
          {refExpanded && (
            hasDiff
              ? <DiffView original={referenceText!} current={segment.text} />
              : <p className="text-xs text-muted-foreground/50 py-1">{referenceText}</p>
          )}
        </div>
      )}
    </div>
  );
}
