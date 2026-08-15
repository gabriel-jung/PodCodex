/**
 * SegmentRow — one editable transcript segment.
 *
 * Memoized so the parent virtualizer can mount/unmount rows without re-running
 * per-keystroke work for the rest. Callbacks must be reference-stable from the
 * parent (TranscriptViewer wraps each in useCallback keyed by the row id).
 */

import { memo, useEffect, useMemo, useRef, useState } from "react";
import type { Segment } from "@/api/types";
import { useAudioStore } from "@/stores";
import { formatTime, selectClass } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";
import { BREAK_SPEAKER } from "@/lib/speakers";
import { computeWordDiff } from "@/lib/diffUtils";
import {
  Play,
  Pause,
  Trash2,
  RotateCcw,
  Merge,
  Scissors,
  X,
  AlertTriangle,
  Timer,
} from "lucide-react";

// ── Insert-before / insert-after SVG glyphs ─────────────────────────────────

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

// ── Word-level diff view ────────────────────────────────────────────────────

function DiffView({ original, current }: { original: string; current: string }) {
  const diff = useMemo(() => computeWordDiff(original, current), [original, current]);
  return (
    <div className="text-sm leading-relaxed py-0">
      {diff.map((part, i) => (
        <span
          key={`${i}:${part.type}:${part.text}`}
          className={
            part.type === "removed"
              ? "bg-destructive/20 text-destructive line-through"
              : part.type === "added"
                ? "bg-success/20 text-success"
                : "text-muted-foreground/70"
          }
        >
          {part.text}
        </span>
      ))}
    </div>
  );
}

// Native auto-sizing textarea (Chrome 123+, Safari 18.4+). When available
// the browser grows the textarea to its content with zero JS — no per-row
// ResizeObserver, no forced reflow storm on mount.
const HAS_FIELD_SIZING =
  typeof CSS !== "undefined" && CSS.supports("field-sizing", "content");

/** Nearest start-of-word boundary so splits never land mid-token. */
function snapToWordStart(text: string, target: number): number {
  if (target <= 0) return 0;
  if (target >= text.length) return text.length;
  const isWordStart = (i: number) =>
    i > 0 && i < text.length && text[i - 1] === " " && text[i] !== " ";
  if (isWordStart(target)) return target;
  for (let i = 1; i <= text.length; i++) {
    if (isWordStart(target - i)) return target - i;
    if (isWordStart(target + i)) return target + i;
  }
  return target;
}

// ── Component ───────────────────────────────────────────────────────────────

export interface SegmentRowProps {
  segment: Segment;
  /** Stable row id — used as the key by every callback so React-side Sets
   *  (selection, dismissed flags, recently edited) survive insert/split. */
  id: number;
  /** Visual segment number shown in the row gutter. Position-based, so it
   *  re-numbers cleanly after inserts/deletes; not used as an identity. */
  displayNumber: number;
  isActive: boolean;
  isPlayingActive: boolean;
  isFlagged: boolean;
  flagReasonText: string | null;
  isChanged: boolean;
  isPendingRemoval?: boolean;
  isDeleted?: boolean;
  /** True when this row's text has been edited or was inserted/split this
   *  session. Used to tint the textarea so unsaved edits stand out. */
  isTextEdited?: boolean;
  selected: boolean;
  onToggleSelect: (id: number) => void;
  audioPath?: string;
  speakers: string[];
  showSpeaker: boolean;
  /** Fade the speaker label until the row is hovered or focused. */
  speakerMuted?: boolean;
  showDelete: boolean;
  onTextChange: (id: number, text: string) => void;
  onSpeakerChange: (id: number, speaker: string) => void;
  onTimestampChange: (id: number, field: "start" | "end", value: number) => void;
  onDelete: (id: number) => void;
  onRestore?: (id: number) => void;
  onDismissFlag?: (id: number) => void;
  onInsertBefore?: (id: number, segment: Segment) => void;
  onInsertAfter?: (id: number, segment: Segment) => void;
  onMergeNext?: (id: number, speaker: string) => void;
  onSplit?: (id: number, cursorPos: number, explicitTime?: number) => void;
  referenceText?: string;
  showDiff?: boolean;
}

const SegmentRow = memo(function SegmentRow({
  segment,
  id,
  displayNumber,
  isActive,
  isPlayingActive,
  isFlagged,
  selected,
  onToggleSelect,
  flagReasonText,
  isChanged,
  isPendingRemoval,
  isDeleted,
  isTextEdited,
  audioPath,
  speakers,
  showSpeaker,
  speakerMuted,
  showDelete,
  onTextChange,
  onSpeakerChange,
  onTimestampChange,
  onDelete,
  onRestore,
  onDismissFlag,
  onInsertBefore,
  onInsertAfter,
  onMergeNext,
  onSplit,
  referenceText,
  showDiff = true,
}: SegmentRowProps) {
  const [editingSpeaker, setEditingSpeaker] = useState(false);
  const [tsExpanded, setTsExpanded] = useState(false);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const speakerInputRef = useRef<HTMLInputElement>(null);

  const getAudioTime = () => useAudioStore.getState().currentTime;

  // Auto-resize textarea. Modern engines handle this via `field-sizing:
  // content` CSS — zero JS needed. For older engines fall back to manual
  // height sync on text change, ref-column toggle, and window resize. No
  // per-row ResizeObserver (N rows × 1 RO each was a mount-time bottleneck).
  const hasRef = referenceText != null;
  useEffect(() => {
    if (HAS_FIELD_SIZING) return;
    const el = textRef.current;
    if (!el) return;
    const recompute = () => {
      el.style.height = "0";
      el.style.height = el.scrollHeight + "px";
    };
    const raf = requestAnimationFrame(recompute);
    window.addEventListener("resize", recompute);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", recompute);
    };
  }, [segment.text, hasRef]);

  useEffect(() => {
    if (editingSpeaker) speakerInputRef.current?.select();
  }, [editingSpeaker]);

  const validTime = isFinite(segment.start) && isFinite(segment.end);

  const handleSeek = () => {
    if (audioPath && validTime) useAudioStore.getState().seekTo(audioPath, segment.start);
  };

  // Resolve a split point given an optional caret offset. Caret takes priority;
  // playback-time ratio is used only when no caret is placed.
  const computeSplit = (sel: number | null | undefined): { pos: number; t?: number } | null => {
    const text = segment.text;
    if (!text) return null;
    const playT = audioPath && validTime ? useAudioStore.getState().currentTime : null;
    const splitT = playT != null && playT > segment.start + 0.05 && playT < segment.end - 0.05
      ? playT
      : undefined;
    let target: number | null = null;
    if (sel != null && sel > 0 && sel < text.length) target = sel;
    else if (splitT != null) target = Math.round(((splitT - segment.start) / (segment.end - segment.start)) * text.length);
    if (target == null) return null;
    const pos = snapToWordStart(text, target);
    if (pos <= 0 || pos >= text.length) return null;
    return { pos, t: splitT };
  };

  const hasDiff = hasRef && referenceText !== segment.text && showDiff;

  if (segment.speaker === BREAK_SPEAKER) {
    const gap = segment.end - segment.start;
    return (
      <div className="py-2 flex items-center gap-2 text-muted-foreground/40 px-2">
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
      className={`group py-1.5 pl-1 pr-2 rounded transition-colors ${
        isPendingRemoval ? "opacity-50 line-through bg-destructive/5 border-l-2 border-l-destructive/50" :
        isActive ? "bg-accent/60" :
        isFlagged ? "border-l-2 border-l-warning" :
        isChanged ? "border-l-2 border-l-info/50" :
        "hover:bg-accent/20"
      }`}
    >
      {/* Main row: [checkbox + segnum stacked] | timestamp | speaker | text */}
      <div className="flex gap-2">
        <div className="shrink-0 flex flex-col items-center pt-1 gap-0.5" style={{ width: "1.1rem" }}>
          <input
            type="checkbox"
            checked={selected}
            onChange={() => onToggleSelect(id)}
            aria-label="Select segment"
            className={`w-3 h-3 accent-primary cursor-pointer transition-opacity ${
              selected ? "opacity-100" : "opacity-40 hover:opacity-100 group-hover:opacity-100"
            }`}
          />
          <div className="font-mono text-3xs text-muted-foreground/40 tabular-nums select-none">
            {displayNumber}
          </div>
        </div>
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

        {showSpeaker && (
          <div className="shrink-0 w-16 pt-0.5 relative">
            {editingSpeaker ? (
              // Float wider than the resting 64px column so long speaker
              // names aren't truncated and the native dropdown anchor is
              // tall enough to show full-height items.
              <div className="absolute left-0 top-0 z-20 min-w-full w-max max-w-48">
                {speakers.length > 0 ? (
                  <select
                    value={segment.speaker}
                    onChange={(e) => {
                      onSpeakerChange(id, e.target.value);
                      setEditingSpeaker(false);
                    }}
                    onBlur={() => setEditingSpeaker(false)}
                    autoFocus
                    className={`${selectClass} w-full text-xs py-0.5 h-6`}
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
                    onChange={(e) => onSpeakerChange(id, e.target.value)}
                    onBlur={() => setEditingSpeaker(false)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === "Escape") setEditingSpeaker(false);
                    }}
                    className="w-full text-xs bg-secondary border border-border rounded px-1 py-0.5 h-6 outline-none"
                  />
                )}
              </div>
            ) : (
              <button
                onClick={() => setEditingSpeaker(true)}
                className={`text-xs font-medium truncate w-full text-left hover:opacity-80 transition ${speakerMuted ? "opacity-0 group-hover:opacity-100 focus-visible:opacity-100" : ""}`}
                style={{ color: speakerColor(segment.speaker) }}
                title="Click to edit speaker"
              >
                {segment.speaker}
              </button>
            )}
          </div>
        )}

        <div className={`flex-1 min-w-0 ${hasRef ? "flex flex-col lg:flex-row lg:gap-2" : ""}`}>
          <div className={hasRef ? "flex-1 min-w-0" : ""}>
            <textarea
              ref={textRef}
              value={segment.text}
              onChange={(e) => onTextChange(id, e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && onSplit) {
                  e.preventDefault();
                  const split = computeSplit(e.currentTarget.selectionStart);
                  if (split) onSplit(id, split.pos, split.t);
                }
              }}
              className={`w-full text-sm leading-relaxed resize-none outline-none overflow-hidden rounded border hover:border-border focus:border-primary/50 focus:bg-accent/10 px-1.5 py-0 transition [field-sizing:content] ${
                isTextEdited
                  ? "bg-info/10 border-info/40"
                  : "bg-transparent border-transparent"
              }`}
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

      {isFlagged && flagReasonText && (
        <div className="flex items-center gap-1 mt-0.5 text-2xs leading-none text-warning/80">
          <AlertTriangle className="w-2.5 h-2.5" />
          <span>{flagReasonText}</span>
          {onDismissFlag && (
            <button
              onClick={() => onDismissFlag(id)}
              className="hover:text-warning transition ml-0.5"
              title="Dismiss this flag"
            >
              <X className="w-2.5 h-2.5" />
            </button>
          )}
        </div>
      )}

      {/* Actions rail — muted by default, brightens on row hover. */}
      <div className="flex items-center gap-0.5 mt-1 opacity-30 group-hover:opacity-100 transition-opacity duration-150 w-fit">
        {audioPath && validTime && (
          <>
            {isPlayingActive ? (
              <button
                onClick={() => useAudioStore.getState().pauseAudio()}
                className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-secondary transition"
                title="Pause"
              >
                <Pause className="w-3.5 h-3.5" />
              </button>
            ) : (
              <button
                onClick={handleSeek}
                className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-secondary transition"
                title="Play from here"
              >
                <Play className="w-3.5 h-3.5" />
              </button>
            )}
            {(onInsertBefore || onMergeNext || onSplit || onInsertAfter) && (
              <span className="mx-1 h-3 w-px bg-border/60" aria-hidden />
            )}
          </>
        )}
        {onInsertBefore && (
          <button
            onClick={() => onInsertBefore(id, segment)}
            className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-secondary transition"
            title="Insert segment before"
          >
            <InsertBeforeIcon className="w-3.5 h-3.5" />
          </button>
        )}
        {onMergeNext && (
          <button
            onClick={() => onMergeNext(id, segment.speaker)}
            className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-secondary transition"
            title="Merge with next"
          >
            <Merge className="w-3.5 h-3.5" />
          </button>
        )}
        {onSplit && (
          <button
            onClick={() => {
              const split = computeSplit(textRef.current?.selectionStart);
              if (split) onSplit(id, split.pos, split.t);
            }}
            className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-secondary transition"
            title="Split at current word (text caret) and current playback time"
          >
            <Scissors className="w-3.5 h-3.5" />
          </button>
        )}
        {onInsertAfter && (
          <button
            onClick={() => onInsertAfter(id, segment)}
            className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-secondary transition"
            title="Insert segment after"
          >
            <InsertAfterIcon className="w-3.5 h-3.5" />
          </button>
        )}
        {showDelete && (
          <>
            <span className="mx-1 h-3 w-px bg-border/60" aria-hidden />
            {isDeleted && onRestore ? (
              <button
                onClick={() => onRestore(id)}
                className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-secondary transition"
                title="Restore segment"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            ) : (
              <button
                onClick={() => onDelete(id)}
                className="text-muted-foreground hover:text-destructive p-1 rounded hover:bg-destructive/10 transition"
                title="Delete segment"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </>
        )}
      </div>

      {tsExpanded && (
        <div className="flex items-center gap-3 pl-1 text-xs py-1">
          <label className="flex items-center gap-1">
            Start:
            <input
              type="number"
              value={segment.start}
              onChange={(e) => onTimestampChange(id, "start", Number(e.target.value))}
              step={0.1}
              className="w-20 bg-secondary text-secondary-foreground rounded px-1.5 py-0.5 border border-border"
            />
            <button
              onClick={() => onTimestampChange(id, "start", Math.round(getAudioTime() * 10) / 10)}
              className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-secondary transition"
              title="Set to current playback time"
            >
              <Timer className="w-3.5 h-3.5" />
            </button>
          </label>
          <label className="flex items-center gap-1">
            End:
            <input
              type="number"
              value={segment.end}
              onChange={(e) => onTimestampChange(id, "end", Number(e.target.value))}
              step={0.1}
              className="w-20 bg-secondary text-secondary-foreground rounded px-1.5 py-0.5 border border-border"
            />
            <button
              onClick={() => onTimestampChange(id, "end", Math.round(getAudioTime() * 10) / 10)}
              className="text-muted-foreground hover:text-foreground p-1 rounded hover:bg-secondary transition"
              title="Set to current playback time"
            >
              <Timer className="w-3.5 h-3.5" />
            </button>
          </label>
        </div>
      )}
    </div>
  );
});

export default SegmentRow;
