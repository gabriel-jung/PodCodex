/**
 * SpeakerStrip — horizontal chip strip for renaming, removing, and auditioning
 * speakers. Lives inside TranscriptViewer, above the action bar.
 *
 * All edits are staged locally and committed by the editor's unified Save.
 * This component is controlled — state is held by the parent so the editor can
 * apply pending renames/removals to segments at save time.
 */

import { useState, useEffect, useMemo, useRef } from "react";
import {
  Plus,
  Trash2,
  Pencil,
  RotateCcw,
  AlertTriangle,
  Play,
  Pause,
  X,
} from "lucide-react";
import type { Segment } from "@/api/types";
import { useAudioStore } from "@/stores";
import { formatTime } from "@/lib/utils";
import { MIN_DENSITY } from "@/hooks/useSegmentFiltering";

// Matches raw diarizer output like SPEAKER_00, SPEAKER_12.
const DIARIZER_DEFAULT_RE = /^SPEAKER_\d+$/;
const isDiarizerDefault = (id: string) =>
  id === "" || DIARIZER_DEFAULT_RE.test(id);

// ── Derived types ────────────────────────────────────────────────────────────

interface Excerpt {
  start: number;
  end: number;
  text: string;
  suspect: boolean;
}

interface SpeakerInfo {
  name: string;
  count: number;
  durationSec: number;
  suspectCount: number;
  excerpts: Excerpt[];
}

function computeSpeakerInfos(segments: Segment[]): SpeakerInfo[] {
  const bySpeaker = new Map<string, Excerpt[]>();
  for (const seg of segments) {
    const sp = seg.speaker || "";
    if (sp === "[BREAK]") continue;
    if (!bySpeaker.has(sp)) bySpeaker.set(sp, []);
    const dur = (seg.end || 0) - (seg.start || 0);
    const text = seg.text || "";
    const suspect = !text || (dur > 0 && text.length / dur < MIN_DENSITY);
    bySpeaker.get(sp)!.push({ start: seg.start, end: seg.end, text, suspect });
  }
  return Array.from(bySpeaker.entries())
    .map(([name, segs]) => {
      const byLen = (a: Excerpt, b: Excerpt) => (b.end - b.start) - (a.end - a.start);
      const clean = segs.filter((s) => !s.suspect).sort(byLen);
      const suspects = segs.filter((s) => s.suspect).sort(byLen);
      const durationSec = segs.reduce((acc, s) => acc + Math.max(0, s.end - s.start), 0);
      return {
        name,
        count: segs.length,
        durationSec,
        suspectCount: suspects.length,
        excerpts: [...clean, ...suspects].slice(0, 5),
      };
    })
    // Sort by speaking time, descending — biggest talker first.
    .sort((a, b) => b.durationSec - a.durationSec);
}

// ── Component ────────────────────────────────────────────────────────────────

interface SpeakerStripProps {
  segments: Segment[];
  pendingRenames: Record<string, string>;
  pendingRemovals: Set<string>;
  addedSpeakers: string[];
  showSpeakers?: string[];
  audioPath?: string;
  onRename: (from: string, to: string) => void;
  onToggleRemoved: (name: string) => void;
  onAddSpeaker: (name: string) => void;
  onRemoveAdded: (name: string) => void;
}

export default function SpeakerStrip({
  segments,
  pendingRenames,
  pendingRemovals,
  addedSpeakers,
  showSpeakers,
  audioPath,
  onRename,
  onToggleRemoved,
  onAddSpeaker,
  onRemoveAdded,
}: SpeakerStripProps) {
  const [drawerFor, setDrawerFor] = useState<string | null>(null);
  const [editingFor, setEditingFor] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [adding, setAdding] = useState(false);
  const [addDraft, setAddDraft] = useState("");
  const [error, setError] = useState<string | null>(null);

  const editInputRef = useRef<HTMLInputElement>(null);
  const addInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingFor) editInputRef.current?.select();
  }, [editingFor]);
  useEffect(() => {
    if (adding) addInputRef.current?.focus();
  }, [adding]);

  const speakerInfos = useMemo(() => computeSpeakerInfos(segments), [segments]);

  const infoByName = useMemo(() => {
    const m = new Map<string, SpeakerInfo>();
    for (const info of speakerInfos) m.set(info.name, info);
    return m;
  }, [speakerInfos]);

  // Set of names that already exist — used for rename/add conflict detection.
  const allNames = useMemo(() => {
    const s = new Set<string>();
    for (const info of speakerInfos) s.add(info.name);
    for (const a of addedSpeakers) s.add(a);
    for (const target of Object.values(pendingRenames)) s.add(target);
    return s;
  }, [speakerInfos, addedSpeakers, pendingRenames]);

  const unnamedCount = speakerInfos.filter((s) => isDiarizerDefault(s.name)).length;

  // Suggestions for the rename/add inputs: show-level speakers minus any
  // already present on a chip (either existing, added, or a rename target).
  const suggestions = useMemo(() => {
    if (!showSpeakers || showSpeakers.length === 0) return [];
    return showSpeakers.filter((s) => !allNames.has(s));
  }, [showSpeakers, allNames]);
  const suggestionsListId = "speaker-suggestions";

  // ── Handlers ───────────────────────────────────────────────────────────────

  const startEdit = (name: string) => {
    setEditingFor(name);
    setRenameDraft(pendingRenames[name] ?? name);
    setError(null);
  };

  const commitEdit = () => {
    if (!editingFor) return;
    const from = editingFor;
    const to = renameDraft.trim();
    if (!to || to === from) {
      // No-op rename: clear any prior pending rename on this chip.
      if (pendingRenames[from] !== undefined) onRename(from, from);
      setEditingFor(null);
      setError(null);
      return;
    }
    // Conflict: target name already in use (by another chip or another pending rename).
    const usedByOther = allNames.has(to) && to !== (pendingRenames[from] ?? null);
    if (usedByOther) {
      setError(`"${to}" already exists — merge via bulk assign instead`);
      return;
    }
    onRename(from, to);
    setEditingFor(null);
    setError(null);
  };

  const cancelEdit = () => {
    setEditingFor(null);
    setError(null);
  };

  const commitAdd = () => {
    const name = addDraft.trim();
    if (!name) {
      setAdding(false);
      setAddDraft("");
      return;
    }
    if (allNames.has(name)) {
      setError(`"${name}" already exists`);
      return;
    }
    onAddSpeaker(name);
    setAddDraft("");
    setAdding(false);
    setError(null);
  };

  const toggleDrawer = (name: string) => {
    setDrawerFor((cur) => (cur === name ? null : name));
  };

  // ── Chip list ──────────────────────────────────────────────────────────────

  const totalDurationSec = useMemo(
    () => speakerInfos.reduce((acc, s) => acc + s.durationSec, 0),
    [speakerInfos],
  );

  const chips = useMemo(
    () => [
      ...speakerInfos.map((s) => ({
        name: s.name,
        count: s.count,
        durationSec: s.durationSec,
        suspectCount: s.suspectCount,
        added: false,
        unnamed: isDiarizerDefault(s.name),
      })),
      ...addedSpeakers.map((a) => ({
        name: a,
        count: 0,
        durationSec: 0,
        suspectCount: 0,
        added: true,
        unnamed: false,
      })),
    ],
    [speakerInfos, addedSpeakers],
  );

  const drawerInfo = drawerFor ? infoByName.get(drawerFor) ?? null : null;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div>
      {suggestions.length > 0 && (
        <datalist id={suggestionsListId}>
          {suggestions.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      )}
      <div className="flex items-start gap-2">
        <span className="text-xs text-muted-foreground shrink-0 w-20 pt-1">Speakers</span>
        <div className="flex items-center gap-1.5 flex-wrap flex-1 min-w-0">
        {chips.map((chip) => {
          const pendingTo = pendingRenames[chip.name];
          const isRenamed = pendingTo != null && pendingTo !== chip.name;
          const isRemoved = pendingRemovals.has(chip.name);
          const isEditing = editingFor === chip.name;
          const isOpen = drawerFor === chip.name;
          const share = totalDurationSec > 0 && !chip.added
            ? chip.durationSec / totalDurationSec
            : 0;
          // Show "<1%" for tiny-but-nonzero shares so they don't read as "0%".
          const pctLabel = chip.added
            ? null
            : share === 0
              ? "0%"
              : share < 0.01
                ? "<1%"
                : `${Math.round(share * 100)}%`;

          let stateClass = "bg-secondary/60 border-border hover:bg-secondary";
          if (isRemoved) {
            stateClass =
              "bg-destructive/5 border-l-2 border-l-destructive border-destructive/30 text-muted-foreground";
          } else if (isRenamed) {
            stateClass =
              "bg-primary/5 border-l-2 border-l-primary border-primary/30";
          } else if (chip.unnamed) {
            stateClass =
              "bg-yellow-500/5 border-yellow-500/30 hover:bg-yellow-500/10";
          }
          if (isOpen) stateClass += " ring-1 ring-primary/40";

          // Proportional sizing: each chip's flex-basis is its share of speaking
          // time (%), so chips are approximately proportional within a row.
          // Added speakers keep their natural width. Min-width keeps tiny-share
          // chips readable without blowing up the wrap layout.
          const chipStyle: React.CSSProperties = chip.added
            ? {}
            : {
                flexGrow: 0,
                flexShrink: 1,
                flexBasis: `${Math.max(share * 100, 6)}%`,
                minWidth: "7rem",
              };
          const sizeClass = chip.added ? "shrink-0" : "";

          return (
            <div
              key={chip.name}
              style={chipStyle}
              className={`group inline-flex items-center gap-1.5 pl-2 pr-1 py-0.5 rounded-md border text-xs transition ${sizeClass} ${stateClass}`}
            >
              {isEditing ? (
                <>
                  <span className="text-muted-foreground shrink-0 tabular-nums">
                    {chip.name}
                  </span>
                  <span className="text-muted-foreground/60">→</span>
                  <input
                    ref={editInputRef}
                    value={renameDraft}
                    list={suggestions.length > 0 ? suggestionsListId : undefined}
                    onChange={(e) => {
                      setRenameDraft(e.target.value);
                      setError(null);
                    }}
                    onBlur={commitEdit}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        commitEdit();
                      } else if (e.key === "Escape") {
                        e.preventDefault();
                        cancelEdit();
                      }
                    }}
                    className="bg-transparent outline-none w-32 text-foreground"
                  />
                </>
              ) : (
                <button
                  type="button"
                  onClick={() => toggleDrawer(chip.name)}
                  className="flex items-center gap-1.5 min-w-0 flex-1"
                  title={
                    chip.added
                      ? "Added speaker (no segments yet)"
                      : `${isRenamed ? `${pendingTo} ← ${chip.name}\n` : ""}${chip.count} segments · ${formatTime(chip.durationSec, false)}${pctLabel ? ` (${pctLabel})` : ""}`
                  }
                >
                  {chip.unnamed && (
                    <AlertTriangle className="w-2.5 h-2.5 text-yellow-500 shrink-0" />
                  )}
                  <span
                    className={`flex-1 min-w-0 truncate text-left ${
                      isRenamed ? "font-medium text-primary" : "font-medium"
                    } ${isRemoved ? "line-through" : ""}`}
                  >
                    {isRenamed ? pendingTo : chip.name}
                  </span>
                  {!chip.added && (
                    <span className="text-muted-foreground/70 text-2xs tabular-nums shrink-0">
                      {formatTime(chip.durationSec, false)} · {pctLabel}
                    </span>
                  )}
                  {chip.suspectCount > 0 && !isRemoved && (
                    <span
                      className="text-yellow-500 text-2xs shrink-0"
                      title={`${chip.suspectCount} suspect segments`}
                    >
                      ⚠{chip.suspectCount}
                    </span>
                  )}
                </button>
              )}

              {/* Per-chip actions (visible on hover or always when staged) */}
              {!isEditing && (
                <div
                  className={`flex items-center gap-0.5 shrink-0 transition-opacity ${
                    isRemoved || isRenamed
                      ? "opacity-100"
                      : "opacity-0 group-hover:opacity-100"
                  }`}
                >
                  {!isRemoved && !chip.added && (
                    <button
                      type="button"
                      onClick={() => startEdit(chip.name)}
                      className="text-muted-foreground hover:text-foreground p-0.5"
                      title="Rename"
                    >
                      <Pencil className="w-3 h-3" />
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      if (chip.added) onRemoveAdded(chip.name);
                      else onToggleRemoved(chip.name);
                    }}
                    className={`p-0.5 transition ${
                      isRemoved
                        ? "text-destructive hover:text-destructive/80"
                        : "text-muted-foreground hover:text-destructive"
                    }`}
                    title={
                      isRemoved
                        ? "Undo remove"
                        : chip.added
                          ? "Cancel added speaker"
                          : "Remove all segments for this speaker"
                    }
                  >
                    {isRemoved ? (
                      <RotateCcw className="w-3 h-3" />
                    ) : (
                      <Trash2 className="w-3 h-3" />
                    )}
                  </button>
                </div>
              )}
            </div>
          );
        })}

        {/* Add-speaker chip */}
        {adding ? (
          <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md border border-primary/40 bg-primary/5 text-xs shrink-0">
            <Plus className="w-3 h-3 text-primary/70" />
            <input
              ref={addInputRef}
              value={addDraft}
              list={suggestions.length > 0 ? suggestionsListId : undefined}
              onChange={(e) => {
                setAddDraft(e.target.value);
                setError(null);
              }}
              onBlur={commitAdd}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  commitAdd();
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  setAdding(false);
                  setAddDraft("");
                  setError(null);
                }
              }}
              placeholder="name…"
              className="bg-transparent outline-none w-24"
            />
          </div>
        ) : (
          <button
            type="button"
            onClick={() => setAdding(true)}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md border border-dashed border-border text-muted-foreground hover:text-foreground hover:border-border/80 text-xs shrink-0 transition"
            title="Add speaker"
          >
            <Plus className="w-3 h-3" />
            add
          </button>
        )}

        {unnamedCount > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-2xs text-yellow-500/80 shrink-0 ml-auto">
            <AlertTriangle className="w-3 h-3" />
            {unnamedCount} unnamed
          </span>
        )}
        </div>
      </div>

      {error && (
        <div className="pl-[5.5rem] mt-1 text-2xs text-destructive flex items-center gap-1">
          <X className="w-3 h-3" />
          {error}
        </div>
      )}

      {drawerInfo && (
        <div className="-mx-4 mt-1.5 px-4 py-1.5 space-y-1 bg-secondary/20 border-t border-border/40">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">
              <span className="font-medium text-foreground">{drawerInfo.name}</span> · {drawerInfo.count} segments
              {drawerInfo.suspectCount > 0 && ` · ${drawerInfo.suspectCount} suspect`}
            </span>
            <button
              type="button"
              onClick={() => setDrawerFor(null)}
              className="text-muted-foreground/60 hover:text-foreground transition"
              title="Close"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
          {drawerInfo.excerpts.length === 0 ? (
            <p className="text-xs text-muted-foreground italic">No excerpts.</p>
          ) : (
            drawerInfo.excerpts.map((ex, i) => (
              <ExcerptRow
                key={i}
                audioPath={audioPath}
                start={ex.start}
                end={ex.end}
                text={ex.text}
                suspect={ex.suspect}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

// ── Excerpt row ──────────────────────────────────────────────────────────────

function ExcerptRow({
  audioPath,
  start,
  end,
  text,
  suspect,
}: {
  audioPath?: string;
  start: number;
  end: number;
  text: string;
  suspect: boolean;
}) {
  const seekTo = useAudioStore((s) => s.seekTo);
  const pauseAudio = useAudioStore((s) => s.pauseAudio);
  // Derive active state in the selector so rows only re-render at boundary crossings,
  // not on every audio tick.
  const isActive = useAudioStore(
    (s) =>
      s.isPlaying &&
      s.audioPath === audioPath &&
      s.currentTime >= start - 0.5 &&
      s.currentTime <= end + 0.5,
  );
  const dur = end - start;
  const handlePlay = () => {
    if (audioPath) seekTo(audioPath, start);
  };

  return (
    <div
      className={`flex items-center gap-2 px-2 py-1 rounded text-xs transition-colors ${
        isActive ? "bg-primary/10 border-l-2 border-l-primary" : "bg-background/40"
      }`}
    >
      <button
        type="button"
        onClick={isActive ? pauseAudio : handlePlay}
        disabled={!audioPath}
        className="shrink-0 text-muted-foreground hover:text-foreground transition disabled:opacity-30"
        title={isActive ? "Pause" : `Play from ${formatTime(start)}`}
      >
        {isActive ? (
          <Pause className="w-3.5 h-3.5" />
        ) : (
          <Play className="w-3.5 h-3.5" />
        )}
      </button>
      <span className="tabular-nums text-muted-foreground shrink-0">
        {formatTime(start, false)} – {formatTime(end, false)}
      </span>
      <span className="text-muted-foreground/70 shrink-0">({dur.toFixed(1)}s)</span>
      {suspect && (
        <span className="text-yellow-500 shrink-0" title="Low speech density">
          ⚠
        </span>
      )}
      <span
        className={`truncate ${
          suspect ? "text-yellow-200/60 italic" : "text-foreground/80"
        }`}
      >
        {text || <span className="italic text-muted-foreground">no text</span>}
      </span>
    </div>
  );
}
