import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, useEffect, useRef } from "react";
import { getSpeakerMap, saveSpeakerMap, getSegments, saveSegments } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { useAudioStore } from "@/stores";
import { formatTime } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Play,
  Pause,
  Plus,
  Trash2,
  X,
} from "lucide-react";

interface SpeakerMapEditorProps {
  audioPath: string;
  onSaved?: () => void;
}

interface Excerpt {
  start: number;
  end: number;
  text: string;
  suspect: boolean;
}

interface SpeakerInfo {
  id: string;
  segCount: number;
  suspectCount: number;
  /** Top segments sorted by duration (longest first), clean ones preferred. */
  excerpts: Excerpt[];
}

const UNKNOWN_SPEAKERS = new Set(["", "SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03", "SPEAKER_04", "SPEAKER_05", "SPEAKER_06", "SPEAKER_07", "SPEAKER_08", "SPEAKER_09"]);

function ExcerptRow({ audioPath, start, end, text, suspect }: { audioPath: string; start: number; end: number; text: string; suspect: boolean }) {
  const seekTo = useAudioStore((s) => s.seekTo);
  const pauseAudio = useAudioStore((s) => s.pauseAudio);
  const isPlaying = useAudioStore((s) => s.isPlaying);
  const currentTime = useAudioStore((s) => s.currentTime);
  const dur = end - start;

  // Consider this excerpt "active" if the audio bar is playing within its range
  const isActive = isPlaying && currentTime >= start - 0.5 && currentTime <= end + 0.5;

  const handlePlay = () => seekTo(audioPath, start);

  return (
    <div
      className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs transition-colors ${
        isActive ? "bg-primary/10 border-l-2 border-l-primary" : "bg-muted/50"
      }`}
    >
      {/* Play/pause — same pattern as SegmentRow */}
      <button
        onClick={isActive ? pauseAudio : handlePlay}
        className="shrink-0 text-muted-foreground hover:text-foreground transition"
        title={isActive ? "Pause" : `Play from ${formatTime(start)}`}
      >
        {isActive ? (
          <Pause className="w-3.5 h-3.5" />
        ) : (
          <Play className="w-3.5 h-3.5" />
        )}
      </button>

      {/* Timestamp */}
      <span className="tabular-nums text-muted-foreground shrink-0">
        {formatTime(start, false)} – {formatTime(end, false)}
      </span>
      <span className="text-muted-foreground shrink-0">({dur.toFixed(1)}s)</span>

      {/* Text */}
      {suspect && <span className="text-yellow-500 shrink-0" title="Low speech density or suspect text">⚠️</span>}
      <span className={`truncate ${suspect ? "text-yellow-200/60 italic" : "text-foreground/80"}`}>
        {text || <span className="italic text-muted-foreground">no text</span>}
      </span>
    </div>
  );
}

export default function SpeakerMapEditor({ audioPath, onSaved }: SpeakerMapEditorProps) {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const [newName, setNewName] = useState("");
  const [expandedSpeakers, setExpandedSpeakers] = useState<Set<string>>(new Set());

  // Load speaker map (raw diarization IDs → names)
  const { data: serverMap } = useQuery({
    queryKey: queryKeys.speakerMap(audioPath),
    queryFn: () => getSpeakerMap(audioPath),
  });

  // Load segments to get current speaker names
  const { data: segments } = useQuery({
    queryKey: queryKeys.transcribeSegments(audioPath),
    queryFn: () => getSegments(audioPath),
  });

  // Build speaker info with top excerpts
  const speakerInfos: SpeakerInfo[] = (() => {
    if (!segments) return [];
    const bySpeaker = new Map<string, Excerpt[]>();
    for (const seg of segments) {
      const sp = seg.speaker || "";
      if (sp === "[BREAK]") continue;
      if (!bySpeaker.has(sp)) bySpeaker.set(sp, []);
      const dur = (seg.end || 0) - (seg.start || 0);
      const text = seg.text || "";
      const suspect = !text || (dur > 0 && text.length / dur < 2);
      bySpeaker.get(sp)!.push({ start: seg.start, end: seg.end, text, suspect });
    }
    return Array.from(bySpeaker.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([id, segs]) => {
        const clean = segs.filter((s) => !s.suspect).sort((a, b) => (b.end - b.start) - (a.end - a.start));
        const suspects = segs.filter((s) => s.suspect).sort((a, b) => (b.end - b.start) - (a.end - a.start));
        return {
          id,
          segCount: segs.length,
          suspectCount: suspects.length,
          // Prefer clean excerpts, fill remaining with suspect ones
          excerpts: [...clean, ...suspects].slice(0, 5),
        };
      });
  })();

  // Detect unnamed speakers
  const hasUnnamed = speakerInfos.some((s) => UNKNOWN_SPEAKERS.has(s.id));
  const unnamedCount = speakerInfos.filter((s) => UNKNOWN_SPEAKERS.has(s.id)).length;

  // Auto-expand once on first load if speakers need naming
  const autoExpandedRef = useRef(false);
  useEffect(() => {
    if (!autoExpandedRef.current && hasUnnamed && speakerInfos.length > 0) {
      autoExpandedRef.current = true;
      setExpanded(true);
    }
  }, [hasUnnamed, speakerInfos.length]);

  // Local state: list of {current, renamed} entries
  const [entries, setEntries] = useState<{ current: string; renamed: string }[]>([]);

  // Also keep the raw speaker map for saving
  const [localMap, setLocalMap] = useState<Record<string, string>>({});

  useEffect(() => {
    if (serverMap) setLocalMap(serverMap);
  }, [serverMap]);

  // Build entries from segment speakers
  useEffect(() => {
    if (speakerInfos.length > 0) {
      setEntries(speakerInfos.map((s) => ({ current: s.id, renamed: s.id })));
    }
  }, [segments]); // eslint-disable-line react-hooks/exhaustive-deps

  const isDirty = entries.some((e) => e.current !== e.renamed)
    || JSON.stringify(localMap) !== JSON.stringify(serverMap ?? {});

  const hasRenames = entries.some((e) => e.current !== e.renamed);

  const mutation = useMutation({
    mutationFn: async () => {
      // 1. Save speaker map if changed
      if (JSON.stringify(localMap) !== JSON.stringify(serverMap ?? {})) {
        await saveSpeakerMap(audioPath, localMap);
      }

      // 2. Apply renames to segments if any changed
      if (hasRenames && segments) {
        const renameMap = new Map<string, string>();
        for (const e of entries) {
          if (e.current !== e.renamed && e.renamed.trim()) {
            renameMap.set(e.current, e.renamed.trim());
          }
        }
        if (renameMap.size > 0) {
          const updated = segments.map((seg: Record<string, unknown>) => {
            const newName = renameMap.get(seg.speaker as string);
            return newName ? { ...seg, speaker: newName } : seg;
          });
          await saveSegments(audioPath, updated);
        }
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.speakerMap(audioPath) });
      queryClient.invalidateQueries({ queryKey: queryKeys.transcribeSegments(audioPath) });
      onSaved?.();
    },
  });

  const addSpeaker = () => {
    const name = newName.trim();
    if (!name || entries.some((e) => e.renamed === name)) return;
    setEntries([...entries, { current: name, renamed: name }]);
    setLocalMap({ ...localMap, [name]: name });
    setNewName("");
  };

  const removeSpeaker = (idx: number) => {
    const entry = entries[idx];
    const next = entries.filter((_, i) => i !== idx);
    setEntries(next);
    const nextMap = { ...localMap };
    delete nextMap[entry.current];
    setLocalMap(nextMap);
  };

  const markRemove = (idx: number) => {
    const next = [...entries];
    next[idx] = { ...next[idx], renamed: "[remove]" };
    setEntries(next);
  };

  const updateRenamed = (idx: number, value: string) => {
    const next = [...entries];
    next[idx] = { ...next[idx], renamed: value };
    setEntries(next);
  };

  const toggleSpeakerExcerpts = (id: string) => {
    setExpandedSpeakers((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const displayCount = entries.length || Object.keys(localMap).length;

  return (
    <div className="border-b border-border">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-2 flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition"
      >
        {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        <span className="font-medium">Speaker names</span>
        <span className="text-xs">({displayCount})</span>
        {hasUnnamed && (
          <span className="flex items-center gap-1 text-xs text-yellow-500 ml-2">
            <AlertTriangle className="w-3 h-3" />
            {unnamedCount} unnamed
          </span>
        )}
      </button>

      {expanded && (
        <div className="px-4 pb-3 space-y-2 max-w-3xl">
          {/* Warning banner */}
          {hasUnnamed && (
            <div className="flex items-start gap-2 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-sm">
              <AlertTriangle className="w-4 h-4 text-yellow-500 shrink-0 mt-0.5" />
              <div>
                <p className="text-foreground font-medium">Speaker map not set up</p>
                <p className="text-muted-foreground text-xs mt-0.5">
                  Listen to the excerpts below to identify each speaker, then type their name.
                </p>
              </div>
            </div>
          )}

          {entries.map((entry, idx) => {
            const info = speakerInfos.find((s) => s.id === entry.current);
            const isExpanded = expandedSpeakers.has(entry.current);
            const isRemove = entry.renamed === "[remove]";
            const isUnnamed = UNKNOWN_SPEAKERS.has(entry.current);

            return (
              <div
                key={entry.current}
                className={`border rounded-lg overflow-hidden ${
                  isUnnamed ? "border-yellow-500/30" : "border-border"
                }`}
              >
                {/* Speaker row */}
                <div className="flex items-center gap-2 px-3 py-2">
                  {/* Expand excerpts */}
                  <button
                    onClick={() => toggleSpeakerExcerpts(entry.current)}
                    className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition shrink-0"
                    title="Show audio excerpts"
                  >
                    {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    <span className="font-medium">{entry.current}</span>
                    {info && <span className="text-muted-foreground">({info.segCount})</span>}
                    {info && info.suspectCount > 0 && (
                      <span className="text-yellow-500" title={`${info.suspectCount} suspect segments (low density / no text)`}>
                        ⚠ {info.suspectCount}
                      </span>
                    )}
                  </button>

                  <span className="text-xs text-muted-foreground">→</span>

                  {/* Name input */}
                  <input
                    value={entry.renamed}
                    onChange={(e) => updateRenamed(idx, e.target.value)}
                    placeholder="Enter name..."
                    className={`input text-sm py-1 flex-1 min-w-0 ${
                      isRemove ? "text-destructive line-through" : ""
                    }`}
                  />

                  {/* Mark [remove] */}
                  <button
                    onClick={() => markRemove(idx)}
                    className="text-muted-foreground hover:text-destructive p-0.5 shrink-0"
                    title="Mark as [remove]"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                  {/* Delete speaker */}
                  <button
                    onClick={() => removeSpeaker(idx)}
                    className="text-muted-foreground hover:text-destructive p-0.5 shrink-0"
                    title="Delete speaker"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Audio excerpts — same play/pause as SegmentRow */}
                {isExpanded && info && info.excerpts.length > 0 && (
                  <div className="px-3 pb-2 space-y-1">
                    {info.excerpts.map((ex, i) => (
                      <ExcerptRow
                        key={i}
                        audioPath={audioPath}
                        start={ex.start}
                        end={ex.end}
                        text={ex.text}
                        suspect={ex.suspect}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          {/* Speaker map entries not already in segments */}
          {Object.keys(localMap).length > 0 && entries.length === 0 && (
            <div className="grid grid-cols-[auto_1fr_auto] gap-2 max-w-md items-center">
              {Object.keys(localMap).sort().map((key) => (
                <div key={key} className="contents">
                  <span className="text-xs text-muted-foreground">{key}</span>
                  <input
                    value={localMap[key] || ""}
                    onChange={(e) => setLocalMap({ ...localMap, [key]: e.target.value })}
                    placeholder={key}
                    className="input text-sm py-1"
                  />
                  <button
                    onClick={() => {
                      const next = { ...localMap };
                      delete next[key];
                      setLocalMap(next);
                    }}
                    className="text-muted-foreground hover:text-destructive p-0.5"
                    title="Remove"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Add new speaker */}
          <div className="flex items-center gap-2 max-w-md">
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addSpeaker()}
              placeholder="Add speaker..."
              className="input text-sm py-1 flex-1"
            />
            <Button
              onClick={addSpeaker}
              disabled={!newName.trim()}
              variant="ghost"
              size="sm"
              className="h-7 px-2"
            >
              <Plus className="w-3.5 h-3.5" />
            </Button>
          </div>

          <Button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || !isDirty}
            size="sm"
          >
            {mutation.isPending ? "Saving..." : "Save"}
          </Button>
        </div>
      )}
    </div>
  );
}
