import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, useEffect, useMemo, useRef } from "react";
import { saveSpeakerMap, getSegments, saveSegments } from "@/api/client";
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
  excerpts: Excerpt[];
}

interface SpeakerEntry {
  /** Original speaker ID in the current segments. */
  current: string;
  /** New name (empty = unchanged). */
  renamed: string;
  /** If true, all segments for this speaker will be removed on save. */
  removed: boolean;
}

// Matches raw diarizer output like SPEAKER_00, SPEAKER_12, or empty string.
// Named differently from useSegmentFiltering's UNKNOWN_SPEAKERS set because
// the two track different concepts (default diarizer IDs vs. backend unknown markers).
const DIARIZER_DEFAULT_RE = /^SPEAKER_\d+$/;
const isDiarizerDefault = (id: string) => id === "" || DIARIZER_DEFAULT_RE.test(id);

function ExcerptRow({ audioPath, start, end, text, suspect }: { audioPath: string; start: number; end: number; text: string; suspect: boolean }) {
  const seekTo = useAudioStore((s) => s.seekTo);
  const pauseAudio = useAudioStore((s) => s.pauseAudio);
  // Derive isActive in the selector so this row only re-renders at boundary crossings,
  // not on every audio tick.
  const isActive = useAudioStore(
    (s) => s.isPlaying && s.currentTime >= start - 0.5 && s.currentTime <= end + 0.5
  );
  const dur = end - start;
  const handlePlay = () => seekTo(audioPath, start);

  return (
    <div
      className={`flex items-center gap-2 px-2 py-1.5 rounded text-xs transition-colors ${
        isActive ? "bg-primary/10 border-l-2 border-l-primary" : "bg-muted/50"
      }`}
    >
      <button
        onClick={isActive ? pauseAudio : handlePlay}
        className="shrink-0 text-muted-foreground hover:text-foreground transition"
        title={isActive ? "Pause" : `Play from ${formatTime(start)}`}
      >
        {isActive ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
      </button>
      <span className="tabular-nums text-muted-foreground shrink-0">
        {formatTime(start, false)} – {formatTime(end, false)}
      </span>
      <span className="text-muted-foreground shrink-0">({dur.toFixed(1)}s)</span>
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

  const { data: segments } = useQuery({
    queryKey: queryKeys.transcribeSegments(audioPath),
    queryFn: () => getSegments(audioPath),
  });

  const speakerInfos: SpeakerInfo[] = useMemo(() => {
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
          excerpts: [...clean, ...suspects].slice(0, 5),
        };
      });
  }, [segments]);

  const { hasUnnamed, unnamedCount } = useMemo(() => {
    let count = 0;
    for (const s of speakerInfos) if (isDiarizerDefault(s.id)) count++;
    return { hasUnnamed: count > 0, unnamedCount: count };
  }, [speakerInfos]);

  const infoById = useMemo(
    () => new Map(speakerInfos.map((s) => [s.id, s])),
    [speakerInfos],
  );

  const autoExpandedRef = useRef(false);
  useEffect(() => {
    if (!autoExpandedRef.current && hasUnnamed && speakerInfos.length > 0) {
      autoExpandedRef.current = true;
      setExpanded(true);
    }
  }, [hasUnnamed, speakerInfos.length]);

  const [entries, setEntries] = useState<SpeakerEntry[]>([]);
  const [initialized, setInitialized] = useState(false);

  // Hydrate once per audioPath from server segments; re-hydrate after save via onSuccess.
  useEffect(() => {
    if (!initialized && speakerInfos.length > 0) {
      setEntries(speakerInfos.map((s) => ({ current: s.id, renamed: s.id, removed: false })));
      setInitialized(true);
    }
  }, [initialized, speakerInfos]);

  const isDirty = entries.some((e) => e.current !== e.renamed || e.removed);

  const mutation = useMutation({
    mutationFn: async () => {
      if (!segments) return;

      const renameMap = new Map<string, string>();
      const removeSet = new Set<string>();
      for (const e of entries) {
        if (e.removed) {
          removeSet.add(e.current);
        } else if (e.current !== e.renamed && e.renamed.trim()) {
          renameMap.set(e.current, e.renamed.trim());
        }
      }

      const speakerMap: Record<string, string> = {};
      for (const [from, to] of renameMap) speakerMap[from] = to;

      const needsSegmentWrite = renameMap.size > 0 || removeSet.size > 0;
      const updatedSegments = needsSegmentWrite
        ? segments
            .filter((seg: Record<string, unknown>) => !removeSet.has(seg.speaker as string))
            .map((seg: Record<string, unknown>) => {
              const newName = renameMap.get(seg.speaker as string);
              return newName ? { ...seg, speaker: newName } : seg;
            })
        : null;

      await Promise.all([
        updatedSegments ? saveSegments(audioPath, updatedSegments) : Promise.resolve(),
        saveSpeakerMap(audioPath, speakerMap),
      ]);
    },
    onSuccess: () => {
      setInitialized(false);
      queryClient.invalidateQueries({ queryKey: queryKeys.transcribeSegments(audioPath) });
      onSaved?.();
    },
  });

  const addSpeaker = () => {
    const name = newName.trim();
    if (!name || entries.some((e) => e.current === name || e.renamed === name)) return;
    setEntries((prev) => [...prev, { current: name, renamed: name, removed: false }]);
    setNewName("");
  };

  const updateRenamed = (idx: number, value: string) => {
    setEntries((prev) => prev.map((e, i) => (i === idx ? { ...e, renamed: value } : e)));
  };

  const toggleRemoved = (idx: number) => {
    setEntries((prev) => prev.map((e, i) => (i === idx ? { ...e, removed: !e.removed } : e)));
  };

  const toggleSpeakerExcerpts = (id: string) => {
    setExpandedSpeakers((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const displayCount = entries.length;
  const removedCount = entries.filter((e) => e.removed).length;

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
            const info = infoById.get(entry.current);
            const isExpanded = expandedSpeakers.has(entry.current);
            const isUnnamed = isDiarizerDefault(entry.current);

            return (
              <div
                key={entry.current}
                className={`border rounded-lg overflow-hidden ${
                  entry.removed ? "border-destructive/40 bg-destructive/5" :
                  isUnnamed ? "border-yellow-500/30" : "border-border"
                }`}
              >
                <div className="flex items-center gap-2 px-3 py-2">
                  <button
                    onClick={() => toggleSpeakerExcerpts(entry.current)}
                    className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition shrink-0"
                    title="Show audio excerpts"
                  >
                    {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    <span className={`font-medium ${entry.removed ? "line-through text-muted-foreground" : ""}`}>
                      {entry.current}
                    </span>
                    {info && <span className="text-muted-foreground">({info.segCount})</span>}
                    {info && info.suspectCount > 0 && (
                      <span className="text-yellow-500" title={`${info.suspectCount} suspect segments`}>
                        ⚠ {info.suspectCount}
                      </span>
                    )}
                  </button>

                  {!entry.removed && (
                    <>
                      <span className="text-xs text-muted-foreground">→</span>
                      <input
                        value={entry.renamed}
                        onChange={(e) => updateRenamed(idx, e.target.value)}
                        placeholder="Enter name..."
                        className="input text-sm py-1 flex-1 min-w-0"
                      />
                    </>
                  )}

                  {entry.removed && (
                    <span className="text-xs text-destructive italic flex-1">
                      {info ? `${info.segCount} segments will be removed` : "will be removed"}
                    </span>
                  )}

                  <button
                    onClick={() => toggleRemoved(idx)}
                    className={`p-0.5 shrink-0 ${entry.removed ? "text-destructive" : "text-muted-foreground hover:text-destructive"}`}
                    title={entry.removed ? "Undo remove" : "Remove all segments for this speaker"}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {isExpanded && !entry.removed && info && info.excerpts.length > 0 && (
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

          <div className="flex items-center gap-3">
            <Button
              onClick={() => mutation.mutate()}
              disabled={mutation.isPending || !isDirty}
              size="sm"
            >
              {mutation.isPending ? "Saving..." : "Save"}
            </Button>
            {removedCount > 0 && (
              <span className="text-xs text-destructive">
                {removedCount} speaker{removedCount > 1 ? "s" : ""} marked for removal
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
