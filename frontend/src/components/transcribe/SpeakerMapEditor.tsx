import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, useEffect } from "react";
import { getSpeakerMap, saveSpeakerMap, getSegments, saveSegments } from "@/api/client";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronRight, Plus, X } from "lucide-react";

interface SpeakerMapEditorProps {
  audioPath: string;
  onSaved?: () => void;
}

export default function SpeakerMapEditor({ audioPath, onSaved }: SpeakerMapEditorProps) {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const [newName, setNewName] = useState("");

  // Load speaker map (raw diarization IDs → names)
  const { data: serverMap } = useQuery({
    queryKey: ["speaker-map", audioPath],
    queryFn: () => getSpeakerMap(audioPath),
  });

  // Load segments to get current speaker names
  const { data: segments } = useQuery({
    queryKey: ["transcribe", "segments", audioPath],
    queryFn: () => getSegments(audioPath),
  });

  // Speakers from segments (actual names used)
  const segmentSpeakers = (() => {
    if (!segments) return [];
    const set = new Set<string>();
    for (const seg of segments) {
      if (seg.speaker && seg.speaker !== "[BREAK]") set.add(seg.speaker);
    }
    return Array.from(set).sort();
  })();

  // Local state: list of {current, renamed} entries
  const [entries, setEntries] = useState<{ current: string; renamed: string }[]>([]);

  // Also keep the raw speaker map for saving
  const [localMap, setLocalMap] = useState<Record<string, string>>({});

  useEffect(() => {
    if (serverMap) setLocalMap(serverMap);
  }, [serverMap]);

  // Build entries from segment speakers
  useEffect(() => {
    if (segmentSpeakers.length > 0) {
      setEntries(segmentSpeakers.map((s) => ({ current: s, renamed: s })));
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
          const updated = segments.map((seg) => {
            const newName = renameMap.get(seg.speaker);
            return newName ? { ...seg, speaker: newName } : seg;
          });
          await saveSegments(audioPath, updated);
        }
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["speaker-map", audioPath] });
      queryClient.invalidateQueries({ queryKey: ["transcribe", "segments", audioPath] });
      onSaved?.();
    },
  });

  const addSpeaker = () => {
    const name = newName.trim();
    if (!name || entries.some((e) => e.renamed === name)) return;
    setEntries([...entries, { current: name, renamed: name }]);
    // Also add to speaker map so it persists
    setLocalMap({ ...localMap, [name]: name });
    setNewName("");
  };

  const removeSpeaker = (idx: number) => {
    const entry = entries[idx];
    const next = entries.filter((_, i) => i !== idx);
    setEntries(next);
    // Remove from local map too
    const nextMap = { ...localMap };
    delete nextMap[entry.current];
    setLocalMap(nextMap);
  };

  const updateRenamed = (idx: number, value: string) => {
    const next = [...entries];
    next[idx] = { ...next[idx], renamed: value };
    setEntries(next);
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
      </button>

      {expanded && (
        <div className="px-4 pb-3 space-y-3">
          {entries.length > 0 && (
            <div className="grid grid-cols-[1fr_auto_1fr_auto] gap-2 max-w-lg items-center text-sm">
              {entries.map((entry, idx) => (
                <div key={entry.current} className="contents">
                  <span className="text-xs text-muted-foreground truncate">{entry.current}</span>
                  <span className="text-xs text-muted-foreground">→</span>
                  <input
                    value={entry.renamed}
                    onChange={(e) => updateRenamed(idx, e.target.value)}
                    placeholder={entry.current}
                    className="input text-sm py-1"
                  />
                  <button
                    onClick={() => removeSpeaker(idx)}
                    className="text-muted-foreground hover:text-destructive p-0.5"
                    title="Remove speaker"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

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
