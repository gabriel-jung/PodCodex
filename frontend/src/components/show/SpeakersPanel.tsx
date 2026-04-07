import { useState, useEffect, useRef, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { ShowMeta } from "@/api/types";
import { updateShowMeta } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { errorMessage } from "@/lib/utils";

interface SpeakersPanelProps {
  folder: string;
  meta: ShowMeta;
}

export default function SpeakersPanel({ folder, meta }: SpeakersPanelProps) {
  const queryClient = useQueryClient();
  const [speakers, setSpeakers] = useState<string[]>(meta.speakers);
  const [newSpeaker, setNewSpeaker] = useState("");

  useEffect(() => {
    setSpeakers(meta.speakers);
  }, [meta.speakers]);

  const isDirty = JSON.stringify(speakers) !== JSON.stringify(meta.speakers);

  const saveMutation = useMutation({
    mutationFn: () =>
      updateShowMeta(folder, { ...meta, speakers }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.showMeta(folder) });
    },
  });

  const saveTimer = useRef<ReturnType<typeof setTimeout>>();
  const autoSave = useCallback(() => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => saveMutation.mutate(), 1500);
  }, [saveMutation]);

  useEffect(() => {
    if (isDirty) autoSave();
    return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
  }, [speakers]); // eslint-disable-line react-hooks/exhaustive-deps

  const addSpeaker = () => {
    const trimmed = newSpeaker.trim();
    if (trimmed && !speakers.includes(trimmed)) {
      setSpeakers([...speakers, trimmed]);
      setNewSpeaker("");
    }
  };

  const removeSpeaker = (speaker: string) => {
    setSpeakers(speakers.filter((s) => s !== speaker));
  };

  return (
    <div className="p-6 space-y-4 max-w-xl">
      <div>
        <h3 className="text-sm font-medium">Speakers</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Known speakers in this podcast. Used for speaker labeling and as LLM context.
        </p>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {speakers.map((s) => (
          <span
            key={s}
            className="inline-flex items-center gap-1 bg-secondary text-secondary-foreground rounded-full px-2.5 py-1 text-xs border border-border"
          >
            {s}
            <button
              onClick={() => removeSpeaker(s)}
              className="text-muted-foreground hover:text-foreground transition"
            >
              <X className="w-3 h-3" />
            </button>
          </span>
        ))}
        {speakers.length === 0 && (
          <span className="text-xs text-muted-foreground">No speakers defined yet.</span>
        )}
      </div>

      <div className="flex gap-2">
        <input
          value={newSpeaker}
          onChange={(e) => setNewSpeaker(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addSpeaker())}
          placeholder="Add speaker..."
          className="input py-1 text-sm flex-1"
        />
        <Button onClick={addSpeaker} variant="outline" size="sm" disabled={!newSpeaker.trim()}>
          Add
        </Button>
      </div>

      {/* Save status */}
      <div className="flex items-center gap-3 text-xs">
        {isDirty && <span className="text-yellow-400">Saving...</span>}
        {saveMutation.isSuccess && !isDirty && <span className="text-success">Saved</span>}
        {saveMutation.isError && <span className="text-destructive">{errorMessage(saveMutation.error)}</span>}
      </div>
    </div>
  );
}
