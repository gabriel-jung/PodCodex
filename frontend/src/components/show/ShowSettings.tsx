import { useState, useEffect, useRef, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { ShowMeta } from "@/api/types";
import { updateShowMeta, syncToQdrant } from "@/api/client";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { errorMessage } from "@/lib/utils";
import HelpLabel from "@/components/common/HelpLabel";
import SectionHeader from "@/components/common/SectionHeader";
import ProgressBar from "@/components/editor/ProgressBar";

interface ShowSettingsProps {
  folder: string;
  meta: ShowMeta;
  hasIndex: boolean;
}

export default function ShowSettings({ folder, meta, hasIndex }: ShowSettingsProps) {
  const queryClient = useQueryClient();

  const [name, setName] = useState(meta.name);
  const [language, setLanguage] = useState(meta.language);
  const [rssUrl, setRssUrl] = useState(meta.rss_url);
  const [artworkUrl, setArtworkUrl] = useState(meta.artwork_url);
  const [speakers, setSpeakers] = useState<string[]>(meta.speakers);
  const [newSpeaker, setNewSpeaker] = useState("");
  const [syncTaskId, setSyncTaskId] = useState<string | null>(null);
  const [overwrite, setOverwrite] = useState(false);

  // Reset form when meta changes (e.g. after save)
  useEffect(() => {
    setName(meta.name);
    setLanguage(meta.language);
    setRssUrl(meta.rss_url);
    setArtworkUrl(meta.artwork_url);
    setSpeakers(meta.speakers);
  }, [meta]);

  const isDirty =
    name !== meta.name ||
    language !== meta.language ||
    rssUrl !== meta.rss_url ||
    artworkUrl !== meta.artwork_url ||
    JSON.stringify(speakers) !== JSON.stringify(meta.speakers);

  const saveMutation = useMutation({
    mutationFn: () =>
      updateShowMeta(folder, {
        name,
        language,
        rss_url: rssUrl,
        speakers,
        artwork_url: artworkUrl,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["showMeta", folder] });
      queryClient.invalidateQueries({ queryKey: ["shows"] });
    },
  });

  // Debounced auto-save: saves 1.5s after last change
  const saveTimer = useRef<ReturnType<typeof setTimeout>>();
  const autoSave = useCallback(() => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => saveMutation.mutate(), 1500);
  }, [saveMutation]);

  useEffect(() => {
    if (isDirty) autoSave();
    return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
  }, [name, language, rssUrl, artworkUrl, speakers]);

  const syncMutation = useMutation({
    mutationFn: () =>
      syncToQdrant({ folder, show: meta.name || name, overwrite }),
    onSuccess: (data) => setSyncTaskId(data.task_id),
  });

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
    <div className="p-6 space-y-6 max-w-3xl">
      {/* Form — two columns on wide screens */}
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left: basic fields */}
        <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm flex-1">
          <HelpLabel label="Name" help="Display name for this podcast." />
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="input py-1 text-sm"
          />

          <HelpLabel label="Language" help="Primary spoken language of the podcast (e.g. French, English). Used as the default for transcription and polishing." />
          <input
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="input py-1 text-sm"
          />

          <HelpLabel label="RSS URL" help="The podcast's RSS feed URL. Used to fetch episode metadata and download audio." />
          <input
            value={rssUrl}
            onChange={(e) => setRssUrl(e.target.value)}
            placeholder="https://..."
            className="input py-1 text-sm"
          />

          <HelpLabel label="Artwork URL" help="URL to the podcast cover image. Shown in the show list and episode pages." />
          <div className="flex items-center gap-2">
            <input
              value={artworkUrl}
              onChange={(e) => setArtworkUrl(e.target.value)}
              placeholder="https://..."
              className="input py-1 text-sm flex-1"
            />
            {artworkUrl && (
              <img
                src={artworkUrl}
                alt="artwork"
                className="w-8 h-8 rounded object-cover shrink-0"
                onError={(e) => (e.currentTarget.style.display = "none")}
              />
            )}
          </div>
        </div>

        {/* Right: speakers */}
        <div className="flex flex-col gap-2 lg:flex-1 lg:border-l lg:border-border lg:pl-6">
          <HelpLabel label="Speakers" help="Known speakers in this podcast. Used for speaker labeling and as LLM context." />
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
        </div>
      </div>

      {/* Save status */}
      <div className="flex items-center gap-3">
        {isDirty && (
          <>
            <span className="text-xs text-yellow-400">Saving...</span>
            <Button onClick={() => { if (saveTimer.current) clearTimeout(saveTimer.current); saveMutation.mutate(); }} variant="ghost" size="sm" className="text-xs">
              Save now
            </Button>
          </>
        )}
        {saveMutation.isSuccess && !isDirty && (
          <span className="text-xs text-green-400">Saved</span>
        )}
        {saveMutation.isError && (
          <span className="text-xs text-destructive">{errorMessage(saveMutation.error)}</span>
        )}
      </div>

      {/* Qdrant sync */}
      {hasIndex && (
        <div className="border-t border-border pt-6 space-y-3">
          <SectionHeader>Qdrant Sync</SectionHeader>
          <p className="text-xs text-muted-foreground">
            Push indexed episodes from the local database to Qdrant for faster search across large collections.
          </p>

          {syncTaskId ? (
            <ProgressBar taskId={syncTaskId} onComplete={() => setSyncTaskId(null)} />
          ) : (
            <div className="flex items-center gap-3">
              <Button
                onClick={() => syncMutation.mutate()}
                disabled={syncMutation.isPending}
                variant="outline"
                size="sm"
              >
                {syncMutation.isPending ? "Starting..." : "Sync to Qdrant"}
              </Button>
              <label className="flex items-center gap-1.5 cursor-pointer text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  checked={overwrite}
                  onChange={(e) => setOverwrite(e.target.checked)}
                  className="accent-primary"
                />
                Overwrite existing
              </label>
              {syncMutation.isError && (
                <span className="text-xs text-destructive">{errorMessage(syncMutation.error)}</span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
