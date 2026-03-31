import { useState, useEffect, useRef, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import type { ShowMeta } from "@/api/types";
import { updateShowMeta, syncToQdrant, moveShow } from "@/api/client";
import { useConfigStore } from "@/stores";
import { Button } from "@/components/ui/button";
import { SettingRow, SettingSection } from "@/components/ui/setting-row";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { errorMessage } from "@/lib/utils";
import SectionHeader from "@/components/common/SectionHeader";
import ProgressBar from "@/components/editor/ProgressBar";
import FolderPicker from "@/components/common/FolderPicker";
import PipelineSettings from "./PipelineSettings";
import { FolderOpen } from "lucide-react";

interface ShowSettingsProps {
  folder: string;
  meta: ShowMeta;
  hasIndex: boolean;
}

export default function ShowSettings({ folder, meta, hasIndex }: ShowSettingsProps) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  // ── Show info ──
  const [name, setName] = useState(meta.name);
  const [language, setLanguage] = useState(meta.language);
  const [rssUrl, setRssUrl] = useState(meta.rss_url);
  const [artworkUrl, setArtworkUrl] = useState(meta.artwork_url);
  const [syncTaskId, setSyncTaskId] = useState<string | null>(null);
  const [overwrite, setOverwrite] = useState(false);

  // ── Move folder ──
  const folderBasename = folder.split("/").filter(Boolean).pop() || folder;
  const folderParent = folder.slice(0, folder.length - folderBasename.length).replace(/\/+$/, "") || "/";
  const [pickerOpen, setPickerOpen] = useState(false);
  const moveFilesRef = useRef(true);
  const [folderName, setFolderName] = useState(folderBasename);

  useEffect(() => {
    setName(meta.name);
    setLanguage(meta.language);
    setRssUrl(meta.rss_url);
    setArtworkUrl(meta.artwork_url);
  }, [meta]);

  const isDirty =
    name !== meta.name ||
    language !== meta.language ||
    rssUrl !== meta.rss_url ||
    artworkUrl !== meta.artwork_url;

  const saveMutation = useMutation({
    mutationFn: () =>
      updateShowMeta(folder, {
        name,
        language,
        rss_url: rssUrl,
        speakers: meta.speakers,
        artwork_url: artworkUrl,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["showMeta", folder] });
      queryClient.invalidateQueries({ queryKey: ["shows"] });
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
  }, [name, language, rssUrl, artworkUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  const syncMutation = useMutation({
    mutationFn: () =>
      syncToQdrant({ folder, show: meta.name || name, overwrite }),
    onSuccess: (data) => setSyncTaskId(data.task_id),
  });

  const moveMutation = useMutation({
    mutationFn: ({ newPath, moveFiles: mf }: { newPath: string; moveFiles: boolean }) =>
      moveShow(folder, newPath, mf),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["shows"] });
      navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(data.new_path) } });
    },
  });

  const handlePickedFolder = (parentPath: string) => {
    const dest = `${parentPath.replace(/\/+$/, "")}/${folderName}`;
    if (dest === folder) return;
    moveFilesRef.current = true;
    confirmDialog.open({
      title: "Move show folder?",
      description: `${folder}  →  ${dest}`,
      content: (
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            defaultChecked={true}
            onChange={(e) => { moveFilesRef.current = e.target.checked; }}
            className="accent-primary"
          />
          Move all files to the new location
        </label>
      ),
      confirmLabel: "Move",
      variant: "destructive",
      onConfirm: () => moveMutation.mutate({ newPath: dest, moveFiles: moveFilesRef.current }),
    });
  };

  // ── Episode filters ──
  const {
    minDurationMinutes, setMinDurationMinutes,
    maxDurationMinutes, setMaxDurationMinutes,
    titleInclude, setTitleInclude,
    titleExclude, setTitleExclude,
  } = useConfigStore();

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-8 max-w-2xl">
      {/* ── Show Info ── */}
      <SettingSection title="Show Info" description="Basic metadata for this podcast.">
        <SettingRow label="Name" help="Display name for this podcast.">
          <input value={name} onChange={(e) => setName(e.target.value)} className="input py-1 text-sm w-48" />
        </SettingRow>
        <SettingRow label="Language" help="Primary spoken language (e.g. French, English).">
          <input value={language} onChange={(e) => setLanguage(e.target.value)} className="input py-1 text-sm w-32" />
        </SettingRow>
        <SettingRow label="RSS URL" help="The podcast's RSS feed URL.">
          <input value={rssUrl} onChange={(e) => setRssUrl(e.target.value)} placeholder="https://..." className="input py-1 text-sm w-64" />
        </SettingRow>
        <SettingRow label="Artwork" help="URL to the podcast cover image.">
          <div className="flex items-center gap-2">
            <input value={artworkUrl} onChange={(e) => setArtworkUrl(e.target.value)} placeholder="https://..." className="input py-1 text-sm w-48" />
            {artworkUrl && (
              <img src={artworkUrl} alt="" className="w-7 h-7 rounded object-cover shrink-0" onError={(e) => (e.currentTarget.style.display = "none")} />
            )}
          </div>
        </SettingRow>
      </SettingSection>

      {/* Save status */}
      {(isDirty || saveMutation.isSuccess || saveMutation.isError) && (
        <div className="flex items-center gap-3 text-xs -mt-4">
          {isDirty && <span className="text-yellow-400">Saving...</span>}
          {saveMutation.isSuccess && !isDirty && <span className="text-green-400">Saved</span>}
          {saveMutation.isError && <span className="text-destructive">{errorMessage(saveMutation.error)}</span>}
        </div>
      )}

      {/* ── Folder Location ── */}
      <SettingSection title="Folder" description="Location of show files on disk.">
        <SettingRow label="Current path">
          <span className="text-xs text-muted-foreground font-mono truncate max-w-xs" title={folder}>{folder}</span>
        </SettingRow>
        <SettingRow label="Folder name" help="Rename the show folder (applied on move).">
          <input
            value={folderName}
            onChange={(e) => setFolderName(e.target.value)}
            className="input py-1 text-sm w-48 font-mono"
          />
        </SettingRow>
        <div className="flex items-center gap-3">
          <Button
            onClick={() => setPickerOpen(true)}
            disabled={moveMutation.isPending || !folderName.trim()}
            variant="outline"
            size="sm"
            className="gap-1.5"
          >
            <FolderOpen className="w-3.5 h-3.5" />
            {moveMutation.isPending ? "Moving..." : "Move folder"}
          </Button>
          {moveMutation.isError && (
            <span className="text-xs text-destructive">{errorMessage(moveMutation.error)}</span>
          )}
        </div>
      </SettingSection>

      <FolderPicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onSelect={handlePickedFolder}
        initialPath={folderParent}
        title={`Move "${folderName}" to...`}
        description="Select the parent directory"
      />

      {/* ── Episode Filters ── */}
      <SettingSection title="Episode Filters" description="Filter which episodes are shown in the list.">
        <SettingRow label="Min duration" help="Hide episodes shorter than this (minutes). 0 = show all.">
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              min={0}
              step={5}
              value={minDurationMinutes || ""}
              onChange={(e) => setMinDurationMinutes(Math.max(0, Number(e.target.value)))}
              placeholder="0"
              className="input w-16 py-1 text-sm text-center"
            />
            <span className="text-xs text-muted-foreground">min</span>
          </div>
        </SettingRow>
        <SettingRow label="Max duration" help="Hide episodes longer than this (minutes). 0 = no limit.">
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              min={0}
              step={5}
              value={maxDurationMinutes || ""}
              onChange={(e) => setMaxDurationMinutes(Math.max(0, Number(e.target.value)))}
              placeholder="0"
              className="input w-16 py-1 text-sm text-center"
            />
            <span className="text-xs text-muted-foreground">min</span>
          </div>
        </SettingRow>
        <SettingRow label="Title contains" help="Only show episodes whose title contains this text.">
          <input
            value={titleInclude}
            onChange={(e) => setTitleInclude(e.target.value)}
            placeholder="filter..."
            className="input py-1 text-sm w-40"
          />
        </SettingRow>
        <SettingRow label="Title excludes" help="Hide episodes whose title contains this text.">
          <input
            value={titleExclude}
            onChange={(e) => setTitleExclude(e.target.value)}
            placeholder="exclude..."
            className="input py-1 text-sm w-40"
          />
        </SettingRow>
      </SettingSection>

      <PipelineSettings language={language} />

      {/* ── Qdrant Sync ── */}
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
                <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} className="accent-primary" />
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
