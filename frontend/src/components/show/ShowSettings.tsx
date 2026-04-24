import { useState, useEffect, useRef, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import type { ShowMeta } from "@/api/types";
import { updateShowMeta, moveShow, deleteShow } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { useEpisodeStore } from "@/stores";
import { Button } from "@/components/ui/button";
import { SettingRow, SettingSection } from "@/components/ui/setting-row";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { errorMessage, inputWidth, selectClass } from "@/lib/utils";
import FolderLocationFields from "@/components/common/FolderLocationFields";
import PipelineSettings from "./PipelineSettings";
import ShowAccessSection from "./ShowAccessSection";
import { StatusDot } from "@/components/ui/status-dot";
import { AdvancedFieldset } from "@/components/ui/advanced-fieldset";
import { FolderOpen, Trash2 } from "lucide-react";

interface ShowSettingsProps {
  folder: string;
  meta: ShowMeta;
}

export default function ShowSettings({ folder, meta }: ShowSettingsProps) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  // ── Show info ──
  const [name, setName] = useState(meta.name);
  const [language, setLanguage] = useState(meta.language);
  const [rssUrl, setRssUrl] = useState(meta.rss_url);
  const [youtubeUrl, setYoutubeUrl] = useState(meta.youtube_url ?? "");
  const [artworkUrl, setArtworkUrl] = useState(meta.artwork_url);
  const [pipeModelSize, setPipeModelSize] = useState(meta.pipeline?.model_size ?? "");
  const [pipeDiarize, setPipeDiarize] = useState(meta.pipeline?.diarize ?? false);
  const [pipeLlmMode, setPipeLlmMode] = useState(meta.pipeline?.llm_mode ?? "");
  const [pipeLlmProvider, setPipeLlmProvider] = useState(meta.pipeline?.llm_provider ?? "");
  const [pipeLlmModel, setPipeLlmModel] = useState(meta.pipeline?.llm_model ?? "");
  const [pipeTargetLang, setPipeTargetLang] = useState(meta.pipeline?.target_lang ?? "");

  // ── Move folder ──
  const folderBasename = folder.split("/").filter(Boolean).pop() || folder;
  const folderParentDefault = folder.slice(0, folder.length - folderBasename.length).replace(/\/+$/, "") || "/";
  const moveFilesRef = useRef(true);
  const [folderName, setFolderName] = useState(folderBasename);
  const [parentPath, setParentPath] = useState(folderParentDefault);
  const destPath = `${parentPath.replace(/\/+$/, "")}/${folderName}`;
  const hasChanges = destPath !== folder;

  useEffect(() => {
    setName(meta.name);
    setLanguage(meta.language);
    setRssUrl(meta.rss_url);
    setYoutubeUrl(meta.youtube_url ?? "");
    setArtworkUrl(meta.artwork_url);
    setPipeModelSize(meta.pipeline?.model_size ?? "");
    setPipeDiarize(meta.pipeline?.diarize ?? false);
    setPipeLlmMode(meta.pipeline?.llm_mode ?? "");
    setPipeLlmProvider(meta.pipeline?.llm_provider ?? "");
    setPipeLlmModel(meta.pipeline?.llm_model ?? "");
    setPipeTargetLang(meta.pipeline?.target_lang ?? "");
  }, [meta]);

  const isDirty =
    name !== meta.name ||
    language !== meta.language ||
    rssUrl !== meta.rss_url ||
    youtubeUrl !== (meta.youtube_url ?? "") ||
    artworkUrl !== meta.artwork_url ||
    pipeModelSize !== (meta.pipeline?.model_size ?? "") ||
    pipeDiarize !== (meta.pipeline?.diarize ?? false) ||
    pipeLlmMode !== (meta.pipeline?.llm_mode ?? "") ||
    pipeLlmProvider !== (meta.pipeline?.llm_provider ?? "") ||
    pipeLlmModel !== (meta.pipeline?.llm_model ?? "") ||
    pipeTargetLang !== (meta.pipeline?.target_lang ?? "");

  const saveMutation = useMutation({
    mutationFn: () =>
      updateShowMeta(folder, {
        name,
        language,
        rss_url: rssUrl,
        youtube_url: youtubeUrl,
        speakers: meta.speakers,
        artwork_url: artworkUrl,
        pipeline: {
          model_size: pipeModelSize,
          diarize: pipeDiarize,
          llm_mode: pipeLlmMode,
          llm_provider: pipeLlmProvider,
          llm_model: pipeLlmModel,
          target_lang: pipeTargetLang,
        },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.showMeta(folder) });
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
    },
  });

  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const isDirtyRef = useRef(isDirty);
  // eslint-disable-next-line react-hooks/refs
  isDirtyRef.current = isDirty;
  const autoSave = useCallback(() => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      if (isDirtyRef.current) saveMutation.mutate();
    }, 1500);
  }, [saveMutation]);

  useEffect(() => {
    if (isDirty) autoSave();
    return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
  }, [name, language, rssUrl, youtubeUrl, artworkUrl, pipeModelSize, pipeDiarize, pipeLlmMode, pipeLlmProvider, pipeLlmModel, pipeTargetLang]); // eslint-disable-line react-hooks/exhaustive-deps

  const moveMutation = useMutation({
    mutationFn: ({ newPath, moveFiles: mf }: { newPath: string; moveFiles: boolean }) =>
      moveShow(folder, newPath, mf),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
      navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(data.new_path) } });
    },
  });

  const deleteFilesRef = useRef(false);
  const deleteMutation = useMutation({
    mutationFn: (deleteFiles: boolean) => deleteShow(folder, deleteFiles),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
      navigate({ to: "/" });
    },
  });

  const handleDelete = () => {
    deleteFilesRef.current = false;
    confirmDialog.open({
      title: "Remove this show?",
      description: `This will unregister "${meta.name}" from PodCodex.`,
      content: (
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            defaultChecked={false}
            onChange={(e) => { deleteFilesRef.current = e.target.checked; }}
            className="accent-destructive"
          />
          Also delete local files on disk
        </label>
      ),
      confirmLabel: "Remove",
      variant: "destructive",
      onConfirm: () => deleteMutation.mutate(deleteFilesRef.current),
    });
  };

  const handleMove = () => {
    if (!hasChanges) return;
    moveFilesRef.current = true;
    confirmDialog.open({
      title: "Move show folder?",
      description: `${folder}  →  ${destPath}`,
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
      onConfirm: () => moveMutation.mutate({ newPath: destPath, moveFiles: moveFilesRef.current }),
    });
  };

  // ── Episode filters ──
  const {
    minDurationMinutes, setMinDurationMinutes,
    maxDurationMinutes, setMaxDurationMinutes,
    titleInclude, setTitleInclude,
    titleExclude, setTitleExclude,
  } = useEpisodeStore();

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="p-6 space-y-8 max-w-2xl">
      {/* ── Show Info ── */}
      <SettingSection title="Show Info" description="Basic metadata for this show.">
        <SettingRow label="Name" help="Display name for this show.">
          <input value={name} onChange={(e) => setName(e.target.value)} className={`input ${inputWidth.medium}`} />
        </SettingRow>
        <SettingRow label="Language" help="Primary spoken language (e.g. French, English).">
          <input value={language} onChange={(e) => setLanguage(e.target.value)} className={`input ${inputWidth.short}`} />
        </SettingRow>
        <SettingRow label="RSS URL" help="The show's RSS feed URL.">
          <input value={rssUrl} onChange={(e) => setRssUrl(e.target.value)} placeholder="https://..." className={`input ${inputWidth.long}`} />
        </SettingRow>
        <SettingRow label="YouTube URL" help="YouTube channel or playlist URL.">
          <input value={youtubeUrl} onChange={(e) => setYoutubeUrl(e.target.value)} placeholder="https://youtube.com/..." className={`input ${inputWidth.long}`} />
        </SettingRow>
        <SettingRow label="Artwork" help="URL to the show cover image.">
          <div className="flex items-center gap-2">
            <input value={artworkUrl} onChange={(e) => setArtworkUrl(e.target.value)} placeholder="https://..." className={`input ${inputWidth.medium}`} />
            {artworkUrl && (
              <img src={artworkUrl} alt="Artwork preview" className="w-7 h-7 rounded object-cover shrink-0" onError={(e) => (e.currentTarget.style.display = "none")} />
            )}
          </div>
        </SettingRow>
      </SettingSection>

      {/* Save status */}
      {(isDirty || saveMutation.isSuccess || saveMutation.isError) && (
        <div className="flex items-center gap-2 text-xs -mt-4">
          {isDirty && (
            <>
              <StatusDot state="busy" />
              <span className="text-muted-foreground">Saving…</span>
            </>
          )}
          {saveMutation.isSuccess && !isDirty && (
            <>
              <StatusDot state="ok" />
              <span className="text-muted-foreground">Saved</span>
            </>
          )}
          {saveMutation.isError && (
            <>
              <StatusDot state="err" />
              <span className="text-destructive">{errorMessage(saveMutation.error)}</span>
            </>
          )}
        </div>
      )}

      {/* ── Folder Location ── */}
      <SettingSection title="Folder" description="Location of show files on disk.">
        <FolderLocationFields
          folderName={folderName}
          onFolderNameChange={setFolderName}
          parentPath={parentPath}
          onParentPathChange={setParentPath}
        />
        {(hasChanges || moveMutation.isPending || moveMutation.isError) && (
          <div className="flex items-center gap-3">
            <Button
              onClick={handleMove}
              disabled={moveMutation.isPending || !folderName.trim() || !hasChanges}
              size="sm"
              className="gap-1.5"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              {moveMutation.isPending ? "Moving..." : "Move folder to new location"}
            </Button>
            {moveMutation.isError && (
              <span className="text-xs text-destructive">{errorMessage(moveMutation.error)}</span>
            )}
          </div>
        )}
      </SettingSection>

      {/* ── Episode Filters ── */}
      <SettingSection title="Episode Filters" description="Filter which episodes appear in the list.">
        <SettingRow label="Min duration" help="Hide episodes shorter than this. 0 = no limit.">
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              min={0}
              step={5}
              value={minDurationMinutes || ""}
              onChange={(e) => setMinDurationMinutes(Math.max(0, Number(e.target.value)))}
              placeholder="0"
              className={`input ${inputWidth.numeric} text-center`}
            />
            <span className="text-xs text-muted-foreground">min</span>
          </div>
        </SettingRow>
        <SettingRow label="Max duration" help="Hide episodes longer than this. 0 = no limit.">
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              min={0}
              step={5}
              value={maxDurationMinutes || ""}
              onChange={(e) => setMaxDurationMinutes(Math.max(0, Number(e.target.value)))}
              placeholder="0"
              className={`input ${inputWidth.numeric} text-center`}
            />
            <span className="text-xs text-muted-foreground">min</span>
          </div>
        </SettingRow>
        <SettingRow label="Title contains" help="Only keep episodes whose title contains this text.">
          <input
            value={titleInclude}
            onChange={(e) => setTitleInclude(e.target.value)}
            placeholder="filter..."
            className={`input ${inputWidth.short}`}
          />
        </SettingRow>
        <SettingRow label="Title excludes" help="Hide episodes whose title contains this text.">
          <input
            value={titleExclude}
            onChange={(e) => setTitleExclude(e.target.value)}
            placeholder="exclude..."
            className={`input ${inputWidth.short}`}
          />
        </SettingRow>
      </SettingSection>

      <PipelineSettings language={language} />

      {/* ── Show Pipeline Defaults (overrides app-level for status comparison) ── */}
      <AdvancedFieldset
        legend="Show pipeline overrides"
        description="Pin this show to specific pipeline settings. Empty values fall back to the app-level defaults. Episodes run with different settings are flagged as outdated."
      >
        <SettingRow label="Whisper model" help="Expected transcription model. Empty = app default.">
          <input
            value={pipeModelSize}
            onChange={(e) => setPipeModelSize(e.target.value)}
            placeholder="(use default)"
            className={`input ${inputWidth.short}`}
          />
        </SettingRow>
        <SettingRow label="Diarize" help="Detect and label different speakers.">
          <input
            type="checkbox"
            checked={pipeDiarize}
            onChange={(e) => setPipeDiarize(e.target.checked)}
            className="accent-primary"
          />
        </SettingRow>
        <SettingRow label="LLM mode" help="Where the AI runs: Ollama (local) or a cloud API.">
          <select
            value={pipeLlmMode}
            onChange={(e) => setPipeLlmMode(e.target.value)}
            className={selectClass}
          >
            <option value="">(use default)</option>
            <option value="ollama">Ollama</option>
            <option value="api">API</option>
          </select>
        </SettingRow>
        <SettingRow label="LLM provider" help="Cloud API provider (openai, anthropic, etc.). Empty = app default.">
          <input
            value={pipeLlmProvider}
            onChange={(e) => setPipeLlmProvider(e.target.value)}
            placeholder="(use default)"
            className={`input ${inputWidth.short}`}
          />
        </SettingRow>
        <SettingRow label="LLM model" help="Specific model name. Empty = app default.">
          <input
            value={pipeLlmModel}
            onChange={(e) => setPipeLlmModel(e.target.value)}
            placeholder="(use default)"
            className={`input ${inputWidth.short}`}
          />
        </SettingRow>
        <SettingRow label="Target language" help="Translation target language. Empty = app default.">
          <input
            value={pipeTargetLang}
            onChange={(e) => setPipeTargetLang(e.target.value)}
            placeholder="(use default)"
            className={`input ${inputWidth.short}`}
          />
        </SettingRow>
      </AdvancedFieldset>

      {/* ── Discord bot access ── */}
      <ShowAccessSection show={meta.name} />


      {/* ── Danger Zone ── */}
      <SettingSection title="Danger Zone" description="Irreversible actions.">
        <div className="flex items-center gap-3">
          <Button
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
            variant="destructive"
            size="sm"
            className="gap-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" />
            {deleteMutation.isPending ? "Removing..." : "Remove show"}
          </Button>
          {deleteMutation.isError && (
            <span className="text-xs text-destructive">{errorMessage(deleteMutation.error)}</span>
          )}
        </div>
      </SettingSection>

      </div>
    </div>
  );
}
