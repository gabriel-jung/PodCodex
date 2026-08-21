import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import type { ShowMeta } from "@/api/types";
import { updateShowMeta, moveShow, deleteShow, previewBroadcastNumber, uploadShowArtwork, deleteShowArtwork } from "@/api/client";
import { artworkUrl as showArtworkEndpoint } from "@/api/filesystem";
import { LOCAL_ARTWORK_MARKER } from "@/lib/showArtwork";
import { removeQueriesForShowName, removeQueriesUnderPath } from "@/api/cacheInvalidation";
import { queryKeys } from "@/api/queryKeys";
import { useIndexConfig } from "@/hooks/useIndexConfig";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { useProviderProfiles } from "@/hooks/useProviderProfiles";
import { useApiKeys } from "@/hooks/useApiKeys";
import { Button } from "@/components/ui/button";
import { SettingRow, SettingSection } from "@/components/ui/setting-row";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { errorMessage, inputWidth, selectClass, splitPath } from "@/lib/utils";
import FolderLocationFields from "@/components/common/FolderLocationFields";
import ShowAccessSection from "./ShowAccessSection";
import BundleExportSection from "./BundleExportSection";
import { StatusDot } from "@/components/ui/status-dot";
import { FolderOpen, Trash2, Upload } from "lucide-react";

interface ShowSettingsProps {
  folder: string;
  meta: ShowMeta;
}

function sameModels(a: Record<string, string>, b: Record<string, string>): boolean {
  const ka = Object.keys(a).filter((k) => a[k]);
  const kb = Object.keys(b).filter((k) => b[k]);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => a[k] === b[k]);
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
  // Mirrors of the above for the resync effect, which must not take artworkUrl
  // as a dependency (it sets it).
  const artworkUrlRef = useRef(artworkUrl);
  // True only while the user has typed a URL that has not reached the server.
  const artworkTouchedRef = useRef(false);
  const [broadcastPattern, setBroadcastPattern] = useState(meta.broadcast_number_pattern ?? "");
  // ── Per-show pipeline config (show.toml [pipeline]) ──
  // Empty string / null means "inherit the app default" (Settings, Pipeline).
  const [pipeModelSize, setPipeModelSize] = useState(meta.pipeline?.model_size ?? "");
  const [pipeDiarize, setPipeDiarize] = useState<boolean | null>(meta.pipeline?.diarize ?? null);
  const [pipeNumSpeakers, setPipeNumSpeakers] = useState(meta.pipeline?.num_speakers ?? "");
  const [pipeLlmMode, setPipeLlmMode] = useState(meta.pipeline?.llm_mode ?? "");
  const [pipeLlmProviderProfile, setPipeLlmProviderProfile] = useState(meta.pipeline?.llm_provider_profile ?? "");
  const [pipeLlmKeyName, setPipeLlmKeyName] = useState(meta.pipeline?.llm_key_name ?? "");
  const [pipeLlmModels, setPipeLlmModels] = useState<Record<string, string>>(
    meta.pipeline?.llm_models_by_mode ?? {},
  );
  const [pipeLlmBatchMinutes, setPipeLlmBatchMinutes] = useState<string>(
    meta.pipeline?.llm_batch_minutes != null ? String(meta.pipeline.llm_batch_minutes) : "",
  );
  const [pipeContext, setPipeContext] = useState(meta.pipeline?.context ?? "");
  const [pipeTargetLang, setPipeTargetLang] = useState(meta.pipeline?.target_lang ?? "");
  const [pipeRagModel, setPipeRagModel] = useState(meta.pipeline?.rag_model ?? "");
  const [pipeRagChunker, setPipeRagChunker] = useState(meta.pipeline?.rag_chunker ?? "");

  const { data: indexConfig } = useIndexConfig();
  const { whisperModels } = useLLMProviders();
  const { profiles } = useProviderProfiles();
  const { keys } = useApiKeys();
  const apiProfiles = useMemo(() => profiles.filter((p) => p.type !== "ollama"), [profiles]);

  // ── Artwork upload ──
  const isLocalArtwork = artworkUrl === LOCAL_ARTWORK_MARKER;
  const artworkFileRef = useRef<HTMLInputElement | null>(null);
  const artworkUploadMutation = useMutation({
    mutationFn: (file: File) => uploadShowArtwork(folder, file),
    onSuccess: () => {
      // Adopt the marker the upload just wrote server-side. Without this the
      // form stays dirty against the refetched meta and the debounced save
      // PUTs the pre-upload artwork_url back, orphaning the uploaded file.
      artworkTouchedRef.current = false;
      setArtworkUrl(LOCAL_ARTWORK_MARKER);
      queryClient.invalidateQueries({ queryKey: queryKeys.showMeta(folder) });
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
    },
  });
  // Feed-backed shows get the feed's own cover back on the next refresh, so
  // the same endpoint reads as "reset" there and "remove" on a local show.
  // One copy object rather than a ternary per string, so the button label can
  // never disagree with the dialog it opens.
  const artworkCopy = (!!meta.rss_url || !!meta.youtube_url)
    ? {
        action: "Reset",
        hint: "Reset to the feed's artwork",
        title: "Reset to feed artwork?",
        description:
          "The uploaded image is deleted and the feed's own artwork comes back on the next refresh.",
      }
    : {
        action: "Remove",
        hint: "Remove the cover image",
        title: "Remove this cover?",
        description:
          "The uploaded image is deleted and the default cover is used instead.",
      };
  const artworkRemoveMutation = useMutation({
    mutationFn: () => deleteShowArtwork(folder),
    onSuccess: () => {
      // Same reason the upload adopts the marker: leaving the old value in the
      // form would let the debounced save PUT it straight back.
      artworkTouchedRef.current = false;
      setArtworkUrl("");
      queryClient.invalidateQueries({ queryKey: queryKeys.showMeta(folder) });
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
    },
  });

  const handleArtworkFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) artworkUploadMutation.mutate(file);
    e.target.value = "";
  };

  // ── Move folder ──
  const { parent: folderParentDefault, basename: folderBasename, sep: pathSep } = splitPath(folder);
  const moveFilesRef = useRef(true);
  const [folderName, setFolderName] = useState(folderBasename);
  const [parentPath, setParentPath] = useState(folderParentDefault);
  const destPath = `${parentPath.replace(/[\\/]+$/, "")}${pathSep}${folderName}`;
  const hasChanges = destPath !== folder;

  // Resync the form from server metadata, but never over unsaved edits to the
  // *same* show: a background refetch (an artwork upload landing, another
  // window saving) would otherwise wipe what the user is mid-way through
  // typing, and their pending autosave carries the form's version anyway. A
  // different folder always resyncs, or one show's dirty edits would leak
  // into another's form and get saved there.
  const syncedFolderRef = useRef(folder);
  useEffect(() => {
    const sameShow = syncedFolderRef.current === folder;
    syncedFolderRef.current = folder;

    // Artwork resyncs on its own, ahead of the dirty guard. That guard reads
    // "differs from meta" as "the user is mid-edit", which is false for this
    // one field: upload and remove write it server-side, and on a feed-backed
    // show the next refresh replaces a cleared value with the feed's own art.
    // Without this the panel keeps showing no cover until it remounts, and the
    // form stays permanently dirty, so the next edit to any other field
    // autosaves the stale value back over the restored artwork.
    //
    // Typing in the URL box is the only artwork edit the guard has to protect,
    // so that is what is tracked, rather than inferring intent from a diff.
    if (artworkTouchedRef.current) {
      if (artworkUrlRef.current === meta.artwork_url) artworkTouchedRef.current = false;
    } else if (artworkUrlRef.current !== meta.artwork_url) {
      setArtworkUrl(meta.artwork_url);
    }

    if (sameShow && isDirtyRef.current) return;
    setName(meta.name);
    setLanguage(meta.language);
    setRssUrl(meta.rss_url);
    setYoutubeUrl(meta.youtube_url ?? "");
    setArtworkUrl(meta.artwork_url);
    setBroadcastPattern(meta.broadcast_number_pattern ?? "");
    setPipeModelSize(meta.pipeline?.model_size ?? "");
    setPipeDiarize(meta.pipeline?.diarize ?? null);
    setPipeNumSpeakers(meta.pipeline?.num_speakers ?? "");
    setPipeLlmMode(meta.pipeline?.llm_mode ?? "");
    setPipeLlmProviderProfile(meta.pipeline?.llm_provider_profile ?? "");
    setPipeLlmKeyName(meta.pipeline?.llm_key_name ?? "");
    setPipeLlmModels(meta.pipeline?.llm_models_by_mode ?? {});
    setPipeLlmBatchMinutes(
      meta.pipeline?.llm_batch_minutes != null ? String(meta.pipeline.llm_batch_minutes) : "",
    );
    setPipeContext(meta.pipeline?.context ?? "");
    setPipeTargetLang(meta.pipeline?.target_lang ?? "");
    setPipeRagModel(meta.pipeline?.rag_model ?? "");
    setPipeRagChunker(meta.pipeline?.rag_chunker ?? "");
  }, [meta, folder]);

  const isDirty =
    name !== meta.name ||
    language !== meta.language ||
    rssUrl !== meta.rss_url ||
    youtubeUrl !== (meta.youtube_url ?? "") ||
    artworkUrl !== meta.artwork_url ||
    broadcastPattern !== (meta.broadcast_number_pattern ?? "") ||
    pipeModelSize !== (meta.pipeline?.model_size ?? "") ||
    pipeDiarize !== (meta.pipeline?.diarize ?? null) ||
    pipeNumSpeakers !== (meta.pipeline?.num_speakers ?? "") ||
    pipeLlmMode !== (meta.pipeline?.llm_mode ?? "") ||
    pipeLlmProviderProfile !== (meta.pipeline?.llm_provider_profile ?? "") ||
    pipeLlmKeyName !== (meta.pipeline?.llm_key_name ?? "") ||
    !sameModels(pipeLlmModels, meta.pipeline?.llm_models_by_mode ?? {}) ||
    pipeLlmBatchMinutes !==
      (meta.pipeline?.llm_batch_minutes != null ? String(meta.pipeline.llm_batch_minutes) : "") ||
    pipeContext !== (meta.pipeline?.context ?? "") ||
    pipeTargetLang !== (meta.pipeline?.target_lang ?? "") ||
    pipeRagModel !== (meta.pipeline?.rag_model ?? "") ||
    pipeRagChunker !== (meta.pipeline?.rag_chunker ?? "");

  const saveMutation = useMutation({
    mutationFn: () =>
      updateShowMeta(folder, {
        name,
        language,
        rss_url: rssUrl,
        youtube_url: youtubeUrl,
        speakers: meta.speakers,
        artwork_url: artworkUrl,
        broadcast_number_pattern: broadcastPattern,
        pipeline: {
          model_size: pipeModelSize,
          diarize: pipeDiarize,
          num_speakers: pipeNumSpeakers,
          llm_mode: pipeLlmMode,
          llm_provider_profile: pipeLlmProviderProfile,
          llm_key_name: pipeLlmKeyName,
          llm_models_by_mode: pipeLlmModels,
          llm_batch_minutes: (() => {
            const trimmed = pipeLlmBatchMinutes.trim();
            if (trimmed === "") return null;
            const n = Number(trimmed);
            return Number.isFinite(n) && n > 0 ? n : null;
          })(),
          context: pipeContext,
          target_lang: pipeTargetLang,
          rag_model: pipeRagModel,
          rag_chunker: pipeRagChunker,
        },
      }),
    // Cache-level: settings save is debounced and the panel can unmount (tab
    // switch, navigation) before it resolves.
    meta: {
      invalidates: [
        queryKeys.showMeta(folder),
        queryKeys.shows(),
        // Showname is the cache key for search/index/bot-access namespaces, so
        // a rename strands every entry under the old name. Those have to be
        // *removed*: invalidating refetches them, and the backend no longer
        // knows that name, so each one 404s (visible as
        // "Unknown show '<old name>'" right after a rename). The namespace
        // invalidations then refill them under the new name.
        (qc) => {
          if (name === meta.name) return;
          removeQueriesForShowName(qc, meta.name);
          qc.invalidateQueries({ queryKey: ["search"] });
          qc.invalidateQueries({ queryKey: ["index"] });
          qc.invalidateQueries({ queryKey: ["bot-access"] });
        },
      ],
    },
  });

  const saveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const isDirtyRef = useRef(isDirty);

  isDirtyRef.current = isDirty;
  artworkUrlRef.current = artworkUrl;
  const autoSave = useCallback(() => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      if (isDirtyRef.current) saveMutation.mutate();
    }, 1500);
  }, [saveMutation]);

  useEffect(() => {
    if (isDirty) autoSave();
    return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
  }, [name, language, rssUrl, youtubeUrl, artworkUrl, broadcastPattern, pipeModelSize, pipeDiarize, pipeNumSpeakers, pipeLlmMode, pipeLlmProviderProfile, pipeLlmKeyName, pipeLlmModels, pipeLlmBatchMinutes, pipeContext, pipeTargetLang, pipeRagModel, pipeRagChunker]); // eslint-disable-line react-hooks/exhaustive-deps

  const moveMutation = useMutation({
    mutationFn: ({ newPath, moveFiles: mf }: { newPath: string; moveFiles: boolean }) =>
      moveShow(folder, newPath, mf),
    onSuccess: (data) => {
      removeQueriesUnderPath(queryClient, folder);
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
      navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(data.new_path) } });
    },
  });

  const deleteFilesRef = useRef(false);
  const deleteMutation = useMutation({
    mutationFn: (deleteFiles: boolean) => deleteShow(folder, deleteFiles),
    onSuccess: () => {
      // Re-adding a show at the same folder path would otherwise hit stale
      // per-folder caches (episodes, versions, roster, ...) before refetch.
      removeQueriesUnderPath(queryClient, folder);
      // And the same for everything keyed by the show *name* rather than its
      // path (search, index, bot-access). Left behind, those entries answer a
      // later namespace invalidation by refetching a show the backend no
      // longer has, which 404s exactly like a rename did.
      removeQueriesForShowName(queryClient, meta.name);
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
      navigate({ to: "/" });
    },
  });

  // ── Broadcast-number pattern live preview (tested against latest episode) ──
  const [debouncedPattern, setDebouncedPattern] = useState(broadcastPattern);
  useEffect(() => {
    const t = setTimeout(() => setDebouncedPattern(broadcastPattern), 400);
    return () => clearTimeout(t);
  }, [broadcastPattern]);
  const { data: broadcastPreview } = useQuery({
    queryKey: queryKeys.broadcastPreview(folder, debouncedPattern),
    queryFn: () => previewBroadcastNumber(folder, debouncedPattern),
    enabled: debouncedPattern.trim().length > 0,
    staleTime: 60_000,
  });

  const handleDelete = () => {
    deleteFilesRef.current = false;
    confirmDialog.open({
      title: "Remove this show?",
      description: `This will unregister "${meta.name}" from PodCodex. Its search index and bot password are kept, so adding the folder back restores the show as it was.`,
      content: (
        <label className="flex items-start gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            defaultChecked={false}
            onChange={(e) => { deleteFilesRef.current = e.target.checked; }}
            className="accent-destructive mt-0.5"
          />
          <span>
            Also delete local files on disk
            <span className="block text-xs text-muted-foreground">
              The show's search index and bot password are deleted too.
            </span>
          </span>
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
        <SettingRow label="Artwork" help="Cover image: paste a URL or upload a file.">
          <div className="flex items-center gap-2">
            <input
              value={isLocalArtwork ? "" : artworkUrl}
              onChange={(e) => {
                artworkTouchedRef.current = true;
                setArtworkUrl(e.target.value);
              }}
              placeholder={isLocalArtwork ? "Uploaded image" : "https://..."}
              className={`input ${inputWidth.medium}`}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => artworkFileRef.current?.click()}
              disabled={artworkUploadMutation.isPending}
              title="Upload a cover image"
            >
              <Upload className="w-3 h-3" />
              {artworkUploadMutation.isPending ? "Uploading…" : "Upload"}
            </Button>
            <input
              ref={artworkFileRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.gif"
              onChange={handleArtworkFile}
              className="hidden"
            />
            {artworkUrl && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    confirmDialog.open({
                      title: artworkCopy.title,
                      description: artworkCopy.description,
                      confirmLabel: artworkCopy.action,
                      variant: "destructive",
                      onConfirm: () => artworkRemoveMutation.mutate(),
                    })
                  }
                  disabled={artworkRemoveMutation.isPending}
                  title={artworkCopy.hint}
                >
                  <Trash2 className="w-3 h-3" />
                  {artworkCopy.action}
                </Button>
                <img
                  src={isLocalArtwork ? showArtworkEndpoint(folder) : artworkUrl}
                  alt="Artwork preview"
                  className="w-7 h-7 rounded object-cover shrink-0"
                  onError={(e) => (e.currentTarget.style.display = "none")}
                />
              </>
            )}
          </div>
          {artworkUploadMutation.isError && (
            <p className="text-destructive text-xs mt-1">{errorMessage(artworkUploadMutation.error)}</p>
          )}
          {artworkRemoveMutation.isError && (
            <p className="text-destructive text-xs mt-1">{errorMessage(artworkRemoveMutation.error)}</p>
          )}
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

      {/* ── Transcription ── */}
      <SettingSection
        title="Transcription"
        description={'How episodes of this show are turned into text. Each setting overrides the matching app default (Settings → Pipeline); leave it on "App default" to inherit.'}
      >
        <SettingRow label="Transcription model" help="Bigger models are more accurate but slower.">
          <select
            value={pipeModelSize}
            onChange={(e) => setPipeModelSize(e.target.value)}
            className={selectClass}
          >
            <option value="">App default</option>
            {Object.entries(whisperModels).map(([key, label]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>
        </SettingRow>
        <SettingRow label="Identify speakers" help="Detect who is talking and label each line with a speaker.">
          <select
            value={pipeDiarize === null ? "" : pipeDiarize ? "yes" : "no"}
            onChange={(e) =>
              setPipeDiarize(e.target.value === "" ? null : e.target.value === "yes")
            }
            className={selectClass}
          >
            <option value="">App default</option>
            <option value="yes">On</option>
            <option value="no">Off</option>
          </select>
        </SettingRow>
        {pipeDiarize === true && (
          <SettingRow label="Speaker count" help="How many people speak in the show. Leave blank to detect automatically.">
            <input
              type="number"
              min={1}
              value={pipeNumSpeakers}
              onChange={(e) => setPipeNumSpeakers(e.target.value)}
              placeholder="Auto"
              className={`input ${inputWidth.numeric}`}
            />
          </SettingRow>
        )}
      </SettingSection>

      {/* ── AI correction & translation ── */}
      <SettingSection
        title="AI correction & translation"
        description="The AI that cleans up raw transcripts and translates them."
      >
        <SettingRow
          label="Where the AI runs"
          help="Ollama runs on your own computer. Cloud API uses a paid online provider. Manual lets you copy the prompts and run them yourself."
        >
          <select
            value={pipeLlmMode}
            onChange={(e) => setPipeLlmMode(e.target.value)}
            className={selectClass}
          >
            <option value="">App default</option>
            <option value="api">Cloud API</option>
            <option value="ollama">Ollama (local)</option>
            <option value="manual">Manual (copy-paste prompts)</option>
          </select>
        </SettingRow>
        {pipeLlmMode === "api" && (
          <>
            <SettingRow label="AI provider" help="Which provider profile to use. Manage profiles in Settings → Credentials.">
              <select
                value={pipeLlmProviderProfile}
                onChange={(e) => setPipeLlmProviderProfile(e.target.value)}
                className={selectClass}
              >
                <option value="">App default</option>
                {apiProfiles.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}{p.builtin ? "" : " (custom)"}
                  </option>
                ))}
              </select>
            </SettingRow>
            <SettingRow label="AI API key" help="Which saved API key to use. Add keys in Settings → Credentials.">
              <select
                value={pipeLlmKeyName}
                onChange={(e) => setPipeLlmKeyName(e.target.value)}
                className={selectClass}
              >
                <option value="">App default</option>
                {keys.map((k) => (
                  <option key={k.name} value={k.name}>{k.name}</option>
                ))}
              </select>
            </SettingRow>
          </>
        )}
        {(pipeLlmMode === "api" || pipeLlmMode === "ollama") && (
          <SettingRow label="AI model" help="Specific model name for this mode. Leave blank to inherit the app default for the same mode.">
            <input
              value={pipeLlmModels[pipeLlmMode] ?? ""}
              onChange={(e) =>
                setPipeLlmModels((prev) => ({ ...prev, [pipeLlmMode]: e.target.value }))
              }
              placeholder="App default"
              className={`input ${inputWidth.short}`}
            />
          </SettingRow>
        )}
        <SettingRow
          label="Minutes per batch"
          help="Max audio duration per LLM request. Smaller batches stay within model context windows; larger batches are fewer requests but heavier prompts. Leave blank to inherit the app default."
        >
          <input
            type="number"
            min={1}
            step={1}
            value={pipeLlmBatchMinutes}
            onChange={(e) => setPipeLlmBatchMinutes(e.target.value)}
            placeholder="App default"
            className={`input ${inputWidth.numeric}`}
          />
        </SettingRow>
        <SettingRow
          label="About this show"
          help="A short description of the show, its hosts and recurring topics. The AI reads it to make better corrections."
          below={
            <textarea
              value={pipeContext}
              onChange={(e) => setPipeContext(e.target.value)}
              placeholder="Describe the show, hosts, recurring topics..."
              className={`input resize-y ${inputWidth.full} min-h-[4rem]`}
            />
          }
        >
          <span />
        </SettingRow>
        <SettingRow label="Translate into" help="Language episodes are translated into.">
          <input
            value={pipeTargetLang}
            onChange={(e) => setPipeTargetLang(e.target.value)}
            placeholder="App default"
            className={`input ${inputWidth.short}`}
          />
        </SettingRow>
      </SettingSection>

      {/* ── Search index ── */}
      <SettingSection
        title="Search index"
        description="The embeddings that make this show searchable, used by AI search and the MCP server."
      >
        <SettingRow
          label="Broadcast number"
          help={String.raw`Regex with one capture group, applied to episode titles to pull out an airing number for search filters. Example: \((\d+)\) captures 252 from "(252) John Powell". Leave blank to skip. Reindex episodes to apply a change.`}
          below={
            // Only show results matching the current input: during the
            // debounce window the fetched data describes the previous pattern.
            broadcastPattern.trim() && broadcastPattern === debouncedPattern ? (
              <div className="text-xs text-muted-foreground">
                {broadcastPreview?.error ? (
                  <span className="text-destructive">{broadcastPreview.error}</span>
                ) : broadcastPreview?.title ? (
                  <>
                    Latest: <span className="text-foreground">{broadcastPreview.title}</span>{" "}
                    {broadcastPreview.number != null ? (
                      <>
                        →{" "}
                        <span className="text-foreground font-medium">
                          {broadcastPreview.number}
                        </span>
                      </>
                    ) : (
                      <span className="text-warning">no match</span>
                    )}
                  </>
                ) : null}
              </div>
            ) : undefined
          }
        >
          <input
            value={broadcastPattern}
            onChange={(e) => setBroadcastPattern(e.target.value)}
            placeholder={String.raw`\((\d+)\)`}
            spellCheck={false}
            className={`input ${inputWidth.medium} font-mono`}
          />
        </SettingRow>
        <SettingRow
          label="Search: embedding model"
          help="Model used to make episodes searchable. AI search queries the model set here."
        >
          <select
            value={pipeRagModel}
            onChange={(e) => setPipeRagModel(e.target.value)}
            className={selectClass}
          >
            <option value="">App default (BGE-M3)</option>
            {Object.entries(indexConfig?.models ?? {}).map(([key, m]) => (
              <option key={key} value={key}>{m.label}</option>
            ))}
          </select>
        </SettingRow>
        <SettingRow
          label="Search: chunking"
          help="How transcripts are split into searchable chunks. Pairs with the embedding model above."
        >
          <select
            value={pipeRagChunker}
            onChange={(e) => setPipeRagChunker(e.target.value)}
            className={selectClass}
          >
            <option value="">App default (semantic)</option>
            {Object.keys(indexConfig?.chunking_strategies ?? {}).map((key) => (
              <option key={key} value={key}>{key}</option>
            ))}
          </select>
        </SettingRow>
      </SettingSection>

      {/* ── Discord bot access ── */}
      <ShowAccessSection show={meta.name} />

      {/* ── Sharing ── */}
      <BundleExportSection folder={folder} showName={meta.name} />

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
