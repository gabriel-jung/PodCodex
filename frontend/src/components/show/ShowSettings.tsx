import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import type { ShowMeta } from "@/api/types";
import { updateShowMeta, moveShow, deleteShow } from "@/api/client";
import { removeQueriesUnderPath } from "@/api/cacheInvalidation";
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
  // ── Per-show pipeline config (show.toml [pipeline]) ──
  // Empty string / null means "inherit the app default" (Settings, Pipeline).
  const [pipeModelSize, setPipeModelSize] = useState(meta.pipeline?.model_size ?? "");
  const [pipeDiarize, setPipeDiarize] = useState<boolean | null>(meta.pipeline?.diarize ?? null);
  const [pipeNumSpeakers, setPipeNumSpeakers] = useState(meta.pipeline?.num_speakers ?? "");
  const [pipeLlmMode, setPipeLlmMode] = useState(meta.pipeline?.llm_mode ?? "");
  const [pipeLlmProviderProfile, setPipeLlmProviderProfile] = useState(meta.pipeline?.llm_provider_profile ?? "");
  const [pipeLlmKeyName, setPipeLlmKeyName] = useState(meta.pipeline?.llm_key_name ?? "");
  const [pipeLlmModel, setPipeLlmModel] = useState(meta.pipeline?.llm_model ?? "");
  const [pipeContext, setPipeContext] = useState(meta.pipeline?.context ?? "");
  const [pipeTargetLang, setPipeTargetLang] = useState(meta.pipeline?.target_lang ?? "");
  const [pipeRagModel, setPipeRagModel] = useState(meta.pipeline?.rag_model ?? "");
  const [pipeRagChunker, setPipeRagChunker] = useState(meta.pipeline?.rag_chunker ?? "");

  const { data: indexConfig } = useIndexConfig();
  const { whisperModels } = useLLMProviders();
  const { profiles } = useProviderProfiles();
  const { keys } = useApiKeys();
  const apiProfiles = useMemo(() => profiles.filter((p) => p.type !== "ollama"), [profiles]);

  // ── Move folder ──
  const { parent: folderParentDefault, basename: folderBasename, sep: pathSep } = splitPath(folder);
  const moveFilesRef = useRef(true);
  const [folderName, setFolderName] = useState(folderBasename);
  const [parentPath, setParentPath] = useState(folderParentDefault);
  const destPath = `${parentPath.replace(/[\\/]+$/, "")}${pathSep}${folderName}`;
  const hasChanges = destPath !== folder;

  useEffect(() => {
    setName(meta.name);
    setLanguage(meta.language);
    setRssUrl(meta.rss_url);
    setYoutubeUrl(meta.youtube_url ?? "");
    setArtworkUrl(meta.artwork_url);
    setPipeModelSize(meta.pipeline?.model_size ?? "");
    setPipeDiarize(meta.pipeline?.diarize ?? null);
    setPipeNumSpeakers(meta.pipeline?.num_speakers ?? "");
    setPipeLlmMode(meta.pipeline?.llm_mode ?? "");
    setPipeLlmProviderProfile(meta.pipeline?.llm_provider_profile ?? "");
    setPipeLlmKeyName(meta.pipeline?.llm_key_name ?? "");
    setPipeLlmModel(meta.pipeline?.llm_model ?? "");
    setPipeContext(meta.pipeline?.context ?? "");
    setPipeTargetLang(meta.pipeline?.target_lang ?? "");
    setPipeRagModel(meta.pipeline?.rag_model ?? "");
    setPipeRagChunker(meta.pipeline?.rag_chunker ?? "");
  }, [meta]);

  const isDirty =
    name !== meta.name ||
    language !== meta.language ||
    rssUrl !== meta.rss_url ||
    youtubeUrl !== (meta.youtube_url ?? "") ||
    artworkUrl !== meta.artwork_url ||
    pipeModelSize !== (meta.pipeline?.model_size ?? "") ||
    pipeDiarize !== (meta.pipeline?.diarize ?? null) ||
    pipeNumSpeakers !== (meta.pipeline?.num_speakers ?? "") ||
    pipeLlmMode !== (meta.pipeline?.llm_mode ?? "") ||
    pipeLlmProviderProfile !== (meta.pipeline?.llm_provider_profile ?? "") ||
    pipeLlmKeyName !== (meta.pipeline?.llm_key_name ?? "") ||
    pipeLlmModel !== (meta.pipeline?.llm_model ?? "") ||
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
        pipeline: {
          model_size: pipeModelSize,
          diarize: pipeDiarize,
          num_speakers: pipeNumSpeakers,
          llm_mode: pipeLlmMode,
          llm_provider_profile: pipeLlmProviderProfile,
          llm_key_name: pipeLlmKeyName,
          llm_model: pipeLlmModel,
          context: pipeContext,
          target_lang: pipeTargetLang,
          rag_model: pipeRagModel,
          rag_chunker: pipeRagChunker,
        },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.showMeta(folder) });
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
      // Showname is the cache key for search/index/bot-access namespaces;
      // a rename strands those entries under the old name.
      if (name !== meta.name) {
        queryClient.invalidateQueries({ queryKey: ["search"] });
        queryClient.invalidateQueries({ queryKey: ["index"] });
        queryClient.invalidateQueries({ queryKey: ["bot-access"] });
      }
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
  }, [name, language, rssUrl, youtubeUrl, artworkUrl, pipeModelSize, pipeDiarize, pipeNumSpeakers, pipeLlmMode, pipeLlmProviderProfile, pipeLlmKeyName, pipeLlmModel, pipeContext, pipeTargetLang, pipeRagModel, pipeRagChunker]); // eslint-disable-line react-hooks/exhaustive-deps

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
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
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
        <SettingRow label="AI model" help="Specific model name. Leave blank to use the provider's default.">
          <input
            value={pipeLlmModel}
            onChange={(e) => setPipeLlmModel(e.target.value)}
            placeholder="App default"
            className={`input ${inputWidth.short}`}
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
