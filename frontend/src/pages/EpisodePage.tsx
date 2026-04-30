import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getEpisodes, getShowMeta, exportZipUrl, openFolder } from "@/api/client";
import { audioFileUrl } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { artworkUrl, deleteFile } from "@/api/filesystem";
import { uploadTranscript, getSpeakerMap, deleteTranscribeVersion } from "@/api/transcribe";
import { getSegmentsPreview as getTranscribePreview } from "@/api/transcribe";
import { getCorrectSegmentsPreview as getCorrectPreview, deleteCorrectVersion } from "@/api/correct";
import { deleteTranslateVersion } from "@/api/translate";
import {
  deleteEpisodeCollection,
  getAllVersions,
  getEpisodeCollections,
  type EpisodeCollection,
} from "@/api/search";
import { useShowActions } from "@/hooks/useShowActions";
import { usePipelineDefaults } from "@/hooks/usePipelineConfig";
import DownloadDropdown from "@/components/common/DownloadDropdown";
import InlineConfirm from "@/components/common/InlineConfirm";
import { useDropZone } from "@/hooks/useDropZone";
import DropOverlay from "@/components/common/DropOverlay";
import EditorialHeader from "@/components/layout/EditorialHeader";
import AppSidebar from "@/components/layout/AppSidebar";
import type { Episode, ShowMeta, VersionEntry } from "@/api/types";
import { standaloneEpisode } from "@/lib/standaloneEpisode";
import { useAudioStore, useEpisodeStore, useTaskStore } from "@/stores";
import { Button } from "@/components/ui/button";
import PanelLoading from "@/components/common/PanelLoading";

const SearchPanel = lazy(() => import("@/components/search/SearchPanel"));
const SegmentContextDialog = lazy(() => import("@/components/search/SegmentContextDialog"));
const IndexInspectorModal = lazy(() => import("@/components/index/IndexInspectorModal"));
import { formatDuration, formatDate, stripHtml, errorMessage, langLabel, versionDate, versionLabel, isEdited } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";
import {
  Play,
  Info,
  Download,
  Search,
  FolderOpen,
  Mic,
  Sparkles,
  Languages,
  AudioLines,
  Database,
  FileAudio,
  FileText,
  Trash2,
  Captions,
  CloudOff,
} from "lucide-react";
import {
  PIPELINE_STEPS,
  STEP_BY_KEY,
  PipelineStatus,
  type ActiveStep,
  type StepStatus,
} from "@/components/episode/PipelineSteps";
import OutputGroup from "@/components/episode/OutputGroup";
import VersionRow from "@/components/episode/VersionRow";

type SidebarItem = {
  key: ActiveStep;
  label: string;
  icon: typeof Info;
  status: StepStatus;
};

function buildSidebarSections(episode: Episode) {
  const meta: SidebarItem[] = [
    { key: "info", label: "Info", icon: Info, status: false as StepStatus },
    { key: "search", label: "Search", icon: Search, status: false as StepStatus },
  ];
  const core: SidebarItem[] = [];
  const bonus: SidebarItem[] = [];
  for (const s of PIPELINE_STEPS) {
    const item: SidebarItem = { key: s.key as ActiveStep, label: s.label, icon: s.icon, status: s.status(episode) };
    if (s.section === "core") core.push(item);
    else bonus.push(item);
  }
  return [{ items: meta }, { items: core }, { items: bonus }];
}

export default function EpisodePage({
  folder,
  stem,
  audioFilePath,
}: {
  folder?: string;
  stem?: string;
  audioFilePath?: string;
}) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const seekTo = useAudioStore((s) => s.seekTo);
  const setAudioMeta = useAudioStore((s) => s.setAudioMeta);
  const [activeStep, setActiveStep] = useState<ActiveStep>("info");
  const [descExpanded, setDescExpanded] = useState(false);

  const isStandalone = !!audioFilePath;
  const { downloadTaskId } = useTaskStore();
  const pipelineDefaults = usePipelineDefaults();

  const { data: meta } = useQuery({
    queryKey: queryKeys.showMeta(folder ?? ""),
    queryFn: () => getShowMeta(folder!),
    enabled: !!folder,
  });

  const { data: episodes } = useQuery({
    queryKey: queryKeys.episodes(folder ?? "", pipelineDefaults),
    queryFn: () => getEpisodes(folder!, pipelineDefaults),
    placeholderData: keepPreviousData,
    enabled: !!folder,
    refetchInterval: downloadTaskId ? 5000 : false,
  });

  const { downloadMutation: episodeDownloadMutation, importSubsMutation, isYouTube } = useShowActions(folder ?? "", meta, { withSubs: false });

  // TaskBar invalidates ["episodes", folder] on completion, but in
  // practice the panel sometimes still shows stale audio_path. Force a
  // second refetch when downloadTaskId clears so the blocker swaps to
  // the live transcribe form without a manual reload.
  const prevDownloadTaskId = useRef<string | null>(null);
  useEffect(() => {
    if (prevDownloadTaskId.current && !downloadTaskId && folder) {
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesForFolder(folder) });
    }
    prevDownloadTaskId.current = downloadTaskId;
  }, [downloadTaskId, folder, queryClient]);

  const episode: Episode | undefined = audioFilePath
    ? standaloneEpisode(audioFilePath)
    : episodes?.find((e) => e.stem === stem || e.id === stem);


  const artwork = episode?.artwork_url || (meta?.artwork_url && folder ? artworkUrl(folder) : "");

  useEffect(() => {
    if (!episode?.audio_path) return;
    setAudioMeta(episode.audio_path, {
      title: episode.title,
      artwork: artwork || undefined,
      showName: meta?.name,
      folder,
      stem: episode.stem || undefined,
    });
  }, [episode?.audio_path, episode?.title, artwork, meta?.name, folder, episode?.stem, setAudioMeta]);

  const setEpisode = useEpisodeStore((s) => s.setEpisode);
  const setShowMeta = useEpisodeStore((s) => s.setShowMeta);
  useEffect(() => {
    setEpisode(episode ?? null, folder);
  }, [episode, folder, setEpisode]);

  useEffect(() => {
    setShowMeta(meta ?? null);
  }, [meta, setShowMeta]);

  const handleFileDrop = useCallback(
    async (files: File[]) => {
      const audioPath = episode?.audio_path;
      if (!audioPath || files.length === 0) return;
      try {
        await uploadTranscript(audioPath, files[0]);
        // Invalidate step-scoped segment queries for every editor step (the
        // previous `["segments"]` prefix never matched `[editorKey, "segments", ...]`).
        queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments("transcribe", audioPath) });
        queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments("correct", audioPath) });
        queryClient.invalidateQueries({ queryKey: queryKeys.transcribeSegments(audioPath) });
        queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
        setActiveStep("transcribe");
      } catch (e) {
        console.error("Transcript drop import failed:", e);
      }
    },
    [episode?.audio_path, queryClient],
  );

  const { isDragging } = useDropZone({
    accept: [".json", ".srt", ".vtt"],
    onDrop: handleFileDrop,
    disabled: !episode?.audio_path,
  });

  const sidebarSections = useMemo(
    () => (episode ? buildSidebarSections(episode) : []),
    [episode],
  );

  if (!isStandalone && !episodes) {
    return <div className="p-6 text-muted-foreground">Loading...</div>;
  }

  if (!episode) {
    return (
      <div className="p-6 text-muted-foreground">
        Episode not found.{" "}
        <Button onClick={() => window.history.back()} variant="link" size="sm">
          Go back
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {isDragging && <DropOverlay message="Drop a transcript file here (JSON, SRT, VTT)" />}
      <EditorialHeader
        title={episode.title}
        breadcrumbs={
          isStandalone
            ? [{ label: "File", onClick: () => navigate({ to: "/" }) }, { label: episode.title }]
            : [
                { label: "Shows", onClick: () => navigate({ to: "/" }) },
                ...(folder
                  ? [{ label: meta?.name || folder, onClick: () => navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(folder) } }) }]
                  : []),
                { label: episode.title },
              ]
        }
        artworkUrl={artwork || undefined}
        fallbackIcon={Mic}
        onArtworkClick={episode.audio_path ? () => seekTo(episode.audio_path!, 0) : undefined}
        artworkOverlay={episode.audio_path ? <Play className="w-8 h-8 text-white fill-white" /> : undefined}
        stats={[
          ...(episode.episode_number != null ? [{ value: `#${episode.episode_number}` }] : []),
          ...(episode.pub_date ? [{ value: formatDate(episode.pub_date) }] : []),
          ...(episode.duration > 0 ? [{ value: formatDuration(episode.duration) }] : []),
          ...(episode.has_subtitles ? [{ value: <span title="Subtitles cached" className="inline-flex items-center gap-1"><Captions className="w-3.5 h-3.5" /> subs</span> }] : []),
          ...(episode.removed ? [{ value: <span title="No longer in the live feed — kept locally" className="inline-flex items-center gap-1 text-muted-foreground"><CloudOff className="w-3.5 h-3.5" /> removed</span> }] : []),
        ]}
        statusSlot={<PipelineStatus episode={episode} />}
        actions={
          <div className="flex items-center gap-1.5">
            {!episode.downloaded && (
              <Button
                onClick={() => episodeDownloadMutation.mutate({ guids: [episode.id] })}
                variant="outline"
                size="icon"
                className="h-8 w-8"
                title="Download audio"
                disabled={episodeDownloadMutation.isPending || !!downloadTaskId}
              >
                <Download className="w-3.5 h-3.5" />
              </Button>
            )}
            {episode.audio_path && (
              <Button
                onClick={() => seekTo(episode.audio_path!, 0)}
                variant="outline"
                size="icon"
                className="h-8 w-8"
                title="Play"
              >
                <Play className="w-3.5 h-3.5" />
              </Button>
            )}
          </div>
        }
      />

      <div className="flex-1 flex overflow-hidden">
        <AppSidebar
          parentLabel={!isStandalone ? (meta?.name ?? "Show") : undefined}
          onParent={!isStandalone && folder ? () => navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(folder) } }) : undefined}
          pageSections={sidebarSections}
          activeItem={activeStep}
          onItemClick={(key) => setActiveStep(key as ActiveStep)}
        />

        <div className="flex-1 flex flex-col overflow-hidden">
          {episode.description && (
            <div className="px-6 py-2 border-b border-border">
              <p
                className={`text-2xs text-muted-foreground whitespace-pre-line select-text ${descExpanded ? "" : "line-clamp-2"}`}
              >
                {stripHtml(episode.description)}
              </p>
              <button
                onClick={() => setDescExpanded(!descExpanded)}
                className="text-2xs text-muted-foreground/50 hover:text-foreground transition"
              >
                {descExpanded ? "Less" : "More"}
              </button>
            </div>
          )}
          <div className="flex-1 overflow-y-auto">
            <Suspense fallback={<PanelLoading />}>
              <StepContent
                step={activeStep}
                episode={episode}
                folder={folder}
                meta={meta}
                isYouTube={isYouTube}
                onDownloadAudio={() => episodeDownloadMutation.mutate({ guids: [episode.id] })}
                onImportSubs={(lang) => importSubsMutation.mutate({ ids: [episode?.id ?? ""], lang })}
                downloadDisabled={episodeDownloadMutation.isPending || importSubsMutation.isPending || !!downloadTaskId}
                downloadError={episodeDownloadMutation.isError ? errorMessage(episodeDownloadMutation.error) : importSubsMutation.isError ? errorMessage(importSubsMutation.error) : undefined}
                onNavigateStep={setActiveStep}
              />
            </Suspense>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Compute a next-action hint from episode state. */
function getNextAction(episode: Episode): string | undefined {
  if (!episode.downloaded && !episode.transcribed) return "Download audio to get started";
  if (episode.downloaded && !episode.transcribed) return "Ready to transcribe";
  if (episode.transcribed && !episode.corrected) return "Transcript ready for review";
  if (episode.corrected && !episode.indexed) return "Ready to index or translate";
  return undefined;
}

function StepContent({ step, episode, folder, meta, isYouTube, onDownloadAudio, onImportSubs, downloadDisabled, downloadError, onNavigateStep }: { step: ActiveStep; episode: Episode; folder?: string; meta?: ShowMeta; isYouTube: boolean; onDownloadAudio: () => void; onImportSubs: (lang: string) => void; downloadDisabled: boolean; downloadError?: string; onNavigateStep: (step: ActiveStep) => void }) {
  if (step === "search") return <SearchPanel scope="episode" />;
  if (step === "info") {
    return (
      <InfoTab
        episode={episode}
        folder={folder}
        meta={meta}
        isYouTube={isYouTube}
        onDownloadAudio={onDownloadAudio}
        onImportSubs={onImportSubs}
        downloadDisabled={downloadDisabled}
        downloadError={downloadError}
        onNavigateStep={onNavigateStep}
      />
    );
  }
  return STEP_BY_KEY[step].component();
}

function InfoTab({ episode, folder, meta, isYouTube, onDownloadAudio, onImportSubs, downloadDisabled, downloadError, onNavigateStep }: { episode: Episode; folder?: string; meta?: ShowMeta; isYouTube: boolean; onDownloadAudio: () => void; onImportSubs: (lang: string) => void; downloadDisabled: boolean; downloadError?: string; onNavigateStep: (step: ActiveStep) => void }) {
  const audioPath = episode.audio_path;
  const hasTranscript = !!episode.transcribed;
  const seekTo = useAudioStore((s) => s.seekTo);
  const [previewSource, setPreviewSource] = useState<string | null>(null);
  const [inspectTarget, setInspectTarget] = useState<{ model: string; chunking: string } | null>(null);
  const queryClient = useQueryClient();

  const { data: speakerMap } = useQuery({
    queryKey: queryKeys.speakerMap(audioPath),
    queryFn: () => getSpeakerMap(audioPath!),
    enabled: !!audioPath && hasTranscript,
  });
  const previewStep = episode.corrected ? "correct" : "transcribe";
  const PREVIEW_LIMIT = 5;
  const { data: previewSegments } = useQuery({
    queryKey: [...queryKeys.stepSegments(previewStep, audioPath), "preview"],
    queryFn: () => (previewStep === "correct" ? getCorrectPreview(audioPath!, PREVIEW_LIMIT) : getTranscribePreview(audioPath!, PREVIEW_LIMIT)),
    enabled: !!audioPath && hasTranscript,
  });

  const { data: allVersions } = useQuery({
    queryKey: queryKeys.allVersions(audioPath),
    queryFn: () => getAllVersions(audioPath!),
    enabled: !!audioPath && hasTranscript,
  });

  const showName = meta?.name ?? "";
  const { data: indexEntries } = useQuery({
    queryKey: queryKeys.episodeCollections(audioPath, showName),
    queryFn: () => getEpisodeCollections(audioPath!, showName),
    enabled: !!audioPath && !!showName && !!episode.indexed,
  });

  const invalidateAll = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: queryKeys.allVersions(audioPath) });
    queryClient.invalidateQueries({ queryKey: queryKeys.episodeCollections(audioPath, showName) });
    queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
    queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments("transcribe", audioPath) });
    queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments("correct", audioPath) });
  }, [audioPath, showName, queryClient]);

  const deleteVersionMutation = useMutation({
    mutationFn: async ({ step, id }: { step: string; id: string }) => {
      if (!audioPath) return;
      if (step === "transcript") return deleteTranscribeVersion(audioPath, id);
      if (step === "corrected") return deleteCorrectVersion(audioPath, id);
      return deleteTranslateVersion(audioPath, step, id);
    },
    onSuccess: invalidateAll,
  });

  const deleteCollectionMutation = useMutation({
    mutationFn: (collection: string) =>
      deleteEpisodeCollection(audioPath!, showName, collection),
    onSuccess: invalidateAll,
  });

  const deleteFileMutation = useMutation({
    mutationFn: (path: string) => deleteFile(path),
    onSuccess: invalidateAll,
  });

  const speakers = useMemo(() => {
    if (!speakerMap) return [];
    return [...new Set(Object.values(speakerMap))].filter(Boolean);
  }, [speakerMap]);

  const versionGroups = useMemo(
    () => groupVersions(allVersions, episode.translations ?? EMPTY_LANGS),
    [allVersions, episode.translations],
  );
  const translations = episode.translations ?? EMPTY_LANGS;

  const nextAction = getNextAction(episode);
  const subtitleFiles = useMemo(
    () => (episode.files ?? []).filter((f) => !f.includes("/") && /\.(vtt|srt|info\.json)$/.test(f)),
    [episode.files],
  );

  return (
    <div className="p-6 space-y-5 max-w-2xl">
      {nextAction && (
        <p className="text-sm text-muted-foreground italic">{nextAction}</p>
      )}

      <SourcesSection
        episode={episode}
        folder={folder}
        meta={meta}
        isYouTube={isYouTube}
        subtitleFiles={subtitleFiles}
        onDownloadAudio={onDownloadAudio}
        onImportSubs={onImportSubs}
        downloadDisabled={downloadDisabled}
        downloadError={downloadError}
        onDeleteSubtitle={(filename) => {
          if (folder) deleteFileMutation.mutate(`${folder}/${filename}`);
        }}
        onPreviewFile={hasTranscript ? () => setPreviewSource(episode.corrected ? "corrected" : "transcript") : undefined}
      />

      <div className="space-y-2">
        <h4 className="text-sm font-medium px-1">Outputs</h4>

        <StepGroup
          title="Transcripts"
          icon={Mic}
          versions={versionGroups.transcript}
          onPreview={() => setPreviewSource("transcript")}
          onOpenEditor={() => onNavigateStep("transcribe")}
          onDelete={(id) => deleteVersionMutation.mutate({ step: "transcript", id })}
          emptyHint="Not transcribed yet."
          emptyOnClick={() => onNavigateStep("transcribe")}
        />

        {episode.transcribed && (
          <StepGroup
            title="Corrected"
            icon={Sparkles}
            versions={versionGroups.corrected}
            onPreview={() => setPreviewSource("corrected")}
            onOpenEditor={() => onNavigateStep("correct")}
            onDelete={(id) => deleteVersionMutation.mutate({ step: "corrected", id })}
            emptyHint="Not corrected yet."
            emptyOnClick={() => onNavigateStep("correct")}
          />
        )}

        {episode.transcribed && translations.length > 0 &&
          translations.map((lang) => (
            <StepGroup
              key={lang}
              title={`Translation · ${langLabel(lang)}`}
              icon={Languages}
              versions={versionGroups.translations[lang] ?? []}
              onPreview={() => setPreviewSource(lang)}
              onOpenEditor={() => onNavigateStep("translate")}
              onDelete={(id) => deleteVersionMutation.mutate({ step: lang, id })}
              emptyHint="No versions."
              emptyOnClick={() => onNavigateStep("translate")}
            />
          ))}

        {episode.synthesized && (
          <OutputGroup
            title="Synthesized audio"
            icon={AudioLines}
            count={1}
            summary="present"
            defaultOpen={false}
          >
            <div className="px-4 py-2 text-xs text-muted-foreground">
              Open the Synthesize step to inspect or rebuild.
            </div>
          </OutputGroup>
        )}

        {episode.transcribed && (
          <IndexGroup
            indexed={!!episode.indexed}
            entries={indexEntries ?? []}
            onOpen={() => onNavigateStep("index")}
            onInspect={(model, chunking) => setInspectTarget({ model, chunking })}
            onDelete={(collection) => deleteCollectionMutation.mutate(collection)}
          />
        )}
      </div>

      {hasTranscript && previewSegments && previewSegments.length > 0 && (
        <button
          onClick={() => onNavigateStep(previewStep)}
          className="w-full text-left rounded-lg bg-muted/50 px-4 py-3 space-y-2.5 hover:bg-muted/70 transition group"
        >
          <div className="flex items-baseline justify-between">
            <div className="flex items-baseline gap-2">
              <h4 className="text-sm font-medium">Preview</h4>
              <span className="text-2xs text-muted-foreground">
                {[
                  episode.segment_count != null ? `${episode.segment_count} segments` : null,
                  speakers.length > 0 ? speakers.join(", ") : null,
                ].filter(Boolean).join(" · ")}
              </span>
            </div>
            <span className="text-2xs text-muted-foreground opacity-0 group-hover:opacity-100 transition shrink-0">
              Open {previewStep === "correct" ? "corrected" : "transcript"} &rarr;
            </span>
          </div>
          <div className="space-y-1 text-sm">
            {previewSegments.map((seg, i) => (
              <p key={i} className="text-muted-foreground line-clamp-1">
                {seg.speaker && <span className="font-medium" style={{ color: speakerColor(seg.speaker) }}>{seg.speaker}: </span>}
                {seg.text}
              </p>
            ))}
          </div>
        </button>
      )}

      <div className="flex items-center gap-4 pt-2 border-t border-border text-xs text-muted-foreground">
        {folder && (
          <button
            onClick={() => openFolder(folder)}
            className="flex items-center gap-1.5 hover:text-foreground transition"
            title={folder}
          >
            <FolderOpen className="w-3.5 h-3.5 shrink-0" />
            <span className="break-all font-mono">{folder}</span>
          </button>
        )}
        {episode.audio_path && (
          <a href={exportZipUrl(episode.audio_path)} download className="flex items-center gap-1.5 hover:text-foreground transition ml-auto shrink-0">
            <Download className="w-3.5 h-3.5" />
            <span>Export ZIP</span>
          </a>
        )}
      </div>

      {audioPath && previewSource && (
        <Suspense fallback={null}>
        <SegmentContextDialog
          open={true}
          onOpenChange={(open) => { if (!open) setPreviewSource(null); }}
          audioPath={audioPath}
          source={previewSource}
          episodeTitle={episode.title}
          onSeek={(t) => seekTo(audioPath, t)}
          onOpenEditor={() => {
            setPreviewSource(null);
            if (previewSource === "corrected") onNavigateStep("correct");
            else if (previewSource === "transcript") onNavigateStep("transcribe");
            else onNavigateStep("translate");
          }}
        />
        </Suspense>
      )}

      {audioPath && inspectTarget && (
        <Suspense fallback={null}>
          <IndexInspectorModal
            open={true}
            onClose={() => setInspectTarget(null)}
            audioPath={audioPath}
            show={meta?.name ?? ""}
            model={inspectTarget.model}
            chunking={inspectTarget.chunking}
          />
        </Suspense>
      )}
    </div>
  );
}

// ── InfoTab helpers ──────────────────────────────────────────────────────

const EMPTY_LANGS: string[] = [];

interface VersionGroups {
  transcript: VersionEntry[];
  corrected: VersionEntry[];
  translations: Record<string, VersionEntry[]>;
}

function groupVersions(
  versions: VersionEntry[] | undefined,
  languages: string[],
): VersionGroups {
  const groups: VersionGroups = { transcript: [], corrected: [], translations: {} };
  for (const lang of languages) groups.translations[lang] = [];
  if (!versions) return groups;
  for (const v of versions) {
    if (v.step === "transcript") groups.transcript.push(v);
    else if (v.step === "corrected") groups.corrected.push(v);
    else if (v.step && languages.includes(v.step)) {
      (groups.translations[v.step] ??= []).push(v);
    }
  }
  return groups;
}


function StepGroup({
  title,
  icon,
  versions,
  onPreview,
  onOpenEditor,
  onDelete,
  emptyHint,
  emptyOnClick,
}: {
  title: string;
  icon: typeof Mic;
  versions: VersionEntry[];
  onPreview: () => void;
  onOpenEditor: () => void;
  onDelete: (id: string) => void;
  emptyHint?: string;
  emptyOnClick?: () => void;
}) {
  if (versions.length === 0) {
    if (!emptyHint) return null;
    const Icon = icon;
    return (
      <div className="rounded-lg border border-border/50 px-4 py-2.5 flex items-center gap-3 text-sm text-muted-foreground italic">
        <Icon className="w-3.5 h-3.5" />
        <span className="flex-1">{emptyHint}</span>
        {emptyOnClick && (
          <button
            onClick={emptyOnClick}
            className="text-xs not-italic text-foreground hover:underline"
          >
            Open &rarr;
          </button>
        )}
      </div>
    );
  }

  const latest = versions[0];
  return (
    <OutputGroup
      title={title}
      icon={icon}
      count={versions.length}
      summary={latestSummary(latest)}
      defaultOpen={versions.length <= 3}
    >
      {versions.map((v, i) => (
        <VersionRow
          key={v.id}
          version={v}
          isLatest={i === 0}
          onOpen={onPreview}
          onDelete={() => onDelete(v.id)}
        />
      ))}
      <div className="px-4 py-1.5 border-t border-border/40">
        <button
          onClick={onOpenEditor}
          className="text-xs text-muted-foreground hover:text-foreground transition"
        >
          Open editor &rarr;
        </button>
      </div>
    </OutputGroup>
  );
}

function latestSummary(v: VersionEntry): string {
  return `${versionLabel(v)} · ${versionDate(v)}${isEdited(v) ? " · edited" : ""}`;
}

function SourcesSection({
  episode,
  folder,
  meta,
  isYouTube,
  subtitleFiles,
  onDownloadAudio,
  onImportSubs,
  downloadDisabled,
  downloadError,
  onDeleteSubtitle,
  onPreviewFile,
}: {
  episode: Episode;
  folder?: string;
  meta?: ShowMeta;
  isYouTube: boolean;
  subtitleFiles: string[];
  onDownloadAudio: () => void;
  onImportSubs: (lang: string) => void;
  downloadDisabled: boolean;
  downloadError?: string;
  onDeleteSubtitle: (filename: string) => void;
  onPreviewFile?: () => void;
}) {
  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium px-1">Sources</h4>

      <div className="rounded-lg border border-border/50 divide-y divide-border/40">
        {episode.audio_path ? (
          <SourceFileRow
            icon={FileAudio}
            label={episode.audio_path.split("/").pop() ?? "audio"}
            sublabel="audio"
            action={
              <a
                href={audioFileUrl(episode.audio_path)}
                download={`${episode.title}.${episode.audio_path.split(".").pop()}`}
                className="text-2xs text-muted-foreground hover:text-foreground transition"
              >
                Save
              </a>
            }
          />
        ) : (
          <div className="px-4 py-3 text-sm text-muted-foreground italic">
            No audio yet.
          </div>
        )}

        {subtitleFiles.map((f) => (
          <SourceFileRow
            key={f}
            icon={FileText}
            label={f}
            sublabel="subtitles"
            onClick={/\.(vtt|srt)$/.test(f) ? onPreviewFile : undefined}
            onDelete={folder ? () => onDeleteSubtitle(f) : undefined}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <DownloadDropdown
          isYouTube={isYouTube}
          showLanguage={meta?.language || ""}
          onDownload={onDownloadAudio}
          onImportSubs={onImportSubs}
          subsLabel={episode.transcribed ? "Re-import subtitles" : "Import subtitles"}
          subsEnabled={true}
          audioLabel="Download audio"
          showAudio={!episode.downloaded}
          audioEnabled={!episode.downloaded}
          disabled={downloadDisabled}
          variant={episode.downloaded ? "outline" : "default"}
          align="right"
        />
      </div>
      {downloadError && (
        <p className="text-destructive text-xs">{downloadError}</p>
      )}
    </div>
  );
}

function SourceFileRow({
  icon: Icon,
  label,
  sublabel,
  action,
  onClick,
  onDelete,
}: {
  icon: typeof FileAudio;
  label: string;
  sublabel?: string;
  action?: React.ReactNode;
  onClick?: () => void;
  onDelete?: () => void;
}) {
  const [confirming, setConfirming] = useState(false);

  if (confirming && onDelete) {
    return (
      <div className="px-4 py-2">
        <InlineConfirm
          message={`Delete ${label}?`}
          onConfirm={() => {
            setConfirming(false);
            onDelete();
          }}
          onCancel={() => setConfirming(false)}
        />
      </div>
    );
  }

  const LabelTag = onClick ? "button" : "span";
  return (
    <div className="px-4 py-2 flex items-center gap-3 group/row hover:bg-accent/30 transition">
      <Icon className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
      <LabelTag
        onClick={onClick}
        className={`flex-1 min-w-0 text-xs font-mono truncate text-left ${onClick ? "hover:underline cursor-pointer" : ""}`}
      >
        {label}
      </LabelTag>
      {sublabel && (
        <span className="text-2xs text-muted-foreground shrink-0">{sublabel}</span>
      )}
      {action}
      {onDelete && (
        <button
          onClick={() => setConfirming(true)}
          className="shrink-0 text-muted-foreground/40 hover:text-destructive p-0.5 opacity-0 group-hover/row:opacity-100 transition"
          title={`Delete ${label}`}
        >
          <Trash2 className="w-3 h-3" />
        </button>
      )}
    </div>
  );
}

function IndexGroup({
  indexed,
  entries,
  onOpen,
  onInspect,
  onDelete,
}: {
  indexed: boolean;
  entries: EpisodeCollection[];
  onOpen: () => void;
  onInspect: (model: string, chunking: string) => void;
  onDelete: (collection: string) => void;
}) {
  if (!indexed || entries.length === 0) {
    return (
      <div className="rounded-lg border border-border/50 px-4 py-2.5 flex items-center gap-3 text-sm text-muted-foreground italic">
        <Database className="w-3.5 h-3.5" />
        <span className="flex-1">Not indexed.</span>
        <button
          onClick={onOpen}
          className="text-xs not-italic text-foreground hover:underline"
        >
          Open &rarr;
        </button>
      </div>
    );
  }

  return (
    <OutputGroup
      title="Search index"
      icon={Database}
      count={entries.length}
      summary={entries
        .map((e) => `${e.model}·${e.chunker}`)
        .slice(0, 2)
        .join(", ")}
      defaultOpen={entries.length <= 3}
    >
      {entries.map((e) => (
        <IndexRow
          key={e.collection}
          entry={e}
          onInspect={() => onInspect(e.model, e.chunker)}
          onDelete={() => onDelete(e.collection)}
        />
      ))}
    </OutputGroup>
  );
}

function IndexRow({
  entry,
  onInspect,
  onDelete,
}: {
  entry: EpisodeCollection;
  onInspect: () => void;
  onDelete: () => void;
}) {
  const [confirming, setConfirming] = useState(false);

  if (confirming) {
    return (
      <div className="px-4 py-2 border-l-2 border-transparent">
        <InlineConfirm
          message={`Remove from ${entry.collection}?`}
          onConfirm={() => {
            setConfirming(false);
            onDelete();
          }}
          onCancel={() => setConfirming(false)}
        />
      </div>
    );
  }

  return (
    <div className="px-4 py-2 flex items-center gap-2 group/row hover:bg-accent/40 transition border-l-2 border-transparent">
      <span className="shrink-0 w-1.5 h-1.5 rounded-full bg-info" />
      <button
        type="button"
        onClick={onInspect}
        className="flex-1 truncate text-xs text-left hover:underline cursor-pointer"
        title="Inspect chunks and vectors"
      >
        <span className="text-foreground">
          {entry.model} · {entry.chunker}
        </span>
        {entry.source && (
          <span className="text-muted-foreground"> · from {entry.source}</span>
        )}
      </button>
      <span className="shrink-0 font-mono text-2xs text-muted-foreground/60 tabular-nums">
        {entry.chunk_count} chunks
      </span>
      <button
        onClick={() => setConfirming(true)}
        className="shrink-0 text-muted-foreground/40 hover:text-destructive p-0.5 opacity-0 group-hover/row:opacity-100 transition"
        title="Remove from this collection"
      >
        <Trash2 className="w-3 h-3" />
      </button>
    </div>
  );
}
