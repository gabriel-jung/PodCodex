import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getEpisodes, getShowMeta, openFolder } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { artworkUrl, deleteFile, saveExportFile } from "@/api/filesystem";
import { usePlatform } from "@/platform";
import { uploadTranscript, getSpeakerMap, deleteTranscribeVersion } from "@/api/transcribe";
import { getSegmentsPreview as getTranscribePreview } from "@/api/transcribe";
import { getCorrectSegmentsPreview as getCorrectPreview, deleteCorrectVersion } from "@/api/correct";
import { deleteTranslateVersion } from "@/api/translate";
import {
  deleteAnyVersion,
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
import { formatDuration, formatDate, formatTime, stripHtml, errorMessage, langLabel, versionDate, versionLabel, isEdited, splitPath, STEP_LABELS } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";
import {
  Play,
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
  LayoutGrid,
} from "lucide-react";
import {
  PIPELINE_STEPS,
  STEP_BY_KEY,
  PipelineStatus,
  type ActiveStep,
  type PipelineStepKey,
  type StepStatus,
} from "@/components/episode/PipelineSteps";

type SidebarItem = {
  key: ActiveStep;
  label: string;
  icon: typeof Mic;
  status: StepStatus;
};

function buildSidebarSections(episode: Episode) {
  const meta: SidebarItem[] = [
    { key: "overview", label: "Overview", icon: LayoutGrid, status: false as StepStatus },
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
  initialTab,
}: {
  folder?: string;
  stem?: string;
  audioFilePath?: string;
  initialTab?: string;
}) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const seekTo = useAudioStore((s) => s.seekTo);
  const setAudioMeta = useAudioStore((s) => s.setAudioMeta);
  const activeStep: ActiveStep = (initialTab as ActiveStep) || "overview";

  const setActiveStep = useCallback((step: ActiveStep) => {
    navigate({
      search: ((prev: Record<string, unknown>) => ({
        ...prev,
        tab: step === "overview" ? undefined : step,
      })) as never,
    });
  }, [navigate]);

  const isStandalone = !!audioFilePath;
  const downloadTaskId = useTaskStore((s) => s.downloadTaskId);
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
    refetchOnWindowFocus: downloadTaskId ? false : undefined,
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
    [episode?.audio_path, queryClient, setActiveStep],
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
          ...(episode.removed ? [{ value: <span title="No longer in the live feed, kept locally" className="inline-flex items-center gap-1 text-muted-foreground"><CloudOff className="w-3.5 h-3.5" /> removed</span> }] : []),
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
          <div className="flex-1 overflow-y-auto">
            <Suspense fallback={<PanelLoading />}>
              <StepContent
                step={activeStep}
                episode={episode}
                folder={folder}
                meta={meta}
                isYouTube={isYouTube}
                onDownloadAudio={() => episodeDownloadMutation.mutate({ guids: [episode.id], force: episode.downloaded })}
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

function StepContent({ step, episode, folder, meta, isYouTube, onDownloadAudio, onImportSubs, downloadDisabled, downloadError, onNavigateStep }: { step: ActiveStep; episode: Episode; folder?: string; meta?: ShowMeta; isYouTube: boolean; onDownloadAudio: () => void; onImportSubs: (lang: string) => void; downloadDisabled: boolean; downloadError?: string; onNavigateStep: (step: ActiveStep) => void }) {
  if (step === "search") return <SearchPanel scope="episode" />;
  const def = STEP_BY_KEY[step as PipelineStepKey];
  if (def) return def.component();
  return (
    <OverviewTab
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


// ── Episode helpers ──────────────────────────────────────────────────────

const EMPTY_LANGS: string[] = [];

interface VersionGroups {
  transcript: VersionEntry[];
  corrected: VersionEntry[];
  translations: Record<string, VersionEntry[]>;
  other: VersionEntry[];
}

function groupVersions(
  versions: VersionEntry[] | undefined,
  languages: string[],
): VersionGroups {
  const groups: VersionGroups = { transcript: [], corrected: [], translations: {}, other: [] };
  for (const lang of languages) groups.translations[lang] = [];
  if (!versions) return groups;
  for (const v of versions) {
    if (v.step === "transcript") groups.transcript.push(v);
    else if (v.step === "corrected") groups.corrected.push(v);
    else if (v.step && languages.includes(v.step)) {
      (groups.translations[v.step] ??= []).push(v);
    } else {
      groups.other.push(v);
    }
  }
  return groups;
}



function latestSummary(v: VersionEntry): string {
  return `${versionLabel(v)} · ${versionDate(v)}${isEdited(v) ? " · edited" : ""}`;
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

// ── Overview tab ────────────────────────────────────────────────────────

type StageColor = "transcribe" | "correct" | "translate" | "synth" | "index";

const STAGE_CARD_CLASSES: Record<StageColor, { text: string; borderL: string; bg: string; dot: string }> = {
  transcribe: { text: "text-stage-transcribe", borderL: "border-l-stage-transcribe/60", bg: "bg-stage-transcribe/15", dot: "bg-stage-transcribe" },
  correct:    { text: "text-stage-correct",    borderL: "border-l-stage-correct/60",    bg: "bg-stage-correct/15",    dot: "bg-stage-correct"    },
  translate:  { text: "text-stage-translate",  borderL: "border-l-stage-translate/60",  bg: "bg-stage-translate/15",  dot: "bg-stage-translate"  },
  synth:      { text: "text-stage-synth",      borderL: "border-l-stage-synth/60",      bg: "bg-stage-synth/15",      dot: "bg-stage-synth"      },
  index:      { text: "text-warning",          borderL: "border-l-warning/60",          bg: "bg-warning/15",          dot: "bg-warning"          },
};

function StageCard({
  stage, icon: Icon, label, status, summary, muted = false, onOpen,
}: {
  stage: StageColor;
  icon: typeof Mic;
  label: string;
  status: StepStatus;
  summary?: string;
  muted?: boolean;
  onOpen: () => void;
}) {
  const c = STAGE_CARD_CLASSES[stage];
  const isEmpty = !status;
  const statusText = status === "done" ? "ready" : status === "partial" ? "needs review" : "not started";
  const statusColor = status === "done" ? "text-success" : status === "partial" ? "text-info" : "text-muted-foreground/60";

  return (
    <button
      type="button"
      onClick={onOpen}
      className={`group/card text-left rounded-md border border-border bg-card hover:bg-accent/40 transition px-3 py-2.5 flex flex-col gap-1.5 min-w-0 ${
        isEmpty ? "border-dashed" : `border-l-2 ${c.borderL}`
      }`}
    >
      <div className="flex items-center gap-2 min-w-0">
        <span
          className={`w-5 h-5 rounded inline-flex items-center justify-center shrink-0 ${
            isEmpty ? "bg-secondary text-muted-foreground" : `${c.bg} ${c.text}`
          }`}
        >
          <Icon className="w-3 h-3" />
        </span>
        <span className={`text-sm font-medium truncate ${muted ? "text-muted-foreground" : "text-foreground"}`}>{label}</span>
        <span className={`ml-auto text-2xs shrink-0 ${statusColor}`}>{statusText}</span>
      </div>
      {summary && (
        <div className="text-2xs text-muted-foreground/70 truncate font-mono tabular-nums">{summary}</div>
      )}
    </button>
  );
}

const TRANSCRIBE_INTERMEDIATE_STEPS = new Set(["segments", "diarization", "diarized_segments", "speaker_map"]);

function stepDisplay(step: string | undefined): { stage: StageColor; label: string; editorStep: ActiveStep } {
  if (step === "transcript") return { stage: "transcribe", label: "Transcribe", editorStep: "transcribe" };
  if (step === "corrected") return { stage: "correct", label: "Correct", editorStep: "correct" };
  if (step && TRANSCRIBE_INTERMEDIATE_STEPS.has(step)) {
    return { stage: "transcribe", label: STEP_LABELS[step], editorStep: "transcribe" };
  }
  return { stage: "translate", label: `Translate · ${langLabel(step ?? "")}`, editorStep: "translate" };
}

function StageArrow() {
  return (
    <div className="hidden sm:flex items-center justify-center text-muted-foreground/40">
      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <path d="M5 12h14M13 5l7 7-7 7" />
      </svg>
    </div>
  );
}

function ActivityLog({ versions, onPreview }: {
  /** Already sorted desc by timestamp by the caller. */
  versions: VersionEntry[] | undefined;
  onPreview: (previewKey: string) => void;
}) {
  if (!versions || versions.length === 0) return null;
  const recent = versions.slice(0, 8);
  return (
    <section className="space-y-2">
      <h4 className="text-sm font-medium px-1">Activity</h4>
      <div className="rounded-lg border border-border bg-card overflow-hidden">
        {recent.map((v) => {
          const { stage, label } = stepDisplay(v.step);
          const c = STAGE_CARD_CLASSES[stage];
          const edited = isEdited(v);
          return (
            <button
              type="button"
              key={v.id}
              onClick={() => onPreview(v.step ?? "")}
              className="w-full text-left px-3 py-2 border-b border-border/40 last:border-b-0 hover:bg-accent/40 transition space-y-1"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className={`text-2xs px-1.5 py-0.5 rounded font-medium shrink-0 ${c.bg} ${c.text}`}>
                  {label}
                </span>
                <span className="text-xs text-foreground truncate flex-1" title={versionLabel(v)}>
                  {versionLabel(v)}
                </span>
              </div>
              <div className="flex items-center gap-2 text-2xs text-muted-foreground/70 font-mono tabular-nums">
                <span>{versionDate(v)}</span>
                <span aria-hidden="true">·</span>
                <span>{v.segment_count} seg</span>
                {edited && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span className="text-success">edited</span>
                  </>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function VersionsTable({ versions, heading, firstColLabel, countColLabel, onPreview, onDelete, showEdited = true }: {
  /** Already sorted desc by timestamp by the caller. */
  versions: VersionEntry[] | undefined;
  heading: string;
  firstColLabel: string;
  countColLabel: string;
  onPreview?: (previewKey: string) => void;
  onDelete: (step: string, id: string) => void;
  showEdited?: boolean;
}) {
  if (!versions || versions.length === 0) return null;
  const clickable = !!onPreview;

  return (
    <section className="space-y-2">
      <h4 className="text-sm font-medium px-1">{heading}</h4>
      <div className="rounded-lg border border-border bg-card overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="bg-background/40 border-b border-border">
            <tr className="text-muted-foreground">
              <th className="text-left px-3 py-2 font-medium">{firstColLabel}</th>
              <th className="text-left px-3 py-2 font-medium">Made with</th>
              <th className="text-left px-3 py-2 font-medium">Created</th>
              <th className="text-right px-3 py-2 font-medium">{countColLabel}</th>
              <th className="px-3 py-2 w-8"></th>
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => {
              const { stage, label } = stepDisplay(v.step);
              const c = STAGE_CARD_CLASSES[stage];
              const edited = showEdited && isEdited(v);
              return (
                <tr
                  key={v.id}
                  onClick={clickable ? () => onPreview!(v.step ?? "") : undefined}
                  className={`group/vrow border-b border-border/40 last:border-b-0 hover:bg-accent/30 transition${clickable ? " cursor-pointer" : ""}`}
                >
                  <td className="px-3 py-2">
                    <span className="inline-flex items-center gap-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${c.dot}`} />
                      <span className="text-foreground">{label}</span>
                      {edited && <span className="ml-1 text-2xs text-success">edited</span>}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground truncate max-w-[320px]" title={versionLabel(v)}>
                    {versionLabel(v)}
                  </td>
                  <td className="px-3 py-2 font-mono text-2xs text-muted-foreground/70 tabular-nums whitespace-nowrap">
                    {versionDate(v)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                    {v.segment_count}
                  </td>
                  <td className="px-3 py-2 text-right whitespace-nowrap">
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); onDelete(v.step ?? "", v.id); }}
                      className="text-muted-foreground/40 hover:text-destructive opacity-0 group-hover/vrow:opacity-100 transition"
                      title="Delete this version"
                    >
                      <Trash2 className="w-3 h-3 inline-block" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function OverviewSourceCard({
  episode, folder, meta, isYouTube, subtitleFiles,
  onDownloadAudio, onImportSubs, downloadDisabled, downloadError,
  onDeleteSubtitle, onPreviewFile,
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
  const platform = usePlatform();
  const audioPath = episode.audio_path;
  const ext = audioPath?.split(".").pop()?.toLowerCase() ?? "";
  const formatLabel = ext ? ext.toUpperCase() : "audio";
  const durationLabel = episode.duration > 0 ? formatDuration(episode.duration) : "";
  const audioMeta = [formatLabel, durationLabel].filter(Boolean).join(" · ");

  const showsDownloadButton = !episode.downloaded || isYouTube;

  return (
    <section className="space-y-2">
      <h4 className="text-sm font-medium px-1">Source</h4>
      <div className="rounded-lg border border-border bg-card">
        {audioPath ? (
          <SourceFileRow
            icon={FileAudio}
            label={splitPath(audioPath).basename || "audio"}
            sublabel={audioMeta || "audio"}
            action={
              <button
                type="button"
                onClick={() => saveExportFile(platform, {
                  audioPath,
                  format: "audio",
                  defaultName: `${episode.title}.${ext || "mp3"}`,
                })}
                className="text-2xs text-muted-foreground hover:text-foreground transition"
              >
                Export
              </button>
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

        {(showsDownloadButton || isYouTube) && (
          <div className="border-t border-border/40 px-3 py-2">
            <DownloadDropdown
              isYouTube={isYouTube}
              showLanguage={meta?.language || ""}
              onDownload={onDownloadAudio}
              onImportSubs={onImportSubs}
              subsLabel={episode.transcribed ? "Re-download subtitles" : "Download subtitles"}
              subsEnabled={true}
              audioLabel={episode.downloaded ? "Re-download audio" : "Download audio"}
              showAudio={true}
              audioEnabled={true}
              disabled={downloadDisabled}
              variant={episode.downloaded ? "outline" : "default"}
              align="left"
            />
          </div>
        )}
      </div>
      {downloadError && (
        <p className="text-destructive text-xs px-1">{downloadError}</p>
      )}
    </section>
  );
}

function ShowNotesCard({ description }: { description: string }) {
  const [expanded, setExpanded] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const ref = useRef<HTMLParagraphElement | null>(null);
  const text = stripHtml(description).trim();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setOverflows(el.scrollHeight > el.clientHeight + 1);
  }, [text]);

  if (!text) return null;
  return (
    <section className="space-y-2">
      <h4 className="text-sm font-medium px-1">Show notes</h4>
      <div className="rounded-lg border border-border bg-card px-3 py-2.5">
        <p
          ref={ref}
          className={`text-xs text-muted-foreground whitespace-pre-line select-text ${expanded ? "" : "line-clamp-4"}`}
        >
          {text}
        </p>
        {(overflows || expanded) && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-2xs text-muted-foreground/60 hover:text-foreground transition mt-1.5"
          >
            {expanded ? "Less" : "More"}
          </button>
        )}
      </div>
    </section>
  );
}

function OverviewTab({ episode, folder, meta, isYouTube, onDownloadAudio, onImportSubs, downloadDisabled, downloadError, onNavigateStep }: { episode: Episode; folder?: string; meta?: ShowMeta; isYouTube: boolean; onDownloadAudio: () => void; onImportSubs: (lang: string) => void; downloadDisabled: boolean; downloadError?: string; onNavigateStep: (step: ActiveStep) => void }) {
  const platform = usePlatform();
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

  const outputDir = episode.output_dir;
  const { data: allVersions } = useQuery({
    queryKey: queryKeys.allVersions(audioPath ?? outputDir),
    queryFn: () => getAllVersions(audioPath, outputDir),
    enabled: (!!audioPath || !!outputDir) && hasTranscript,
  });

  const showName = meta?.name ?? "";
  const { data: indexEntries } = useQuery({
    queryKey: queryKeys.episodeCollections(audioPath ?? outputDir, showName),
    queryFn: () => getEpisodeCollections(audioPath, showName, outputDir),
    enabled: (!!audioPath || !!outputDir) && !!showName && !!episode.indexed,
  });

  const invalidateAll = useCallback(() => {
    const key = audioPath ?? outputDir;
    queryClient.invalidateQueries({ queryKey: queryKeys.allVersions(key) });
    queryClient.invalidateQueries({ queryKey: queryKeys.episodeCollections(key, showName) });
    queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
    queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments("transcribe", audioPath) });
    queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments("correct", audioPath) });
  }, [audioPath, outputDir, showName, queryClient]);

  const translations = episode.translations ?? EMPTY_LANGS;

  const deleteVersionMutation = useMutation({
    mutationFn: async ({ step, id }: { step: string; id: string }) => {
      const od = outputDir ?? undefined;
      if (step === "transcript") return deleteTranscribeVersion(audioPath, id, od);
      if (step === "corrected") return deleteCorrectVersion(audioPath, id, od);
      if (translations.includes(step)) return deleteTranslateVersion(audioPath, step, id, od);
      return deleteAnyVersion(audioPath, id, od);
    },
    onSuccess: invalidateAll,
  });

  const deleteCollectionMutation = useMutation({
    mutationFn: (collection: string) =>
      deleteEpisodeCollection(audioPath, showName, collection, outputDir),
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

  const subtitleFiles = useMemo(
    () => (episode.files ?? []).filter((f) => /\.(vtt|srt)$/i.test(f)),
    [episode.files],
  );

  const transcribeStatus = STEP_BY_KEY.transcribe.status(episode);
  const correctStatus = STEP_BY_KEY.correct.status(episode);
  const indexStatus = STEP_BY_KEY.index.status(episode);
  const translateStatus = STEP_BY_KEY.translate.status(episode);
  const synthStatus = STEP_BY_KEY.synthesize.status(episode);

  const transcriptVersions = useMemo(
    () => [
      ...versionGroups.transcript,
      ...versionGroups.corrected,
      ...Object.values(versionGroups.translations).flat(),
    ].sort((a, b) => b.timestamp.localeCompare(a.timestamp)),
    [versionGroups],
  );

  const indexSummary = useMemo(() => {
    if (!episode.indexed) return undefined;
    if (!indexEntries || indexEntries.length === 0) return "indexed";
    const total = indexEntries.reduce((acc, e) => acc + (e.chunk_count ?? 0), 0);
    const models = [...new Set(indexEntries.map((e) => e.model))].slice(0, 2).join(", ");
    return total ? `${total} chunks · ${models}` : models;
  }, [episode.indexed, indexEntries]);

  const openPreview = useCallback((previewKey: string) => {
    if (audioPath || outputDir) {
      setPreviewSource(previewKey);
      return;
    }
    onNavigateStep(stepDisplay(previewKey).editorStep);
  }, [audioPath, outputDir, onNavigateStep]);

  return (
    <div className="p-6 space-y-5 max-w-6xl">
      {/* Folder + Export ZIP */}
      {(folder || episode.audio_path || outputDir) && (
        <nav className="flex items-baseline gap-2 text-2xs text-muted-foreground/70 min-w-0">
          {folder && (
            <button
              type="button"
              onClick={() => openFolder(folder)}
              className="flex items-baseline gap-1.5 min-w-0 truncate hover:text-foreground transition"
              title={folder}
            >
              <FolderOpen className="w-3 h-3 shrink-0 self-center" />
              <span className="font-mono truncate">{folder}</span>
            </button>
          )}
          {(episode.audio_path || outputDir) && (
            <button
              type="button"
              onClick={() => saveExportFile(platform, {
                audioPath: episode.audio_path ?? undefined,
                outputDir: outputDir ?? undefined,
                format: "zip",
                defaultName: `${episode.stem || "episode"}.zip`,
              })}
              className="ml-auto flex items-center gap-1 shrink-0 hover:text-foreground transition"
            >
              <Download className="w-3 h-3" />
              <span>Export ZIP</span>
            </button>
          )}
        </nav>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr_auto_1fr] gap-2 items-stretch">
        <StageCard
          stage="transcribe"
          icon={Mic}
          label="Transcribe"
          status={transcribeStatus}
          summary={versionGroups.transcript[0] && latestSummary(versionGroups.transcript[0])}
          onOpen={() => hasTranscript ? openPreview("transcript") : onNavigateStep("transcribe")}
        />
        <StageArrow />
        <StageCard
          stage="correct"
          icon={Sparkles}
          label="Correct"
          status={correctStatus}
          summary={versionGroups.corrected[0] && latestSummary(versionGroups.corrected[0])}
          onOpen={() => episode.corrected ? openPreview("corrected") : onNavigateStep("correct")}
        />
        <StageArrow />
        <StageCard
          stage="index"
          icon={Database}
          label="Search index"
          status={indexStatus}
          summary={indexSummary}
          onOpen={() => {
            if (indexEntries && indexEntries.length === 1) {
              const e = indexEntries[0];
              setInspectTarget({ model: e.model, chunking: e.chunker });
            } else {
              onNavigateStep("index");
            }
          }}
        />
      </div>

      {/* Optional outputs (translation, synthesized audio) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        <StageCard
          stage="translate"
          icon={Languages}
          label="Translate"
          status={translateStatus}
          summary={
            translations.length > 0
              ? translations.map(langLabel).join(", ")
              : undefined
          }
          muted={!translateStatus}
          onOpen={() => {
            const onlyLang = translations.length === 1 ? translations[0] : null;
            if (onlyLang && (versionGroups.translations[onlyLang] ?? []).length > 0) {
              openPreview(onlyLang);
            } else {
              onNavigateStep("translate");
            }
          }}
        />
        <StageCard
          stage="synth"
          icon={AudioLines}
          label="Synthesized audio"
          status={synthStatus}
          summary={synthStatus ? "audio ready" : undefined}
          muted={!synthStatus}
          onOpen={() => onNavigateStep("synthesize")}
        />
      </div>

      {/* 2-col: preview LEFT, source/activity/notes RIGHT */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-5">
        <div className="space-y-5 min-w-0">
          {hasTranscript && previewSegments && previewSegments.length > 0 ? (() => {
            const previewLabel = previewStep === "correct" ? "corrected" : "transcript";
            const previewStage: StageColor = previewStep === "correct" ? "correct" : "transcribe";
            const previewVersion = previewStep === "correct" ? versionGroups.corrected[0] : versionGroups.transcript[0];
            const c = STAGE_CARD_CLASSES[previewStage];
            const subline = [
              episode.segment_count != null ? `${episode.segment_count} segments` : null,
              speakers.length > 0 ? speakers.join(", ") : null,
            ].filter(Boolean).join(" · ");
            return (
              <button
                onClick={() => openPreview(episode.corrected ? "corrected" : "transcript")}
                className="w-full text-left rounded-lg border border-border bg-card px-4 py-3.5 space-y-3 hover:bg-accent/40 transition group"
              >
                <div className="space-y-1">
                  <div className="flex items-baseline justify-between gap-3">
                    <div className="flex items-baseline gap-2 min-w-0">
                      <h4 className="text-sm font-medium">Transcript preview</h4>
                      <span className={`text-2xs px-1.5 py-0.5 rounded font-medium shrink-0 ${c.bg} ${c.text}`}>
                        {previewLabel}
                      </span>
                      {previewVersion && (
                        <span className="text-2xs text-muted-foreground truncate">
                          {latestSummary(previewVersion)}
                        </span>
                      )}
                    </div>
                    <span className="text-2xs text-primary opacity-0 group-hover:opacity-100 transition shrink-0">
                      Open &rarr;
                    </span>
                  </div>
                  {subline && <div className="text-2xs text-muted-foreground">{subline}</div>}
                </div>
                <div className="space-y-1.5 text-sm">
                  {previewSegments.map((seg) => (
                    <p key={`${seg.start}-${seg.speaker ?? ""}`} className="text-muted-foreground line-clamp-2">
                      <span className="font-mono tabular-nums text-2xs text-muted-foreground/50 mr-2">{formatTime(seg.start ?? 0, false)}</span>
                      {seg.speaker && <span className="font-medium" style={{ color: speakerColor(seg.speaker) }}>{seg.speaker}: </span>}
                      {seg.text}
                    </p>
                  ))}
                </div>
              </button>
            );
          })() : hasTranscript ? null : (
            <div className="rounded-lg border border-dashed border-border px-4 py-6 text-center text-sm text-muted-foreground italic">
              Transcribe the episode to see a preview here.
            </div>
          )}

        </div>

        <aside className="space-y-5">
          <OverviewSourceCard
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
            onPreviewFile={hasTranscript ? () => openPreview(episode.corrected ? "corrected" : "transcript") : undefined}
          />

          {episode.description && <ShowNotesCard description={episode.description} />}
        </aside>
      </div>

      <ActivityLog
        versions={transcriptVersions}
        onPreview={openPreview}
      />

      <VersionsTable
        versions={transcriptVersions}
        heading="All transcript versions"
        firstColLabel="Step"
        countColLabel="Segments"
        onPreview={openPreview}
        onDelete={(step, id) => deleteVersionMutation.mutate({ step, id })}
      />

      {episode.indexed && indexEntries && indexEntries.length > 0 && (
        <section className="space-y-2">
          <h4 className="text-sm font-medium px-1">All index versions</h4>
          <div className="rounded-lg border border-border bg-card overflow-hidden">
            {indexEntries.map((e) => (
              <IndexRow
                key={e.collection}
                entry={e}
                onInspect={() => setInspectTarget({ model: e.model, chunking: e.chunker })}
                onDelete={() => deleteCollectionMutation.mutate(e.collection)}
              />
            ))}
          </div>
        </section>
      )}

      <VersionsTable
        versions={versionGroups.other}
        heading="All other files"
        firstColLabel="File"
        countColLabel="Entries"
        showEdited={false}
        onDelete={(step, id) => deleteVersionMutation.mutate({ step, id })}
      />

      {(audioPath || outputDir) && previewSource && (
        <Suspense fallback={null}>
          <SegmentContextDialog
            open={true}
            onOpenChange={(open) => { if (!open) setPreviewSource(null); }}
            audioPath={audioPath ?? undefined}
            outputDir={outputDir ?? undefined}
            source={previewSource}
            episodeTitle={episode.title}
            onSeek={(t) => audioPath && seekTo(audioPath, t)}
            onOpenEditor={() => {
              setPreviewSource(null);
              onNavigateStep(stepDisplay(previewSource).editorStep);
            }}
          />
        </Suspense>
      )}

      {(audioPath || outputDir) && inspectTarget && (
        <Suspense fallback={null}>
          <IndexInspectorModal
            open={true}
            onClose={() => setInspectTarget(null)}
            audioPath={audioPath ?? undefined}
            outputDir={outputDir ?? undefined}
            show={meta?.name ?? ""}
            model={inspectTarget.model}
            chunking={inspectTarget.chunking}
          />
        </Suspense>
      )}
    </div>
  );
}
