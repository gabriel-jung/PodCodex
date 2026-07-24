import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { Fragment, lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getEpisodes, getShowMeta, getEpisodeSpeakers, openFolder } from "@/api/client";
import { invalidateSpeakerViews } from "@/api/cacheInvalidation";
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
} from "@/api/search";
import { useShowActions } from "@/hooks/useShowActions";
import { isVerifiedVersion, VERIFIED_CAPTION } from "@/lib/verified";
import { usePipelineDefaults } from "@/hooks/usePipelineConfig";
import DownloadDropdown from "@/components/common/DownloadDropdown";
import InlineConfirm from "@/components/common/InlineConfirm";
import { useDropZone } from "@/hooks/useDropZone";
import DropOverlay from "@/components/common/DropOverlay";
import EditorialHeader from "@/components/layout/EditorialHeader";
import AppSidebar from "@/components/layout/AppSidebar";
import type { Episode, ShowMeta, VersionEntry } from "@/api/types";
import { standaloneEpisode } from "@/lib/standaloneEpisode";
import { useAudioStore, useEpisodeStore, useTaskStore, useSeedPipelineFromShow } from "@/stores";
import { Button } from "@/components/ui/button";
import PanelLoading from "@/components/common/PanelLoading";

const SearchPanel = lazy(() => import("@/components/search/SearchPanel"));
const SegmentContextDialog = lazy(() => import("@/components/search/SegmentContextDialog"));
const IndexInspectorModal = lazy(() => import("@/components/index/IndexInspectorModal"));
import IndexRow from "@/components/index/IndexRow";
import { STAGE_CLASSES, type StageKey } from "@/lib/stageClasses";
import { formatDuration, formatDate, formatTime, formatBytes, stripHtml, errorMessage, langLabel, versionDate, versionLabel, isEdited, shortVersionId, splitPath, STEP_LABELS } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";
import { byDefaultOrder } from "@/lib/episodeSort";
import {
  Play,
  Download,
  Search,
  FolderOpen,
  Mic,
  Users,
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
  ChevronRight,
  AlertTriangle,
  Star,
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
  const registerMeta = useAudioStore((s) => s.registerMeta);
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

  useSeedPipelineFromShow(folder, meta?.pipeline, !!meta);

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

  // Prev/next navigation walks the sibling list in the show's default order
  // (newest first). Note: a user-applied sort or filter on the show list does
  // not change the arrows; they always follow the default date order.
  const siblings = useMemo(
    () => (episodes ? [...episodes].sort(byDefaultOrder) : []),
    [episodes],
  );
  const goToEpisode = (ep: Episode) =>
    navigate({
      to: "/show/$folder/episode/$stem",
      params: {
        folder: encodeURIComponent(folder!),
        stem: encodeURIComponent(ep.stem || ep.id),
      },
    });
  const navIdx = isStandalone
    ? -1
    : siblings.findIndex((e) => e.stem === stem || e.id === stem);
  const headerNav =
    isStandalone || !folder || navIdx < 0 || siblings.length < 2
      ? undefined
      : {
          onPrev: navIdx > 0 ? () => goToEpisode(siblings[navIdx - 1]) : undefined,
          onNext:
            navIdx < siblings.length - 1
              ? () => goToEpisode(siblings[navIdx + 1])
              : undefined,
        };


  const artwork = episode?.artwork_url || (meta?.artwork_url && folder ? artworkUrl(folder) : "");

  // Register meta for this episode's audio WITHOUT loading it into the player.
  // Navigation stays separate from playback: the bar keeps playing whatever is
  // playing, and only reveals this episode once the user explicitly plays it.
  useEffect(() => {
    if (!episode?.audio_path) return;
    registerMeta(episode.audio_path, {
      title: episode.title,
      artwork: artwork || undefined,
      showName: meta?.name,
      folder,
      stem: episode.stem || undefined,
    });
  }, [episode?.audio_path, episode?.title, artwork, meta?.name, folder, episode?.stem, registerMeta]);

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
        nav={headerNav}
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

      <div className="flex-1 flex flex-col overflow-hidden">
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
  if (def) {
    // Force remount on episode change so per-episode UI state (selection,
    // expansion, in-flight task ids, etc.) cannot bleed across episodes.
    const Panel = def.component;
    return <Panel key={`${step}|${episode.id}`} />;
  }
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
  synthesize: VersionEntry[];
  other: VersionEntry[];
}

function groupVersions(
  versions: VersionEntry[] | undefined,
  languages: string[],
): VersionGroups {
  const groups: VersionGroups = { transcript: [], corrected: [], translations: {}, synthesize: [], other: [] };
  for (const lang of languages) groups.translations[lang] = [];
  if (!versions) return groups;
  for (const v of versions) {
    if (v.step === "transcript") groups.transcript.push(v);
    else if (v.step === "corrected") groups.corrected.push(v);
    else if (v.step === "synthesize") groups.synthesize.push(v);
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


// ── Overview tab ────────────────────────────────────────────────────────

type StageColor = StageKey;

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
  const c = STAGE_CLASSES[stage];
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

// A diarized run writes its two transcript versions back-to-back (seconds
// apart). Two transcripts further apart than this belong to separate runs —
// catches a json/subtitle import, which has no intermediates to split runs.
const RUN_GAP_MS = 2 * 60 * 1000;

function stepDisplay(step: string | undefined): { stage: StageColor; label: string; editorStep: ActiveStep } {
  if (step === "transcript") return { stage: "transcribe", label: "Transcribe", editorStep: "transcribe" };
  if (step === "corrected") return { stage: "correct", label: "Correct", editorStep: "correct" };
  if (step === "synthesize") return { stage: "synth", label: "Synthesize", editorStep: "synthesize" };
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
  onPreview: (previewKey: string, versionId?: string) => void;
}) {
  if (!versions || versions.length === 0) return null;
  const recent = versions.slice(0, 8);
  return (
    <section className="space-y-2">
      <h4 className="text-sm font-medium px-1">Activity</h4>
      <div className="rounded-lg border border-border bg-card overflow-hidden">
        {recent.map((v) => {
          const { stage, label } = stepDisplay(v.step);
          const c = STAGE_CLASSES[stage];
          const edited = isEdited(v);
          return (
            <button
              type="button"
              key={v.id}
              onClick={() => onPreview(v.step ?? "", v.id)}
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

function VersionsTable({ versions, heading, firstColLabel, countColLabel, onPreview, onDelete, showEdited = true, sizeColumn = false, childrenByVersionId, verifiedVersionId }: {
  /** Already sorted desc by timestamp by the caller. */
  versions: VersionEntry[] | undefined;
  heading: string;
  firstColLabel: string;
  countColLabel: string;
  onPreview?: (previewKey: string, versionId?: string) => void;
  onDelete: (step: string, id: string) => void;
  showEdited?: boolean;
  /** When set, the row whose version id matches gets a "verified" star. */
  verifiedVersionId?: string | null;
  /** Render the last column as formatted file size from params.file_size_bytes
   *  instead of segment_count. */
  sizeColumn?: boolean;
  /** Optional child versions keyed by parent version id. When a parent has
   *  children, its row gains a disclosure chevron; expanded children render
   *  indented underneath. */
  childrenByVersionId?: Map<string, VersionEntry[]>;
}) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  if (!versions || versions.length === 0) return null;
  const clickable = !!onPreview;

  const toggleExpanded = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const renderRow = (v: VersionEntry, isChild: boolean) => {
    const { stage, label } = stepDisplay(v.step);
    const c = STAGE_CLASSES[stage];
    const edited = showEdited && isEdited(v);
    const isVerified = !!verifiedVersionId && v.id === verifiedVersionId;
    const kids = childrenByVersionId?.get(v.id);
    const hasKids = !!kids && kids.length > 0;
    const open = expanded.has(v.id);
    return (
      <tr
        key={v.id}
        onClick={clickable ? () => onPreview!(v.step ?? "", v.id) : undefined}
        className={`group/vrow border-b border-border/40 last:border-b-0 hover:bg-accent/30 transition${clickable ? " cursor-pointer" : ""}${isChild ? " bg-background/30" : ""}`}
      >
        <td className="px-3 py-2">
          <span className={`flex items-center gap-1.5${isChild ? " pl-5" : ""}`}>
            <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${c.dot}`} />
            <span className="text-foreground flex-1 min-w-0 truncate">{label}</span>
            {isVerified && (
              <span className="ml-0.5 inline-flex items-center gap-0.5 text-2xs text-verified shrink-0" title={`Verified version: ${VERIFIED_CAPTION}`}>
                <Star className="w-2.5 h-2.5" fill="currentColor" />
                verified
              </span>
            )}
            {edited && <span className="ml-1 text-2xs text-success shrink-0">edited</span>}
            {hasKids && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); toggleExpanded(v.id); }}
                className="ml-0.5 p-1 -my-1 text-muted-foreground/60 hover:text-foreground transition shrink-0"
                title={open ? "Hide intermediates" : "Show intermediates"}
              >
                <ChevronRight className={`w-3.5 h-3.5 transition-transform ${open ? "rotate-90" : ""}`} />
              </button>
            )}
          </span>
        </td>
        <td className="px-3 py-2 text-muted-foreground truncate max-w-[320px]" title={versionLabel(v)}>
          {versionLabel(v)}
        </td>
        <td className="px-3 py-2 font-mono text-2xs text-muted-foreground/70 tabular-nums whitespace-nowrap">
          {versionDate(v)}
        </td>
        <td className="px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
          {sizeColumn
            ? formatBytes((v.params as { file_size_bytes?: number } | undefined)?.file_size_bytes)
            : v.segment_count}
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
  };

  return (
    <section className="space-y-2">
      <h4 className="text-sm font-medium px-1">{heading}</h4>
      <div className="rounded-lg border border-border bg-card overflow-x-auto">
        {/* table-fixed: column widths stay put when expanding a row exposes
            wider child labels (otherwise the longer names push every column). */}
        <table className="w-full text-xs table-fixed">
          <thead className="bg-background/40 border-b border-border">
            <tr className="text-muted-foreground">
              <th className="text-left px-3 py-2 font-medium w-52">{firstColLabel}</th>
              <th className="text-left px-3 py-2 font-medium">Made with</th>
              <th className="text-left px-3 py-2 font-medium w-28">Created</th>
              <th className="text-right px-3 py-2 font-medium w-20">{countColLabel}</th>
              <th className="px-3 py-2 w-8"></th>
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => {
              const kids = childrenByVersionId?.get(v.id);
              const open = expanded.has(v.id);
              return (
                <Fragment key={v.id}>
                  {renderRow(v, false)}
                  {open && kids?.map((child) => renderRow(child, true))}
                </Fragment>
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
  const [previewVersionId, setPreviewVersionId] = useState<string | null>(null);
  const [inspectTarget, setInspectTarget] = useState<{ model: string; chunking: string } | null>(null);
  const queryClient = useQueryClient();

  const { data: speakerMap } = useQuery({
    queryKey: queryKeys.speakerMap(audioPath),
    queryFn: () => getSpeakerMap(audioPath!),
    enabled: !!audioPath && hasTranscript,
  });
  // Speakers of the canonical transcript with per-speaker airtime share.
  const { data: episodeSpeakers } = useQuery({
    queryKey: queryKeys.episodeSpeakers(folder ?? "", episode.stem ?? ""),
    queryFn: () => getEpisodeSpeakers(folder!, episode.stem!),
    enabled: !!folder && !!episode.stem && hasTranscript,
  });
  const previewStep = episode.corrected ? "correct" : "transcribe";
  const PREVIEW_LIMIT = 5;
  const outputDir = episode.output_dir;
  const { data: previewSegments } = useQuery({
    queryKey: [...queryKeys.stepSegments(previewStep, audioPath ?? outputDir), "preview"],
    queryFn: () =>
      previewStep === "correct"
        ? getCorrectPreview(audioPath, PREVIEW_LIMIT, outputDir ?? undefined)
        : getTranscribePreview(audioPath, PREVIEW_LIMIT, outputDir ?? undefined),
    enabled: (!!audioPath || !!outputDir) && hasTranscript,
  });

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
    queryClient.invalidateQueries({ queryKey: queryKeys.speakerMap(audioPath) });
    queryClient.invalidateQueries({ queryKey: queryKeys.bestSourceSegments(audioPath) });
    invalidateSpeakerViews(queryClient);
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
    onMutate: async ({ id }) => {
      // Optimistic remove so the trash click feels instant; rollback below
      // restores the prior list if the server rejects the delete.
      const key = audioPath ?? outputDir;
      const qk = queryKeys.allVersions(key);
      await queryClient.cancelQueries({ queryKey: qk });
      const prev = queryClient.getQueryData<VersionEntry[]>(qk);
      if (prev) {
        queryClient.setQueryData<VersionEntry[]>(qk, prev.filter((v) => v.id !== id));
      }
      return { prev, qk };
    },
    onError: (err, _vars, ctx) => {
      if (ctx?.prev && ctx.qk) {
        queryClient.setQueryData(ctx.qk, ctx.prev);
      }
      console.error("Delete version failed:", err);
    },
    onSuccess: invalidateAll,
  });

  const deleteCollectionMutation = useMutation({
    mutationFn: (collection: string) =>
      deleteEpisodeCollection(audioPath, showName, collection, outputDir),
    onSuccess: () => {
      invalidateAll();
      queryClient.invalidateQueries({ queryKey: ["search"] });
      queryClient.invalidateQueries({ queryKey: ["index"] });
    },
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

  // Each intermediate parquet (segments / diarization / diarized_segments /
  // speaker_map) is a by-product of one transcribe run. A diarized batch run
  // emits TWO transcript versions (undiarized + diarized) from a single set
  // of intermediates, so a plain "closest later transcript" pairing files the
  // diarization intermediates under the undiarized transcript. Group by run
  // instead: raw whisper `segments` belong under every transcript of the run;
  // diarization-derived intermediates belong under the diarized transcript(s).
  // Orphans (no later transcript) stay in "All other files".
  const { childrenByTranscriptId, orphanIntermediates } = useMemo(() => {
    const stream = [
      ...versionGroups.transcript.map((v) => ({ v, inter: false })),
      ...versionGroups.other
        .filter((v) => v.step && TRANSCRIBE_INTERMEDIATE_STEPS.has(v.step))
        .map((v) => ({ v, inter: true })),
    ].sort((a, b) => a.v.timestamp.localeCompare(b.v.timestamp));

    const map = new Map<string, VersionEntry[]>();
    const orphans: VersionEntry[] = [];
    let pending: VersionEntry[] = [];
    let runTx: VersionEntry[] = [];

    const flush = () => {
      const diarizedTx = runTx.filter(
        (t) => (t.params as { diarize?: unknown } | undefined)?.diarize === true,
      );
      for (const inter of pending) {
        // Whisper `segments` → every transcript of the run; diarization data
        // → the diarized transcript(s), falling back to all when untagged.
        const targets =
          inter.step !== "segments" && diarizedTx.length > 0 ? diarizedTx : runTx;
        if (targets.length === 0) {
          orphans.push(inter);
          continue;
        }
        for (const t of targets) {
          const arr = map.get(t.id) ?? [];
          arr.push(inter);
          map.set(t.id, arr);
        }
      }
      pending = [];
      runTx = [];
    };

    for (const { v, inter } of stream) {
      if (inter) {
        if (runTx.length > 0) flush(); // an intermediate after a run begins the next
        pending.push(v);
      } else {
        // A transcript far in time from the current run's transcripts is a
        // separate run — a json/subtitle import emits no intermediates, so
        // without this it would absorb the previous run's intermediates.
        const prev = runTx[runTx.length - 1];
        if (prev && Date.parse(v.timestamp) - Date.parse(prev.timestamp) > RUN_GAP_MS) {
          flush();
        }
        runTx.push(v);
      }
    }
    flush();

    for (const arr of map.values()) {
      arr.sort((a, b) => b.timestamp.localeCompare(a.timestamp));
    }
    return { childrenByTranscriptId: map, orphanIntermediates: orphans };
  }, [versionGroups.transcript, versionGroups.other]);

  const otherFilesVersions = useMemo(
    () => [
      ...versionGroups.synthesize,
      ...versionGroups.other.filter(
        (v) => !v.step || !TRANSCRIBE_INTERMEDIATE_STEPS.has(v.step),
      ),
      ...orphanIntermediates,
    ].sort((a, b) => b.timestamp.localeCompare(a.timestamp)),
    [versionGroups.synthesize, versionGroups.other, orphanIntermediates],
  );

  // Activity feed shows all step events, including synth. The "All transcript
  // versions" table uses transcriptVersions (no synth).
  const recentActivityVersions = useMemo(
    () => [...transcriptVersions, ...versionGroups.synthesize].sort(
      (a, b) => b.timestamp.localeCompare(a.timestamp),
    ),
    [transcriptVersions, versionGroups.synthesize],
  );

  const indexSummary = useMemo(() => {
    if (!episode.indexed) return undefined;
    if (!indexEntries || indexEntries.length === 0) return "indexed";
    const total = indexEntries.reduce((acc, e) => acc + (e.chunk_count ?? 0), 0);
    const models = [...new Set(indexEntries.map((e) => e.model))].slice(0, 2).join(", ");
    return total ? `${total} chunks · ${models}` : models;
  }, [episode.indexed, indexEntries]);

  // versionId targets a specific version row; omit it to preview the step's
  // current (active) version, e.g. from the transcript-preview card.
  const openPreview = useCallback((previewKey: string, versionId?: string) => {
    if (audioPath || outputDir) {
      setPreviewSource(previewKey);
      setPreviewVersionId(versionId ?? null);
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

      {episodeSpeakers && episodeSpeakers.speakers.length > 0 && (
        <div
          className="flex items-center gap-x-1.5 gap-y-1 flex-wrap text-xs text-muted-foreground"
          title="Share of the episode duration spoken by each speaker (music and gaps are not counted)"
        >
          <Users className="w-3.5 h-3.5 shrink-0" />
          {episodeSpeakers.speakers.map((s, i) => (
            <span key={s.name} className="inline-flex items-center gap-1">
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ background: speakerColor(s.name) }}
              />
              <span className="text-foreground/80">{s.name}</span>
              <span className="tabular-nums">({Math.round(s.pct)}%)</span>
              {i < episodeSpeakers.speakers.length - 1 && (
                <span className="opacity-40">,</span>
              )}
            </span>
          ))}
        </div>
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
            const c = STAGE_CLASSES[previewStage];
            // When the previewed (latest) version is the episode's verified
            // source, the badge becomes the canonical "verified" marker instead
            // of the plain step name (replaces the old standalone verified pill).
            const previewVerified = isVerifiedVersion(episode.verified, previewVersion?.id);
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
                      {previewVerified ? (
                        <span
                          className="text-2xs px-1.5 py-0.5 rounded font-medium shrink-0 inline-flex items-center gap-1 bg-verified/15 text-verified"
                          title={`verified version (v${shortVersionId(episode.verified!.version_id)}) · ${VERIFIED_CAPTION}`}
                        >
                          <Star className="w-2.5 h-2.5" fill="currentColor" />
                          verified
                        </span>
                      ) : (
                        <span className={`text-2xs px-1.5 py-0.5 rounded font-medium shrink-0 ${c.bg} ${c.text}`}>
                          {previewLabel}
                        </span>
                      )}
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
        versions={recentActivityVersions}
        onPreview={openPreview}
      />

      <VersionsTable
        versions={transcriptVersions}
        heading="All transcript versions"
        firstColLabel="Step"
        countColLabel="Segments"
        onPreview={openPreview}
        onDelete={(step, id) => deleteVersionMutation.mutate({ step, id })}
        childrenByVersionId={childrenByTranscriptId}
        verifiedVersionId={episode.verified?.version_id ?? null}
      />

      {(episode.llm_failed_steps?.length ?? 0) > 0 && (
        <section className="space-y-2">
          <h4 className="text-sm font-medium px-1">Rejected correction &amp; translation batches</h4>
          <div className="rounded-lg border border-destructive/30 bg-destructive/5 divide-y divide-destructive/10">
            {episode.llm_failed_steps!.map((step) => (
              <button
                key={step}
                onClick={() => onNavigateStep(stepDisplay(step).editorStep)}
                className="w-full flex items-center gap-2 px-4 py-2 text-left text-xs hover:bg-destructive/10 transition"
              >
                <AlertTriangle className="w-3.5 h-3.5 text-destructive shrink-0" />
                <span className="text-destructive flex-1">
                  {stepDisplay(step).label}: some batches were rejected in the last auto run
                </span>
                <span className="text-2xs text-muted-foreground">Open &rarr;</span>
              </button>
            ))}
          </div>
        </section>
      )}

      {episode.indexed && indexEntries && indexEntries.length > 0 && (
        <section className="space-y-2">
          <h4 className="text-sm font-medium px-1">All index versions</h4>
          <div className="rounded-lg border border-border bg-card overflow-hidden">
            {indexEntries.map((e) => (
              <IndexRow
                key={e.collection}
                model={e.model}
                chunker={e.chunker}
                source={e.source}
                chunkCount={e.chunk_count}
                onInspect={() => setInspectTarget({ model: e.model, chunking: e.chunker })}
                deletion={{
                  onConfirm: () => deleteCollectionMutation.mutate(e.collection),
                  message: `Remove from ${e.collection}?`,
                }}
              />
            ))}
          </div>
        </section>
      )}

      <VersionsTable
        versions={otherFilesVersions}
        heading="All other files"
        firstColLabel="File"
        countColLabel="Size"
        showEdited={false}
        sizeColumn
        onDelete={(step, id) => deleteVersionMutation.mutate({ step, id })}
      />

      {(audioPath || outputDir) && previewSource && (
        <Suspense fallback={null}>
          <SegmentContextDialog
            open={true}
            onOpenChange={(open) => { if (!open) { setPreviewSource(null); setPreviewVersionId(null); } }}
            audioPath={audioPath ?? undefined}
            outputDir={outputDir ?? undefined}
            source={previewSource}
            versionId={previewVersionId ?? undefined}
            episodeTitle={episode.title}
            onSeek={(t) => audioPath && seekTo(audioPath, t)}
            onOpenEditor={() => {
              setPreviewSource(null);
              setPreviewVersionId(null);
              onNavigateStep(stepDisplay(previewSource).editorStep);
            }}
            verified={episode.verified ?? null}
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
