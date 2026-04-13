import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useState } from "react";
import { getEpisodes, getShowMeta, exportZipUrl, openFolder } from "@/api/client";
import { audioFileUrl } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { artworkUrl } from "@/api/filesystem";
import { uploadTranscript, getSpeakerMap } from "@/api/transcribe";
import { getSegmentsPreview as getTranscribePreview } from "@/api/transcribe";
import { getCorrectSegmentsPreview as getCorrectPreview } from "@/api/correct";
import { useShowActions } from "@/hooks/useShowActions";
import { usePipelineDefaults } from "@/hooks/usePipelineConfig";
import DownloadDropdown from "@/components/common/DownloadDropdown";
import { useDropZone } from "@/hooks/useDropZone";
import DropOverlay from "@/components/common/DropOverlay";
import PageHeader from "@/components/layout/PageHeader";
import AppSidebar from "@/components/layout/AppSidebar";
import type { Episode, ShowMeta } from "@/api/types";
import { useAudioStore, useEpisodeStore, useTaskStore } from "@/stores";
import { Button } from "@/components/ui/button";
import SearchPanel from "@/components/search/SearchPanel";
import { formatDuration, formatDate, stripHtml, errorMessage } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";
import {
  Play,
  Info,
  Download,
  Search,
  FolderOpen,
} from "lucide-react";
import {
  PIPELINE_STEPS,
  STEP_BY_KEY,
  PipelineRow,
  PipelineStatus,
  type ActiveStep,
  type StepStatus,
} from "@/components/episode/PipelineSteps";

function buildSidebarSections(episode: Episode) {
  const meta = [
    { key: "info" as const, label: "Info", icon: Info, status: false as StepStatus },
    { key: "search" as const, label: "Search", icon: Search, status: false as StepStatus },
  ];
  const core: typeof meta = [];
  const bonus: typeof meta = [];
  for (const s of PIPELINE_STEPS) {
    const item = { key: s.key as ActiveStep, label: s.label, icon: s.icon, status: s.status(episode) };
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
  const { seekTo, setAudioMeta } = useAudioStore();
  const [activeStep, setActiveStep] = useState<ActiveStep>("info");
  const [descExpanded, setDescExpanded] = useState(false);

  const isStandalone = !!audioFilePath;
  const { downloadTaskId } = useTaskStore();
  const pipelineDefaults = usePipelineDefaults();

  const { data: meta } = useQuery({
    queryKey: queryKeys.showMeta(folder),
    queryFn: () => getShowMeta(folder!),
    enabled: !!folder,
  });

  const { data: episodes } = useQuery({
    queryKey: queryKeys.episodes(folder, pipelineDefaults),
    queryFn: () => getEpisodes(folder!, pipelineDefaults),
    placeholderData: keepPreviousData,
    enabled: !!folder,
    refetchInterval: downloadTaskId ? 5000 : false,
  });

  const { downloadMutation: episodeDownloadMutation, importSubsMutation, isYouTube } = useShowActions(folder ?? "", meta, { withSubs: false });

  const episode = isStandalone
    ? {
        id: audioFilePath,
        title: audioFilePath.split("/").pop()?.replace(/\.[^.]+$/, "") || "Audio",
        stem: audioFilePath.split("/").pop()?.replace(/\.[^.]+$/, "") || null,
        pub_date: null,
        description: "",
        audio_url: null,
        duration: 0,
        episode_number: null,
        audio_path: audioFilePath,
        downloaded: true,
        transcribed: false,
        corrected: false,
        indexed: false,
        synthesized: false,
        translations: [] as string[],
        artwork_url: "",
      }
    : episodes?.find((e) => e.stem === stem || e.id === stem);


  const artwork = episode?.artwork_url || (meta?.artwork_url ? artworkUrl(folder) : "");

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
        queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments("transcribe", episode?.audio_path) });
        queryClient.invalidateQueries({ queryKey: queryKeys.stepSegments("correct", episode?.audio_path) });
        queryClient.invalidateQueries({ queryKey: queryKeys.transcribeSegments(episode?.audio_path) });
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

  const sidebarSections = useMemo(() => buildSidebarSections(episode), [episode]);

  return (
    <div className="flex flex-col h-full">
      {isDragging && <DropOverlay message="Drop a transcript file here (JSON, SRT, VTT)" />}
      <PageHeader
        title={episode.title}
        className="relative overflow-hidden"
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
        artwork={
          episode.audio_path && artwork ? (
            <button
              onClick={() => seekTo(episode.audio_path!, 0)}
              className="relative group shrink-0"
            >
              <img src={artwork} alt={episode.title} className="w-8 h-8 object-cover rounded-md" />
              <div className="absolute inset-0 rounded-lg bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                <Play className="w-4 h-4 text-white fill-white" />
              </div>
            </button>
          ) : artwork ? (
            <img src={artwork} alt={episode.title} className="w-8 h-8 object-cover rounded-md shrink-0" />
          ) : undefined
        }
        subtitle={
          <>
            <div className="flex gap-2 text-xs text-muted-foreground">
              {meta?.name && <span>{meta.name}</span>}
              {episode.episode_number != null && <span>#{episode.episode_number}</span>}
              {episode.pub_date && <span>{formatDate(episode.pub_date)}</span>}
              {episode.duration > 0 && <span>{formatDuration(episode.duration)}</span>}
            </div>
            <PipelineStatus episode={episode} />
          </>
        }
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
      >
        {artwork && (
          <div
            className="absolute inset-0 bg-cover bg-center opacity-[0.08] blur-2xl scale-110 pointer-events-none"
            style={{ backgroundImage: `url(${artwork})` }}
          />
        )}
      </PageHeader>

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
          </div>
        </div>
      </div>
    </div>
  );
}

/** Extract a human-readable provenance label for a pipeline step. */
function getProvenanceLabel(provenance: Record<string, unknown> | undefined): string | undefined {
  if (!provenance) return undefined;
  const params = provenance.params as Record<string, unknown> | undefined;
  const sourceChain = params?.source_chain as string[] | undefined;
  // Show the last meaningful entry from the source chain
  let label = sourceChain?.length ? sourceChain[sourceChain.length - 1] : undefined;
  // Fall back to model if no chain
  if (!label && provenance.model) label = String(provenance.model);
  if (!label) return undefined;
  // Append "(edited)" for manually edited versions
  if (provenance.manual_edit) label += ", edited";
  return label;
}

/** Map pipeline step keys to their provenance dict keys. */
function stepProvenanceKey(stepKey: string, episode: Episode): string | undefined {
  if (stepKey === "transcribe") return "transcript";
  if (stepKey === "correct") return "corrected";
  if (stepKey === "translate") {
    // Return the first translation language found in provenance
    return episode.translations?.find((lang) => episode.provenance?.[lang]);
  }
  return undefined;
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

  const speakers = useMemo(() => {
    if (!speakerMap) return [];
    return [...new Set(Object.values(speakerMap))].filter(Boolean);
  }, [speakerMap]);

  const nextAction = getNextAction(episode);

  return (
        <div className="p-6 space-y-6 max-w-2xl">
          {nextAction && (
            <p className="text-sm text-muted-foreground italic">{nextAction}</p>
          )}

          <div className="space-y-3">
            <h4 className="text-sm font-medium">Input</h4>
            <PipelineRow
              label="Downloaded"
              status={episode.downloaded ? "done" : (episode.files ?? []).some((f) => !f.includes("/") || f.endsWith(".vtt") || f.endsWith(".srt")) ? "partial" : false}
              detail={!episode.downloaded && (episode.files ?? []).some((f) => f.endsWith(".vtt") || f.endsWith(".srt")) ? "subtitles only" : undefined}
              files={(episode.files ?? []).filter((f) => !f.includes("/") || f.endsWith(".vtt") || f.endsWith(".srt"))}
            />
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
              {episode.audio_path && (
                <a href={audioFileUrl(episode.audio_path)} download={`${episode.title}.${episode.audio_path.split(".").pop()}`}>
                  <Button variant="outline" size="sm">
                    <Download className="w-3.5 h-3.5" /> Save audio
                  </Button>
                </a>
              )}
            </div>
            {downloadError && (
              <p className="text-destructive text-xs">{downloadError}</p>
            )}
          </div>

          <div className="space-y-3">
            <h4 className="text-sm font-medium">Pipeline</h4>
            <div className="grid gap-1">
              {PIPELINE_STEPS.map((s) => {
                const provKey = stepProvenanceKey(s.key, episode);
                const prov = provKey ? (episode.provenance?.[provKey] as Record<string, unknown> | undefined) : undefined;
                return (
                  <PipelineRow
                    key={s.key}
                    label={s.rowLabel}
                    status={s.status(episode)}
                    detail={s.detail?.(episode)}
                    subtitle={getProvenanceLabel(prov)}
                    provenance={prov}
                    files={s.matchFiles ? episode.files?.filter((f) => s.matchFiles!(episode, f)) : undefined}
                    onClick={() => onNavigateStep(s.key)}
                  />
                );
              })}
            </div>
          </div>

          {previewSegments && previewSegments.length > 0 && (
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
        </div>
  );
}
