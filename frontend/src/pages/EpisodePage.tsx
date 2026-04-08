import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useState } from "react";
import { getEpisodes, getShowMeta, exportZipUrl, openFolder } from "@/api/client";
import { audioFileUrl } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { artworkUrl } from "@/api/filesystem";
import { uploadTranscript } from "@/api/transcribe";
import { useShowActions } from "@/hooks/useShowActions";
import { usePipelineDefaults } from "@/hooks/usePipelineConfig";
import DownloadDropdown from "@/components/common/DownloadDropdown";
import { useDropZone } from "@/hooks/useDropZone";
import DropOverlay from "@/components/common/DropOverlay";
import type { Episode, ShowMeta } from "@/api/types";
import { useAudioStore, useEpisodeStore, useTaskStore, useLayoutStore } from "@/stores";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import ShowSettings from "@/components/show/ShowSettings";
import SearchPanel from "@/components/search/SearchPanel";
import { formatDuration, formatDate, stripHtml, errorMessage } from "@/lib/utils";
import { useTheme } from "@/hooks/useTheme";
import {
  ArrowLeft,
  Play,
  Info,
  Download,
  Search,
  Settings,
  PanelLeftOpen,
  PanelLeftClose,
  FolderOpen,
  Home,
  Sun,
  Moon,
  Monitor,
} from "lucide-react";
import {
  PIPELINE_STEPS,
  STEP_BY_KEY,
  PipelineRow,
  SidebarButton,
  PipelineStatus,
  EpisodeDetails,
  type ActiveStep,
  type StepStatus,
} from "@/components/episode/PipelineSteps";

export default function EpisodePage({
  folder,
  stem,
  audioFilePath,
}: {
  folder?: string;
  stem?: string;
  audioFilePath?: string;
}) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { seekTo, setAudioMeta } = useAudioStore();
  const [activeStep, setActiveStep] = useState<ActiveStep>("info");
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const setHideAppSidebar = useLayoutStore((s) => s.setHideAppSidebar);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    setHideAppSidebar(true);
    return () => setHideAppSidebar(false);
  }, [setHideAppSidebar]);

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

  const goBack = () => {
    if (isStandalone) {
      navigate({ to: "/" });
    } else {
      navigate({
        to: "/show/$folder",
        params: { folder: encodeURIComponent(folder!) },
      });
    }
  };

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
        <Button onClick={goBack} variant="link" size="sm">
          Go back
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {isDragging && <DropOverlay message="Drop transcript file to import (JSON, SRT, VTT)" />}
      {/* Header */}
      <div className="px-6 py-4 border-b border-border flex items-center gap-4 relative overflow-hidden">
        {artwork && (
          <div
            className="absolute inset-0 bg-cover bg-center opacity-[0.08] blur-2xl scale-110 pointer-events-none"
            style={{ backgroundImage: `url(${artwork})` }}
          />
        )}
        <Button onClick={goBack} variant="ghost" size="sm">
          <ArrowLeft /> {isStandalone ? "Home" : "Episodes"}
        </Button>
        {episode.audio_path && artwork ? (
          <button
            onClick={() => seekTo(episode.audio_path!, 0)}
            className="relative group shrink-0"
          >
            <img src={artwork} alt={episode.title} className="w-12 h-8 object-cover rounded-lg" />
            <div className="absolute inset-0 rounded-lg bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
              <Play className="w-4 h-4 text-white fill-white" />
            </div>
          </button>
        ) : artwork ? (
          <img src={artwork} alt={episode.title} className="w-12 h-8 object-cover rounded-lg shrink-0" />
        ) : null}
        <div className="flex-1 min-w-0">
          <h2 className="text-lg font-semibold truncate">
            {episode.title}
          </h2>
          <div className="flex items-center gap-3 mt-0.5">
            <div className="flex gap-2 text-xs text-muted-foreground">
              {episode.episode_number != null && <span>#{episode.episode_number}</span>}
              {episode.pub_date && <span>{formatDate(episode.pub_date)}</span>}
              {episode.duration > 0 && <span>{formatDuration(episode.duration)}</span>}
            </div>
            <PipelineStatus episode={episode} />
          </div>
        </div>
        {episode.audio_path && (
          <Button
            onClick={() => seekTo(episode.audio_path!, 0)}
            variant="outline"
            size="sm"
          >
            <Play className="w-3.5 h-3.5" /> Play
          </Button>
        )}
      </div>

      {/* Description */}
      {episode.description && (
        <div className="px-6 py-3 border-b border-border">
          <p className="text-sm text-muted-foreground line-clamp-3">{stripHtml(episode.description)}</p>
        </div>
      )}

      {/* Body: sidebar + content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Unified sidebar */}
        <div
          className={`border-r border-border flex flex-col shrink-0 transition-all duration-200 ${
            sidebarExpanded ? "w-48" : "w-14"
          }`}
        >
          <nav className="flex-1 py-2 flex flex-col overflow-y-auto">
            {/* App items */}
            <SidebarButton icon={Home} label="Home" expanded={sidebarExpanded} onClick={() => navigate({ to: "/" })} />
            <SidebarButton icon={Settings} label="Settings" expanded={sidebarExpanded} onClick={() => navigate({ to: "/settings" })} />
            {(() => {
              const nextTheme = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
              const ThemeIcon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor;
              return <SidebarButton icon={ThemeIcon} label={`Theme: ${theme}`} expanded={sidebarExpanded} onClick={() => setTheme(nextTheme)} />;
            })()}

            {/* Episode sections: info/search meta items, then pipeline steps by section */}
            {[
              { items: [
                { key: "info" as const, label: "Info", icon: Info, status: false as StepStatus },
                { key: "search" as const, label: "Search", icon: Search, status: false as StepStatus },
              ]},
              { items: PIPELINE_STEPS.filter((s) => s.section === "core").map((s) => ({
                key: s.key as ActiveStep, label: s.label, icon: s.icon, status: s.status(episode),
              }))},
              { items: PIPELINE_STEPS.filter((s) => s.section === "bonus").map((s) => ({
                key: s.key as ActiveStep, label: s.label, icon: s.icon, status: s.status(episode),
              }))},
            ].map((section, si) => (
              <div key={si}>
                <div className="mx-3 my-1.5 border-t border-border" />
                {section.items.map(({ key, label, icon: Icon, status }) => (
                  <button
                    key={key}
                    onClick={() => setActiveStep(key)}
                    title={sidebarExpanded ? undefined : label}
                    className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm transition ${
                      activeStep === key
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                    }`}
                  >
                    <Icon className="w-5 h-5 shrink-0" />
                    {sidebarExpanded && <span className="truncate">{label}</span>}
                    {status && (
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${sidebarExpanded ? "ml-auto" : ""} ${status === "partial" ? "bg-warning" : "bg-success"}`} />
                    )}
                  </button>
                ))}
              </div>
            ))}
          </nav>
          <button
            onClick={() => setSidebarExpanded(!sidebarExpanded)}
            className="px-4 py-3 text-muted-foreground hover:text-foreground transition border-t border-border"
          >
            {sidebarExpanded ? (
              <PanelLeftClose className="w-5 h-5" />
            ) : (
              <PanelLeftOpen className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Main content */}
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
          />
        </div>
      </div>
    </div>
  );
}

function StepContent({ step, episode, folder, meta, isYouTube, onDownloadAudio, onImportSubs, downloadDisabled, downloadError }: { step: ActiveStep; episode: Episode; folder?: string; meta?: ShowMeta; isYouTube: boolean; onDownloadAudio: () => void; onImportSubs: (lang: string) => void; downloadDisabled: boolean; downloadError?: string }) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  if (step !== "info" && step !== "search") {
    return STEP_BY_KEY[step].component();
  }
  if (step === "search") return <SearchPanel scope="episode" />;
  return (
        <div className="p-6 space-y-6">
          <EpisodeDetails episode={episode} />

          {/* Show folder */}
          {folder && (
            <div className="space-y-1">
              <h4 className="text-sm font-medium">Show Folder</h4>
              <div className="flex items-center gap-2">
                <p className="text-xs text-muted-foreground font-mono break-all">{folder}</p>
                <button
                  onClick={() => openFolder(folder)}
                  className="shrink-0 text-muted-foreground hover:text-foreground transition"
                  title="Open in file manager"
                >
                  <FolderOpen className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* Input files */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Input Files</h4>
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
                <a href={audioFileUrl(episode.audio_path)} download>
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

          {/* Pipeline status */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Pipeline</h4>
            <div className="grid gap-1 max-w-md">
              {PIPELINE_STEPS.map((s) => (
                <PipelineRow
                  key={s.key}
                  label={s.rowLabel}
                  status={s.status(episode)}
                  detail={s.detail?.(episode)}
                  provenance={
                    s.provenanceKey
                      ? (episode.provenance?.[s.provenanceKey] as Record<string, unknown> | undefined)
                      : undefined
                  }
                  files={s.matchFiles ? episode.files?.filter((f) => s.matchFiles!(episode, f)) : undefined}
                />
              ))}
            </div>
          </div>

          {/* Export */}
          {episode.audio_path && (
            <div className="space-y-3">
              <h4 className="text-sm font-medium">Export</h4>
              <a href={exportZipUrl(episode.audio_path)} download>
                <Button variant="outline" size="sm">
                  <Download className="w-3.5 h-3.5" /> Download ZIP
                </Button>
              </a>
            </div>
          )}

          {/* Show settings dialog */}
          {folder && meta && (
            <>
              <Button variant="outline" size="sm" onClick={() => setSettingsOpen(true)}>
                <Settings className="w-3.5 h-3.5" /> Show Settings
              </Button>
              <Dialog open={settingsOpen} onOpenChange={setSettingsOpen}>
                <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
                  <DialogHeader>
                    <DialogTitle>Show Settings — {meta.name}</DialogTitle>
                  </DialogHeader>
                  <ShowSettings folder={folder} meta={meta} />
                </DialogContent>
              </Dialog>
            </>
          )}
        </div>
  );
}
