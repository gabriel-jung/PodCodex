import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useState } from "react";
import { getEpisodes, getShowMeta, exportZipUrl, openFolder } from "@/api/client";
import { audioFileUrl } from "@/api/client";
import { artworkUrl } from "@/api/filesystem";
import { uploadTranscript } from "@/api/transcribe";
import { useShowActions } from "@/hooks/useShowActions";
import { usePipelineConfig } from "@/hooks/usePipelineConfig";
import DownloadDropdown from "@/components/common/DownloadDropdown";
import { useDropZone } from "@/hooks/useDropZone";
import DropOverlay from "@/components/common/DropOverlay";
import type { Episode, ShowMeta } from "@/api/types";
import { useAudioStore, useEpisodeStore, useTaskStore, useLayoutStore } from "@/stores";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import ShowSettings from "@/components/show/ShowSettings";
import TranscribePanel from "@/components/transcribe/TranscribePanel";
import PolishPanel from "@/components/polish/PolishPanel";
import TranslatePanel from "@/components/translate/TranslatePanel";
import SynthesizePanel from "@/components/synthesize/SynthesizePanel";
import IndexPanel from "@/components/index/IndexPanel";
import SearchPanel from "@/components/search/SearchPanel";
import { formatDuration, formatDate, stripHtml, errorMessage } from "@/lib/utils";
import { useTheme } from "@/hooks/useTheme";
import {
  ArrowLeft,
  Play,
  Mic,
  Sparkles,
  Languages,
  AudioLines,
  Database,
  Search,
  Info,
  Download,
  Settings,
  PanelLeftOpen,
  PanelLeftClose,
  FolderOpen,
  Home,
  Sun,
  Moon,
  Monitor,
} from "lucide-react";

type PipelineStep = "info" | "transcribe" | "polish" | "translate" | "synthesize" | "index" | "search";

type SidebarSection = {
  items: { key: PipelineStep | string; label: string; icon: typeof Mic }[];
};

const SIDEBAR_SECTIONS: SidebarSection[] = [
  // Episode info
  { items: [
    { key: "info", label: "Info", icon: Info },
    { key: "search", label: "Search", icon: Search },
  ]},
  // Pipeline
  { items: [
    { key: "transcribe", label: "Transcribe", icon: Mic },
    { key: "polish", label: "Polish", icon: Sparkles },
    { key: "index", label: "Index", icon: Database },
  ]},
  // Bonus
  { items: [
    { key: "translate", label: "Translate", icon: Languages },
    { key: "synthesize", label: "Synthesize", icon: AudioLines },
  ]},
];

/** "done" = green, "partial" = yellow (outdated/raw), false = grey */
function stepStatus(step: PipelineStep, episode: Episode): "done" | "partial" | false {
  switch (step) {
    case "transcribe": {
      const s = episode.transcribe_status;
      if (s === "outdated") return "partial";
      return episode.transcribed ? (s === "done" ? "done" : "done") : false;
    }
    case "polish": {
      const s = episode.polish_status;
      if (s === "outdated") return "partial";
      return episode.polished ? "done" : false;
    }
    case "translate": {
      const s = episode.translate_status;
      if (s === "outdated") return "partial";
      return episode.translations.length > 0 ? "done" : false;
    }
    case "synthesize": return episode.synthesized ? "done" : false;
    case "index": return episode.indexed ? "done" : false;
    default: return false;
  }
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
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { seekTo, setAudioMeta } = useAudioStore();
  const [activeStep, setActiveStep] = useState<PipelineStep>("info");
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const setHideAppSidebar = useLayoutStore((s) => s.setHideAppSidebar);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    setHideAppSidebar(true);
    return () => setHideAppSidebar(false);
  }, [setHideAppSidebar]);

  const isStandalone = !!audioFilePath;
  const { downloadTaskId } = useTaskStore();
  const { tc, llm, targetLang } = usePipelineConfig();

  const pipelineDefaults = useMemo(() => ({
    model_size: tc.modelSize,
    diarize: tc.diarize,
    llm_mode: llm.mode === "api" ? "api" : "ollama",
    llm_provider: llm.mode === "api" ? llm.provider : "",
    llm_model: llm.model,
    target_lang: targetLang,
  }), [tc.modelSize, tc.diarize, llm.mode, llm.provider, llm.model, targetLang]);

  const { data: meta } = useQuery({
    queryKey: ["showMeta", folder],
    queryFn: () => getShowMeta(folder!),
    enabled: !!folder,
  });

  const { data: episodes } = useQuery({
    queryKey: ["episodes", folder, pipelineDefaults],
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
        polished: false,
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
        queryClient.invalidateQueries({ queryKey: ["segments"] });
        queryClient.invalidateQueries({ queryKey: ["episodes"] });
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
            <img src={artwork} alt="" className="w-12 h-8 object-cover rounded-lg" />
            <div className="absolute inset-0 rounded-lg bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
              <Play className="w-4 h-4 text-white fill-white" />
            </div>
          </button>
        ) : artwork ? (
          <img src={artwork} alt="" className="w-12 h-8 object-cover rounded-lg shrink-0" />
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

            {/* Episode sections */}
            {SIDEBAR_SECTIONS.map((section, si) => (
              <div key={si}>
                <div className="mx-3 my-1.5 border-t border-border" />
                {section.items.map(({ key, label, icon: Icon }) => {
                  const status = stepStatus(key as PipelineStep, episode);
                  return (
                    <button
                      key={key}
                      onClick={() => setActiveStep(key as PipelineStep)}
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
                        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${sidebarExpanded ? "ml-auto" : ""} ${status === "partial" ? "bg-yellow-500" : "bg-green-500"}`} />
                      )}
                    </button>
                  );
                })}
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

function EpisodeDetails({ episode }: { episode: Episode }) {
  if (episode.episode_number == null && !episode.pub_date && episode.duration <= 0) return null;
  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium">Details</h4>
      <div className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm max-w-md">
        {episode.episode_number != null && (
          <>
            <span className="text-muted-foreground">Episode</span>
            <span>#{episode.episode_number}</span>
          </>
        )}
        {episode.pub_date && (
          <>
            <span className="text-muted-foreground">Published</span>
            <span>{formatDate(episode.pub_date)}</span>
          </>
        )}
        {episode.duration > 0 && (
          <>
            <span className="text-muted-foreground">Duration</span>
            <span>{formatDuration(episode.duration)}</span>
          </>
        )}
      </div>
    </div>
  );
}

function StepContent({ step, episode, folder, meta, isYouTube, onDownloadAudio, onImportSubs, downloadDisabled, downloadError }: { step: PipelineStep; episode: Episode; folder?: string; meta?: ShowMeta; isYouTube: boolean; onDownloadAudio: () => void; onImportSubs: (lang: string) => void; downloadDisabled: boolean; downloadError?: string }) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  switch (step) {
    case "info":
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
            <DownloadStatus episode={episode} />
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
              <PipelineRow
                label="Transcribed"
                status={stepStatus("transcribe", episode)}
                provenance={episode.provenance?.transcript as Record<string, unknown> | undefined}
                files={episode.files?.filter((f) => f.includes("transcript.") || f.includes("segments.") || f.includes("diarization.") || f.includes("speaker_map."))}
              />
              <PipelineRow
                label="Polished"
                status={stepStatus("polish", episode)}
                provenance={episode.provenance?.polished as Record<string, unknown> | undefined}
                files={episode.files?.filter((f) => f.includes(".polished."))}
              />
              <PipelineRow
                label="Translated"
                status={stepStatus("translate", episode)}
                detail={episode.translations.length > 0 ? episode.translations.join(", ") : undefined}
                files={episode.files?.filter((f) => f.includes(".translated.") || episode.translations.some((lang) => f.includes(`.${lang}.`)))}
              />
              <PipelineRow
                label="Synthesized"
                status={episode.synthesized ? "done" : false}
                files={episode.files?.filter((f) => f.includes(".synthesized."))}
              />
              <PipelineRow
                label="Indexed"
                status={episode.indexed ? "done" : false}
              />
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

    case "transcribe":
      return <TranscribePanel />;

    case "polish":
      return <PolishPanel />;

    case "translate":
      return <TranslatePanel />;

    case "synthesize":
      return <SynthesizePanel />;

    case "index":
      return <IndexPanel />;

    case "search":
      return <SearchPanel scope="episode" />;
  }
}

const STATUS_BADGES: { step: PipelineStep; label: string }[] = [
  { step: "transcribe", label: "Transcribed" },
  { step: "polish", label: "Polished" },
  { step: "index", label: "Indexed" },
];

function PipelineRow({ label, status, detail, provenance, files }: {
  label: string;
  status: "done" | "partial" | false;
  detail?: string;
  provenance?: Record<string, unknown>;
  files?: string[];
}) {
  const [expanded, setExpanded] = useState(false);
  const hasFiles = (files?.length ?? 0) > 0;
  const hasInfo = !!provenance || hasFiles;

  return (
    <div>
      <div className="flex items-center gap-3 text-sm">
        <span className={`w-2 h-2 rounded-full shrink-0 ${status === "done" ? "bg-green-500" : status === "partial" ? "bg-yellow-500" : "bg-muted-foreground/30"}`} />
        <span className={status ? "text-foreground" : "text-muted-foreground"}>{label}</span>
        {detail && <span className="text-xs text-muted-foreground">{detail}</span>}
        {hasInfo && status && (
          <button onClick={() => setExpanded(!expanded)} className="text-xs text-muted-foreground hover:text-foreground transition">
            {hasFiles ? `${files!.length} file${files!.length !== 1 ? "s" : ""} ` : ""}{expanded ? "▾" : "▸"}
          </button>
        )}
      </div>
      {expanded && (
        <div className="mt-1 ml-5 space-y-0.5">
          {files?.map((f) => (
            <div key={f} className="text-xs text-muted-foreground font-mono truncate">{f}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function DownloadStatus({ episode }: { episode: Episode }) {
  // Input files: audio (no slash = show root) + subtitles (.vtt/.srt)
  const inputFiles = (episode.files ?? []).filter((f) =>
    !f.includes("/") || f.endsWith(".vtt") || f.endsWith(".srt"),
  );
  const hasInputFiles = inputFiles.length > 0;
  const status = episode.downloaded ? "done" : hasInputFiles ? "partial" : false;
  const [expanded, setExpanded] = useState(false);

  return (
    <div>
      <div className="flex items-center gap-3 text-sm">
        <span className={`w-2 h-2 rounded-full shrink-0 ${status === "done" ? "bg-green-500" : status === "partial" ? "bg-yellow-500" : "bg-muted-foreground/30"}`} />
        <span className={status ? "text-foreground" : "text-muted-foreground"}>Downloaded</span>
        {status === "partial" && <span className="text-xs text-muted-foreground">subtitles only</span>}
        {hasInputFiles && (
          <button onClick={() => setExpanded(!expanded)} className="text-xs text-muted-foreground hover:text-foreground transition">
            {inputFiles.length} file{inputFiles.length !== 1 ? "s" : ""} {expanded ? "▾" : "▸"}
          </button>
        )}
      </div>
      {expanded && (
        <div className="mt-1.5 ml-5 space-y-0.5">
          {inputFiles.map((f) => (
            <div key={f} className="text-xs text-muted-foreground font-mono truncate">{f}</div>
          ))}
        </div>
      )}
    </div>
  );
}


function SidebarButton({ icon: Icon, label, expanded, onClick, active }: {
  icon: typeof Home;
  label: string;
  expanded: boolean;
  onClick: () => void;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={expanded ? undefined : label}
      className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm transition ${
        active
          ? "bg-accent text-accent-foreground"
          : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
      }`}
    >
      <Icon className="w-5 h-5 shrink-0" />
      {expanded && <span className="truncate">{label}</span>}
    </button>
  );
}

function PipelineStatus({ episode }: { episode: Episode }) {
  const visible = STATUS_BADGES.filter(({ step }) => stepStatus(step, episode));
  if (visible.length === 0) return null;
  return (
    <div className="flex gap-1.5">
      {visible.map(({ step, label }) => {
        const status = stepStatus(step, episode);
        return (
          <span
            key={step}
            className={`text-[10px] px-1.5 py-0.5 rounded-full ${
              status === "partial"
                ? "bg-yellow-900/40 text-yellow-400"
                : "bg-green-900/40 text-green-400"
            }`}
          >
            {label}
          </span>
        );
      })}
    </div>
  );
}
