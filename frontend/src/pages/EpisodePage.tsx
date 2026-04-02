import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useEffect, useState } from "react";
import { getEpisodes, getShowMeta, exportZipUrl } from "@/api/client";
import { audioFileUrl } from "@/api/client";
import { uploadTranscript } from "@/api/transcribe";
import { useDropZone } from "@/hooks/useDropZone";
import DropOverlay from "@/components/common/DropOverlay";
import type { Episode, ShowMeta } from "@/api/types";
import { useAudioStore, useEpisodeStore } from "@/stores";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import ShowSettings from "@/components/show/ShowSettings";
import TranscribePanel from "@/components/transcribe/TranscribePanel";
import PolishPanel from "@/components/polish/PolishPanel";
import TranslatePanel from "@/components/translate/TranslatePanel";
import SynthesizePanel from "@/components/synthesize/SynthesizePanel";
import IndexPanel from "@/components/index/IndexPanel";
import SearchPanel from "@/components/search/SearchPanel";
import { formatDuration, formatDate, stripHtml } from "@/lib/utils";
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
} from "lucide-react";

type PipelineStep = "info" | "transcribe" | "polish" | "translate" | "synthesize" | "index" | "search";

const PIPELINE_STEPS: { key: PipelineStep; label: string; icon: typeof Mic }[] = [
  { key: "info", label: "Info", icon: Info },
  { key: "transcribe", label: "Transcribe", icon: Mic },
  { key: "polish", label: "Polish", icon: Sparkles },
  { key: "translate", label: "Translate", icon: Languages },
  { key: "synthesize", label: "Synthesize", icon: AudioLines },
  { key: "index", label: "Index", icon: Database },
  { key: "search", label: "Search", icon: Search },
];

function stepDoneCheck(step: PipelineStep, episode: Episode): boolean {
  switch (step) {
    case "transcribe": return episode.transcribed;
    case "polish": return episode.polished;
    case "translate": return episode.translations.length > 0;
    case "synthesize": return episode.synthesized;
    case "index": return episode.indexed;
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

  const isStandalone = !!audioFilePath;

  const { data: meta } = useQuery({
    queryKey: ["showMeta", folder],
    queryFn: () => getShowMeta(folder!),
    enabled: !!folder,
  });

  const { data: episodes } = useQuery({
    queryKey: ["episodes", folder],
    queryFn: () => getEpisodes(folder!),
    enabled: !!folder,
  });

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

  const artwork = episode?.artwork_url || meta?.artwork_url || "";

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
    setEpisode(episode ?? null);
  }, [episode, setEpisode]);

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
    accept: [".json"],
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
      {isDragging && <DropOverlay message="Drop transcript JSON to import" />}
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
            <img src={artwork} alt="" className="w-10 h-10 rounded-lg" />
            <div className="absolute inset-0 rounded-lg bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
              <Play className="w-4 h-4 text-white fill-white" />
            </div>
          </button>
        ) : artwork ? (
          <img src={artwork} alt="" className="w-10 h-10 rounded-lg shrink-0" />
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
        {episode.audio_path && (
          <a href={audioFileUrl(episode.audio_path)} download>
            <Button variant="outline" size="sm" title="Download audio file">
              <Download className="w-3.5 h-3.5" />
            </Button>
          </a>
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
        {/* Pipeline sidebar */}
        <div
          className={`border-r border-border flex flex-col shrink-0 transition-all duration-200 ${
            sidebarExpanded ? "w-48" : "w-14"
          }`}
        >
          <nav className="flex-1 py-3 flex flex-col gap-1">
            {PIPELINE_STEPS.map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveStep(key)}
                title={sidebarExpanded ? undefined : label}
                className={`w-full flex items-center gap-3 px-4 py-3 text-sm transition ${
                  activeStep === key
                    ? "bg-accent text-accent-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                }`}
              >
                <Icon className="w-5 h-5 shrink-0" />
                {sidebarExpanded && <span className="truncate">{label}</span>}
                {stepDoneCheck(key, episode) && (
                  <span className={`w-1.5 h-1.5 rounded-full bg-green-500 shrink-0 ${sidebarExpanded ? "ml-auto" : ""}`} />
                )}
              </button>
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
          <StepContent step={activeStep} episode={episode} folder={folder} meta={meta} episodes={episodes} />
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

function StepContent({ step, episode, folder, meta }: { step: PipelineStep; episode: Episode; folder?: string; meta?: ShowMeta }) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  switch (step) {
    case "info":
      return (
        <div className="p-6 space-y-6">
          <EpisodeDetails episode={episode} />

          {/* Pipeline status */}
          <div className="space-y-3">
            <h4 className="text-sm font-medium">Pipeline</h4>
            <div className="grid gap-2 max-w-sm">
              {[
                { label: "Downloaded", done: episode.downloaded },
                { label: "Transcribed", done: episode.transcribed },
                { label: "Polished", done: episode.polished },
                { label: "Translated", done: episode.translations.length > 0, detail: episode.translations.length > 0 ? episode.translations.join(", ") : undefined },
                { label: "Synthesized", done: episode.synthesized },
                { label: "Indexed", done: episode.indexed },
              ].map(({ label, done, detail }) => (
                <div key={label} className="flex items-center gap-3 text-sm">
                  <span className={`w-2 h-2 rounded-full shrink-0 ${done ? "bg-green-500" : "bg-muted-foreground/30"}`} />
                  <span className={done ? "text-foreground" : "text-muted-foreground"}>{label}</span>
                  {detail && <span className="text-xs text-muted-foreground ml-auto">{detail}</span>}
                </div>
              ))}
            </div>
          </div>

          {/* Audio path */}
          {episode.audio_path && (
            <div className="space-y-1">
              <h4 className="text-sm font-medium">File</h4>
              <p className="text-xs text-muted-foreground font-mono break-all">{episode.audio_path}</p>
            </div>
          )}

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
  { step: "translate", label: "Translated" },
  { step: "synthesize", label: "Synthesized" },
  { step: "index", label: "Indexed" },
];

function PipelineStatus({ episode }: { episode: Episode }) {
  return (
    <div className="flex gap-2 mt-1">
      {STATUS_BADGES.map(({ step, label }) => {
        const done = stepDoneCheck(step, episode);
        return (
          <span
            key={step}
            className={`text-xs px-2 py-0.5 rounded-full ${
              done
                ? "bg-green-900/40 text-green-400"
                : "bg-muted text-muted-foreground"
            }`}
          >
            {label}
          </span>
        );
      })}
    </div>
  );
}
