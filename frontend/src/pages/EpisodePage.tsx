import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { getEpisodes, getShowMeta } from "@/api/client";
import type { Episode, ShowMeta } from "@/api/types";
import { useAppStore } from "@/store";
import { Button } from "@/components/ui/button";
import TranscribePanel from "@/components/transcribe/TranscribePanel";
import PolishPanel from "@/components/polish/PolishPanel";
import TranslatePanel from "@/components/translate/TranslatePanel";
import SynthesizePanel from "@/components/synthesize/SynthesizePanel";
import IndexPanel from "@/components/index/IndexPanel";
import SearchPanel from "@/components/search/SearchPanel";
import { formatDuration, formatDate } from "@/lib/utils";
import {
  ArrowLeft,
  Play,
  Mic,
  Sparkles,
  Languages,
  AudioLines,
  Database,
  Search,
  PanelLeftOpen,
  PanelLeftClose,
  Info,
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
  const { playAudio } = useAppStore();
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
        raw_transcript: false,
        validated_transcript: false,
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

  const stepDone = (step: PipelineStep): boolean => {
    switch (step) {
      case "transcribe": return episode.transcribed;
      case "polish": return episode.polished;
      case "translate": return episode.translations.length > 0;
      case "synthesize": return episode.synthesized;
      case "index": return episode.indexed;
      default: return false;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border flex items-center gap-4">
        <Button onClick={goBack} variant="ghost" size="sm">
          <ArrowLeft /> {isStandalone ? "Home" : "Episodes"}
        </Button>
        {episode.audio_path && artwork ? (
          <button
            onClick={() => playAudio(episode.audio_path!, episode.title, artwork, meta?.name)}
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
            onClick={() => playAudio(episode.audio_path!, episode.title, artwork, meta?.name)}
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
          <p className="text-sm text-muted-foreground line-clamp-3">{episode.description}</p>
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
                {!sidebarExpanded && stepDone(key) && (
                  <span className="w-1.5 h-1.5 rounded-full bg-green-500 shrink-0" />
                )}
                {sidebarExpanded && stepDone(key) && (
                  <span className="ml-auto w-1.5 h-1.5 rounded-full bg-green-500 shrink-0" />
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
          <StepContent
            step={activeStep}
            episode={episode}
            showMeta={meta}
          />
        </div>
      </div>
    </div>
  );
}

function StepContent({
  step,
  episode,
  showMeta,
}: {
  step: PipelineStep;
  episode: Episode;
  showMeta?: ShowMeta | null;
}) {
  switch (step) {
    case "info":
      return (
        <div className="p-6 space-y-4">
          <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm max-w-md">
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
            <span className="text-muted-foreground">Downloaded</span>
            <span>{episode.downloaded ? "Yes" : "No"}</span>
            <span className="text-muted-foreground">Transcribed</span>
            <span>{episode.transcribed ? "Yes" : "No"}</span>
            <span className="text-muted-foreground">Polished</span>
            <span>{episode.polished ? "Yes" : "No"}</span>
            {episode.translations.length > 0 && (
              <>
                <span className="text-muted-foreground">Translations</span>
                <span>{episode.translations.join(", ")}</span>
              </>
            )}
            <span className="text-muted-foreground">Indexed</span>
            <span>{episode.indexed ? "Yes" : "No"}</span>
          </div>
        </div>
      );

    case "transcribe":
      return <TranscribePanel episode={episode} showMeta={showMeta} />;

    case "polish":
      return <PolishPanel episode={episode} showMeta={showMeta} />;

    case "translate":
      return <TranslatePanel episode={episode} showMeta={showMeta} />;

    case "synthesize":
      return <SynthesizePanel episode={episode} showMeta={showMeta} />;

    case "index":
      return <IndexPanel episode={episode} showMeta={showMeta} />;

    case "search":
      return <SearchPanel episode={episode} showMeta={showMeta} />;
  }
}

function PipelineStatus({ episode }: { episode: Episode }) {
  const steps = [
    { key: "transcribed", label: "Transcribed", done: episode.transcribed },
    { key: "polished", label: "Polished", done: episode.polished },
    {
      key: "translated",
      label: "Translated",
      done: episode.translations.length > 0,
    },
    { key: "synthesized", label: "Synthesized", done: episode.synthesized },
    { key: "indexed", label: "Indexed", done: episode.indexed },
  ];

  return (
    <div className="flex gap-2 mt-1">
      {steps.map((step) => (
        <span
          key={step.key}
          className={`text-xs px-2 py-0.5 rounded-full ${
            step.done
              ? "bg-green-900/40 text-green-400"
              : "bg-muted text-muted-foreground"
          }`}
        >
          {step.label}
        </span>
      ))}
    </div>
  );
}
