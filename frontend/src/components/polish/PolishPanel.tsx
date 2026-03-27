import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Episode, ShowMeta } from "@/api/types";
import {
  getPolishSegments,
  getPolishSegmentsRaw,
  getPolishVersionInfo,
  savePolishSegments,
  getSegments,
  startPolish,
  skipPolish,
  getPolishManualPrompts,
  applyPolishManual,
} from "@/api/client";
import { buildDefaultContext } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { SkipForward } from "lucide-react";
import SegmentEditor from "@/components/editor/SegmentEditor";
import PipelinePanel from "@/components/common/PipelinePanel";
import HelpLabel from "@/components/common/HelpLabel";
import LLMControls, { type LLMConfig } from "@/components/common/LLMControls";

interface PolishPanelProps {
  episode: Episode;
  showMeta?: ShowMeta | null;
}

export default function PolishPanel({ episode, showMeta }: PolishPanelProps) {
  const queryClient = useQueryClient();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(!episode.polished);

  const [config, setConfig] = useState<LLMConfig>({
    mode: "ollama",
    provider: "openai",
    model: "",
    context: buildDefaultContext(episode, showMeta),
    sourceLang: showMeta?.language || "French",
    batchSize: 10,
  });
  const [engine, setEngine] = useState("Whisper");

  const { data: referenceSegments } = useQuery({
    queryKey: ["transcribe", "segments", episode.audio_path],
    queryFn: () => getSegments(episode.audio_path!),
    enabled: !!episode.audio_path && episode.transcribed,
  });

  const startMutation = useMutation({
    mutationFn: () =>
      startPolish({
        audio_path: episode.audio_path!,
        mode: config.mode === "api" ? "api" : "ollama",
        provider: config.mode === "api" ? config.provider : undefined,
        model: config.model,
        context: config.context,
        source_lang: config.sourceLang,
        batch_size: config.batchSize,
        engine,
      }),
    onSuccess: (data) => setTaskId(data.task_id),
  });

  const skipMutation = useMutation({
    mutationFn: () => skipPolish(episode.audio_path!),
    onSuccess: () => {
      refreshQueries();
      setExpanded(false);
    },
  });

  const refreshQueries =() => {
    queryClient.invalidateQueries({ queryKey: ["polish"] });
    queryClient.invalidateQueries({ queryKey: ["episodes"] });
  };

  const handleComplete = () => {
    refreshQueries();
    setTaskId(null);
    setExpanded(false);
  };

  return (
    <PipelinePanel
      title="Polish"
      description="Use AI to fix spelling mistakes, punctuation, and other transcription errors. Can run locally or through a cloud service."
      prerequisite={!episode.transcribed ? "You need a transcript first. Go to the Transcribe tab to create one." : undefined}
      done={episode.polished}
      expanded={expanded}
      onToggle={() => setExpanded(!expanded)}
      rerunLabel="Re-run polish"
      taskId={taskId}
      onTaskComplete={handleComplete}
      emptyMessage="Polish pipeline not yet run for this episode."
      controls={
        <>
          <LLMControls
            config={config}
            onChange={(patch) => setConfig({ ...config, ...patch })}
            onRun={() => startMutation.mutate()}
            isPending={startMutation.isPending}
            error={startMutation.isError ? (startMutation.error as Error).message : null}
            runLabel="Run polish"
            extraFields={
              <>
                <HelpLabel label="Engine" help="Which transcription engine produced the text. Affects the correction prompt — Whisper and Voxtral make different kinds of errors." />
                <select
                  value={engine}
                  onChange={(e) => setEngine(e.target.value)}
                  className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
                >
                  <option value="Whisper">Whisper</option>
                  <option value="Voxtral">Voxtral</option>
                </select>
              </>
            }
            manualPrompts={{
              generate: () =>
                getPolishManualPrompts({
                  audio_path: episode.audio_path!,
                  context: config.context,
                  source_lang: config.sourceLang,
                  engine,
                }),
              apply: (corrections) =>
                applyPolishManual({ audio_path: episode.audio_path!, corrections }),
              onApplied: () => {
                refreshQueries();
                setExpanded(false);
              },
            }}
          />
          <div className="px-4 pb-3 border-t border-border/50 pt-3">
            <div className="flex items-center gap-3">
              <Button
                onClick={() => skipMutation.mutate()}
                disabled={skipMutation.isPending || startMutation.isPending}
                variant="outline"
                size="sm"
              >
                <SkipForward className="w-3.5 h-3.5 mr-1.5" />
                {skipMutation.isPending ? "Copying..." : "Skip polish"}
              </Button>
              <span className="text-xs text-muted-foreground">
                Copy transcript as-is without AI correction
              </span>
            </div>
            {skipMutation.isError && (
              <p className="text-destructive text-xs mt-1">{(skipMutation.error as Error).message}</p>
            )}
          </div>
        </>
      }
    >
      {episode.polished && !taskId && (
        <SegmentEditor
          editorKey="polish"
          audioPath={episode.audio_path ?? undefined}
          episodeDuration={episode.duration}
          loadSegments={() => getPolishSegments(episode.audio_path!)}
          loadRawSegments={() => getPolishSegmentsRaw(episode.audio_path!)}
          loadVersionInfo={() => getPolishVersionInfo(episode.audio_path!)}
          saveSegments={(segs) => savePolishSegments(episode.audio_path!, segs)}
          showDelete
          showFlags={false}
          showSpeaker
          referenceSegments={referenceSegments}
          referenceLabel="Original transcript"
          speakers={showMeta?.speakers}
        />
      )}
    </PipelinePanel>
  );
}
