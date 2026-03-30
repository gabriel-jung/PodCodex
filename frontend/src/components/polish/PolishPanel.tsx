import { useMutation, useQuery } from "@tanstack/react-query";
import { useEpisodeStore, usePipelineConfigStore } from "@/stores";
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
import { errorMessage, selectClass } from "@/lib/utils";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { useLLMConfig } from "@/hooks/useLLMPipeline";
import { Button } from "@/components/ui/button";
import { SkipForward } from "lucide-react";
import SegmentEditor from "@/components/editor/SegmentEditor";
import PipelinePanel from "@/components/common/PipelinePanel";
import HelpLabel from "@/components/common/HelpLabel";
import LLMControls from "@/components/common/LLMControls";

export default function PolishPanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  if (!episode) return null;
  const task = usePipelineTask(episode.audio_path, "polish");
  const expanded = task.expanded || !episode.polished;

  const [config, setConfig] = useLLMConfig(episode, showMeta);
  const engine = usePipelineConfigStore((s) => s.engine);
  const setEngine = usePipelineConfigStore((s) => s.setEngine);

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
        provider: config.mode === "api" && config.provider !== "custom" ? config.provider : undefined,
        model: config.model,
        context: config.context,
        source_lang: config.sourceLang,
        batch_minutes: config.batchMinutes,
        engine,
        api_base_url: config.apiBaseUrl || undefined,
        api_key: config.apiKey || undefined,
      }),
    onSuccess: (data) => task.startTask(data.task_id),
  });

  const skipMutation = useMutation({
    mutationFn: () => skipPolish(episode.audio_path!),
    onSuccess: () => {
      task.refreshQueries();
      task.setExpanded(false);
    },
  });

  return (
    <PipelinePanel
      title="Polish"
      description="Use AI to fix spelling mistakes, punctuation, and other transcription errors. Can run locally or through a cloud service."
      prerequisite={!episode.transcribed ? "You need a transcript first. Go to the Transcribe tab to create one." : undefined}
      done={episode.polished}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel="Re-run polish"
      taskId={task.activeTaskId}
      onTaskComplete={task.handleComplete}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="Polish pipeline not yet run for this episode."
      controls={
        <>
          <LLMControls
            config={config}
            onChange={(patch) => setConfig({ ...config, ...patch })}
            onRun={() => startMutation.mutate()}
            isPending={startMutation.isPending}
            error={startMutation.isError ? errorMessage(startMutation.error) : null}
            runLabel="Run polish"
            extraFields={
              <>
                <HelpLabel label="Engine" help="Which transcription engine produced the text. Affects the correction prompt — Whisper and Voxtral make different kinds of errors." />
                <select
                  value={engine}
                  onChange={(e) => setEngine(e.target.value)}
                  className={selectClass}
                >
                  <option value="Whisper">Whisper</option>
                  <option value="Voxtral">Voxtral</option>
                </select>
              </>
            }
            manualPrompts={{
              generate: (batchMinutes) =>
                getPolishManualPrompts({
                  audio_path: episode.audio_path!,
                  context: config.context,
                  source_lang: config.sourceLang,
                  engine,
                  batch_minutes: batchMinutes,
                }),
              apply: (corrections) =>
                applyPolishManual({ audio_path: episode.audio_path!, corrections }),
              onApplied: () => {
                task.refreshQueries();
                task.setExpanded(false);
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
              <p className="text-destructive text-xs mt-1">{errorMessage(skipMutation.error)}</p>
            )}
          </div>
        </>
      }
    >
      {episode.polished && !task.activeTaskId && (
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
