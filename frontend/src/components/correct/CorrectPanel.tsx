import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEpisodeStore, useAudioPath } from "@/stores";
import {
  deleteCorrectVersion,
  getCorrectSegments,
  getCorrectVersions,
  loadCorrectVersion,
  saveCorrectSegments,
  getSegments,
  startCorrect,
  skipCorrect,
  getCorrectManualPrompts,
  applyCorrectManual,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { errorMessage } from "@/lib/utils";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { useLLMConfig, buildLLMRequest } from "@/hooks/useLLMPipeline";
import { Button } from "@/components/ui/button";
import { SkipForward } from "lucide-react";
import TranscriptViewer from "@/components/editor/TranscriptViewer";
import PipelinePanel from "@/components/common/PipelinePanel";
import LLMControls from "@/components/common/LLMControls";

export default function CorrectPanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const audioPath = useAudioPath();
  if (!episode) return null;
  const task = usePipelineTask(audioPath, "correct");
  const expanded = task.expanded || !episode.corrected;

  const [config, setConfig] = useLLMConfig(episode, showMeta);

  const { data: transcriptSegments } = useQuery({
    queryKey: queryKeys.transcribeSegments(audioPath),
    queryFn: () => getSegments(audioPath!),
    enabled: !!audioPath && episode.transcribed,
  });

  const startMutation = useMutation({
    mutationFn: () =>
      startCorrect(buildLLMRequest(audioPath!, config)),
    onSuccess: (data) => task.startTask(data.task_id),
  });

  const skipMutation = useMutation({
    mutationFn: () => skipCorrect(audioPath!),
    onSuccess: () => {
      task.refreshQueries();
      task.setExpanded(false);
    },
  });

  return (
    <PipelinePanel
      title="Correct"
      description="Use AI to fix spelling mistakes, punctuation, and other transcription errors. Can run locally or through a cloud service."
      prerequisite={!episode.transcribed ? "You need a transcript first. Go to the Transcribe tab to create one." : undefined}
      done={episode.corrected}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel="Re-run correction"
      taskId={task.activeTaskId}
      onTaskComplete={() => { task.handleComplete(); }}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="Correct pipeline not yet run for this episode."
      controls={
        <>
          <LLMControls
            config={config}
            onChange={(patch) => setConfig({ ...config, ...patch })}
            onRun={() => startMutation.mutate()}
            isPending={startMutation.isPending}
            error={startMutation.isError ? errorMessage(startMutation.error) : null}
            runLabel="Correct with AI"
            manualPrompts={{
              generate: (batchMinutes) =>
                getCorrectManualPrompts({
                  audio_path: audioPath!,
                  context: config.context,
                  source_lang: config.sourceLang,
                  batch_minutes: batchMinutes,
                }),
              apply: (corrections) =>
                applyCorrectManual({ audio_path: audioPath!, corrections }),
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
                {skipMutation.isPending ? "Copying..." : "Skip correction"}
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
      {episode.corrected && !task.activeTaskId && (
        <TranscriptViewer
          editorKey="correct"
          audioPath={audioPath ?? undefined}
          loadSegments={() => getCorrectSegments(audioPath!)}
          saveSegments={(segs) => saveCorrectSegments(audioPath!, segs)}
          exportSource="corrected"
          showDelete
          showFlags={false}
          showSpeaker
          referenceSegments={transcriptSegments}
          referenceLabel="Input transcript"
          speakers={showMeta?.speakers}
          loadVersions={() => getCorrectVersions(audioPath!)}
          loadVersion={(id) => loadCorrectVersion(audioPath!, id)}
          deleteVersion={(id) => deleteCorrectVersion(audioPath!, id)}
        />
      )}
    </PipelinePanel>
  );
}
