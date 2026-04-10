import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEpisodeStore, useAudioPath, usePipelineConfigStore } from "@/stores";
import {
  deleteCorrectVersion,
  getCorrectSegments,
  getCorrectVersions,
  loadCorrectVersion,
  saveCorrectSegments,
  getSegments,
  startCorrect,
  getCorrectManualPrompts,
  applyCorrectManual,
} from "@/api/client";
import { getAllVersions } from "@/api/search";
import { queryKeys } from "@/api/queryKeys";
import { errorMessage } from "@/lib/utils";
import { filterVersionsForStep } from "@/lib/pipelineInputs";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { useLLMConfig, buildLLMRequest } from "@/hooks/useLLMPipeline";
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
  const [sourceVersionId, setSourceVersionId] = useState<string | null>(null);

  const [config, setConfig] = useLLMConfig(episode, showMeta);
  const llmPreset = usePipelineConfigStore((s) => s.llmPreset);
  const applyLLMPreset = usePipelineConfigStore((s) => s.applyLLMPreset);

  const { data: transcriptSegments } = useQuery({
    queryKey: queryKeys.transcribeSegments(audioPath),
    queryFn: () => getSegments(audioPath!),
    enabled: !!audioPath && episode.transcribed,
  });

  // All valid input versions (transcripts) — surfaced in the controls so the
  // user can pick which one the correction will read from (defaults to latest).
  // Uses the unified versions endpoint so the filter mirrors the batch pipeline.
  const { data: allVersions } = useQuery({
    queryKey: queryKeys.allVersions(audioPath),
    queryFn: () => getAllVersions(audioPath),
    enabled: !!audioPath && episode.transcribed,
  });
  const inputVersions = useMemo(
    () => (allVersions ? filterVersionsForStep(allVersions, "correct") : undefined),
    [allVersions],
  );

  const startMutation = useMutation({
    mutationFn: () =>
      startCorrect({
        ...buildLLMRequest(audioPath!, config),
        source_version_id: sourceVersionId ?? undefined,
      }),
    onSuccess: (data) => task.startTask(data.task_id),
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
        <LLMControls
          config={config}
          onChange={(patch) => setConfig({ ...config, ...patch })}
          onRun={() => startMutation.mutate()}
          isPending={startMutation.isPending}
          error={startMutation.isError ? errorMessage(startMutation.error) : null}
          runLabel="Correct with AI"
          preset={llmPreset}
          onPresetChange={applyLLMPreset}
          inputVersions={inputVersions}
          selectedInputVersionId={sourceVersionId}
          onInputVersionChange={setSourceVersionId}
          inputLabel="Transcript"
          manualPrompts={{
            generate: (batchMinutes) =>
              getCorrectManualPrompts({
                audio_path: audioPath!,
                context: config.context,
                source_lang: config.sourceLang,
                batch_minutes: batchMinutes,
                source_version_id: sourceVersionId ?? undefined,
              }),
            apply: (corrections) =>
              applyCorrectManual({ audio_path: audioPath!, corrections }),
            onApplied: () => {
              task.refreshQueries();
              task.setExpanded(false);
            },
          }}
        />
      }
    >
      {episode.corrected && !task.activeTaskId && (
        <TranscriptViewer
          editorKey="correct"
          audioPath={audioPath ?? undefined}
          loadSegments={() => getCorrectSegments(audioPath!)}
          saveSegments={(segs) => saveCorrectSegments(audioPath!, segs)}
          exportSource="corrected"
          exportFilename={episode.stem ? `${episode.stem}_corrected` : undefined}
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
