import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEpisodeStore, useAudioPath } from "@/stores";
import {
  deleteCorrectVersion,
  getCorrectSegments,
  getCorrectVersions,
  loadCorrectVersion,
  saveCorrectSegments,
  getSegments,
  getTranscribeVersions,
  loadTranscribeVersion,
  startCorrect,
  getCorrectManualPrompts,
  applyCorrectManual,
} from "@/api/client";
import { getCorrectFailures, dismissCorrectFailures } from "@/api/llmFailures";
import { queryKeys } from "@/api/queryKeys";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import LlmFailuresBanner from "@/components/common/LlmFailuresBanner";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import {
  useLLMConfig,
  buildLLMRequest,
  useLLMBackendStatus,
  useInputVersions,
  batchCountFor,
} from "@/hooks/useLLMPipeline";
import { modeToPreset } from "@/stores/pipelineConfigStore";
import type { LLMConfig } from "@/stores/pipelineConfigStore";
import TranscriptViewer from "@/components/editor/TranscriptViewer";
import PipelinePanel from "@/components/common/PipelinePanel";
import HelpLabel from "@/components/common/HelpLabel";
import MissingDependency from "@/components/common/MissingDependency";
import ManualModePanel from "@/components/common/ManualModePanel";
import LanguageChipRack from "@/components/common/LanguageChipRack";
import LLMControlsForm from "@/components/common/LLMControlsForm";
import PipelineRunFooter from "@/components/common/PipelineRunFooter";
import { reviewStatus } from "@/lib/stepStatus";

export default function CorrectPanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const audioPath = useAudioPath();
  const queryClient = useQueryClient();

  const task = usePipelineTask(audioPath, "correct", {
    targetStem: episode?.stem,
    optimisticPatch: () => ({ corrected: true }),
  });
  const [sourceVersionId, setSourceVersionId] = useState<string | null>(null);

  const [config, setConfig] = useLLMConfig(episode, showMeta);
  const patch = (p: Partial<LLMConfig>) => setConfig({ ...config, ...p });
  const activePreset = modeToPreset(config.mode);

  const { hasLLM, backendMissing, disabledTitle } = useLLMBackendStatus(activePreset);

  const expanded = task.expanded || !episode?.corrected;

  const { data: transcriptSegments } = useQuery({
    queryKey: queryKeys.transcribeSegments(audioPath),
    queryFn: () => getSegments(audioPath!),
    enabled: !!audioPath && !!episode?.transcribed,
  });

  const inputVersions = useInputVersions(audioPath, "correct", !!episode?.transcribed && expanded);

  const { data: correctFailures } = useQuery({
    queryKey: ["llmFailures", "correct", audioPath],
    queryFn: () => getCorrectFailures(audioPath!),
    enabled: !!audioPath && !!episode?.corrected,
  });
  const dismissFailures = useMutation({
    mutationFn: () => dismissCorrectFailures(audioPath!),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["llmFailures", "correct", audioPath] }),
  });

  // Saving a new version supersedes the recorded batch failures — offer to
  // clear them so the warning banner doesn't linger after a hand-fix.
  const handleSaved = () => {
    if (!correctFailures || correctFailures.rejected === 0) return;
    confirmDialog.open({
      title: "Clear rejected-batch records?",
      description:
        `The last correction run had ${correctFailures.rejected} rejected ` +
        "batch(es). You just saved a new version — clear those records?",
      confirmLabel: "Clear records",
      cancelLabel: "Keep",
      onConfirm: async () => { await dismissFailures.mutateAsync(); },
    });
  };

  const startMutation = useMutation({
    mutationFn: () =>
      startCorrect({
        ...buildLLMRequest(audioPath!, config),
        source_version_id: sourceVersionId ?? undefined,
      }),
    onSuccess: (data) => task.startTask(data.task_id),
  });

  if (!episode) return null;

  return (
    <PipelinePanel
      title="Correct"
      description="Use AI to fix spelling mistakes, punctuation, and other transcription errors. Runs locally or through a cloud service."
      prerequisite={!episode.transcribed ? "You need a transcript first. Go to the Transcribe tab to create one." : undefined}
      status={reviewStatus(!!episode.corrected, episode.provenance?.corrected)}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel="Re-run correction"
      settingsLabel="Correction settings"
      taskId={task.activeTaskId}
      onTaskComplete={task.handleComplete}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="No correction yet."
      controls={
        <div className="px-4 pt-3 pb-4 space-y-4">
          {!hasLLM && (
            <MissingDependency
              extra="pipeline"
              label="LLM libraries"
              description="Required for automatic AI processing. Manual mode works without them, and gives you prompts to paste into any chatbot."
            />
          )}

          <LLMControlsForm
            episode={episode}
            config={config}
            patch={patch}
            activePreset={activePreset}
            inputVersions={inputVersions}
            sourceVersionId={sourceVersionId}
            onSourceVersionChange={setSourceVersionId}
            sourceLabel="Transcript"
            sourceHelp="Which transcript version the AI should correct. Defaults to the latest."
            languageRows={
              <>
                <HelpLabel label="Language" help="The language spoken in the podcast. Helps the AI produce better corrections." />
                <LanguageChipRack value={config.sourceLang} onChange={(v) => patch({ sourceLang: v })} />
              </>
            }
            contextHelp="Describe the podcast: host names, recurring guests, technical terms, niche vocabulary. Helps the AI spell names correctly and understand jargon."
          />

          {activePreset !== "manual" && (
            <PipelineRunFooter
              onRun={() => startMutation.mutate()}
              isPending={startMutation.isPending}
              mutationError={startMutation.isError ? startMutation.error : null}
              hasExisting={episode.corrected}
              initialLabel="Correct with AI"
              rerunLabel="Re-run correction"
              disabled={backendMissing}
              disabledTitle={disabledTitle}
            />
          )}

          {activePreset === "manual" && (
            <div className="border-t border-border/50 pt-3">
              <ManualModePanel
                batchMinutes={config.batchMinutes}
                generatePrompts={(batchMinutes) =>
                  getCorrectManualPrompts({
                    audio_path: audioPath!,
                    context: config.context,
                    source_lang: config.sourceLang,
                    batch_minutes: batchMinutes,
                    batch_count: batchCountFor(episode, batchMinutes) ?? undefined,
                    source_version_id: sourceVersionId ?? undefined,
                  })
                }
                applyCorrections={(corrections) =>
                  applyCorrectManual({ audio_path: audioPath!, corrections })
                }
                onApplied={() => {
                  task.refreshQueries();
                  task.setExpanded(false);
                }}
              />
            </div>
          )}
        </div>
      }
    >
      {episode.corrected && !task.activeTaskId && correctFailures && correctFailures.rejected > 0 && (
        <div className="px-4 pt-3">
          <LlmFailuresBanner
            failures={correctFailures}
            onDismiss={() => dismissFailures.mutate()}
            dismissing={dismissFailures.isPending}
          />
        </div>
      )}
      {episode.corrected && !task.activeTaskId && (
        <TranscriptViewer
          editorKey="correct"
          audioPath={audioPath ?? undefined}
          loadSegments={() => getCorrectSegments(audioPath!)}
          saveSegments={(segs) => saveCorrectSegments(audioPath!, segs)}
          onSaved={handleSaved}
          exportSource="corrected"
          exportFilename={episode.stem ? `${episode.stem}_corrected` : undefined}
          showDelete
          showSpeaker
          referenceSegments={transcriptSegments}
          referenceLabel="Input transcript"
          speakers={showMeta?.speakers}
          loadVersions={() => getCorrectVersions(audioPath!)}
          loadCompareVersions={async () => {
            // Broader list for the compare ("vs") picker so the user can diff
            // against any earlier transcript, not just the latest. Reuse the
            // primary query's cache for the corrected slice (TranscriptViewer
            // already fetched it via loadVersions) and only hit the network for
            // transcripts. Each entry keeps its `step` so loadVersion routes
            // to the right API.
            const corrected = await queryClient.ensureQueryData({
              queryKey: queryKeys.stepVersions("correct", audioPath ?? undefined),
              queryFn: () => getCorrectVersions(audioPath!),
            });
            const transcripts = await getTranscribeVersions(audioPath!);
            return [...corrected, ...transcripts];
          }}
          loadVersion={(id, v) =>
            v?.step === "transcript"
              ? loadTranscribeVersion(audioPath!, id)
              : loadCorrectVersion(audioPath!, id)
          }
          deleteVersion={(id) => deleteCorrectVersion(audioPath!, id)}
        />
      )}
    </PipelinePanel>
  );
}
