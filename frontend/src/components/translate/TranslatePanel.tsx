import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEpisodeStore, useAudioPath, usePipelineConfigStore } from "@/stores";
import {
  deleteTranslateVersion,
  getTranslateSegments,
  getTranslateVersions,
  loadTranslateVersion,
  saveTranslateSegments,
  startTranslate,
  getTranslateManualPrompts,
  applyTranslateManual,
} from "@/api/client";
import { getAllVersions } from "@/api/search";
import { queryKeys } from "@/api/queryKeys";
import { selectClass } from "@/lib/utils";
import { filterVersionsForStep } from "@/lib/pipelineInputs";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import {
  useLLMConfig,
  buildLLMRequest,
  useBestSourceSegments,
  useLLMBackendStatus,
} from "@/hooks/useLLMPipeline";
import { useCapabilities } from "@/hooks/useCapabilities";
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

// Backend identifies translations by a normalized filesystem-safe key.
const langKey = (lang: string) => lang.toLowerCase().replace(/\s+/g, "_");

export default function TranslatePanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const audioPath = useAudioPath();
  if (!episode) return null;

  const targetLang = usePipelineConfigStore((s) => s.targetLang);
  const setTargetLang = usePipelineConfigStore((s) => s.setTargetLang);
  const [editingLang, setEditingLang] = useState(episode.translations[0] || "");
  const [sourceVersionId, setSourceVersionId] = useState<string | null>(null);

  const task = usePipelineTask(audioPath, "translate", {
    onComplete: () => setEditingLang(langKey(targetLang)),
  });
  const expanded = task.expanded || episode.translations.length === 0;

  const [config, setConfig] = useLLMConfig(episode, showMeta);
  const patch = (p: Partial<LLMConfig>) => setConfig({ ...config, ...p });
  const activePreset = modeToPreset(config.mode);

  const { has: hasCap } = useCapabilities();
  const hasLLM = hasCap("ollama") || hasCap("openai");
  const { hasOllama, backendMissing, disabledTitle } = useLLMBackendStatus(activePreset);

  const { data: referenceSegments } = useBestSourceSegments(
    audioPath,
    { enabled: episode.transcribed, corrected: episode.corrected },
  );

  // User can pick any corrected OR transcript version as input —
  // filterVersionsForStep mirrors the batch pipeline's rules.
  const { data: allVersions } = useQuery({
    queryKey: queryKeys.allVersions(audioPath),
    queryFn: () => getAllVersions(audioPath),
    enabled: !!audioPath && episode.transcribed && expanded,
  });
  const inputVersions = useMemo(
    () => (allVersions ? filterVersionsForStep(allVersions, "translate") : undefined),
    [allVersions],
  );

  const startMutation = useMutation({
    mutationFn: () =>
      startTranslate({
        ...buildLLMRequest(audioPath!, config),
        target_lang: targetLang,
        source_version_id: sourceVersionId ?? undefined,
      }),
    onSuccess: (data) => task.startTask(data.task_id),
  });

  const hasTranslations = episode.translations.length > 0;
  const missingTarget = !targetLang.trim();
  const runDisabled = backendMissing || missingTarget;
  const runDisabledTitle = disabledTitle || (missingTarget ? "Pick a target language first" : undefined);

  return (
    <PipelinePanel
      title="Translate"
      description="Translate the transcript into another language using AI. If you corrected the text first, the translation will use that improved version."
      prerequisite={!episode.transcribed ? "You need a transcript first. Go to the Transcribe tab to create one." : undefined}
      done={hasTranslations}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel="New translation"
      settingsLabel="Translation settings"
      taskId={task.activeTaskId}
      onTaskComplete={task.handleComplete}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="No translations yet."
      controls={
        <>
          {hasTranslations && !expanded && (
            <div className="px-4 pb-2 flex justify-end">
              <select
                value={editingLang}
                onChange={(e) => setEditingLang(e.target.value)}
                className={selectClass}
              >
                {episode.translations.map((lang) => (
                  <option key={lang} value={lang}>{lang}</option>
                ))}
              </select>
            </div>
          )}
          {expanded && (
            <div className="px-4 pt-3 pb-4 space-y-4">
              {!hasLLM && (
                <MissingDependency
                  extra="pipeline"
                  label="LLM libraries"
                  description="Required for automatic AI processing. Manual mode works without them — it gives you prompts to paste into any chatbot."
                />
              )}

              <LLMControlsForm
                episode={episode}
                config={config}
                patch={patch}
                activePreset={activePreset}
                hasOllama={hasOllama}
                inputVersions={inputVersions}
                sourceVersionId={sourceVersionId}
                onSourceVersionChange={setSourceVersionId}
                sourceLabel="Source text"
                sourceHelp="Which version the AI should translate from. Defaults to the latest corrected version, or the transcript if no corrections exist."
                languageRows={
                  <>
                    <HelpLabel label="From" help="The language spoken in the podcast." />
                    <LanguageChipRack value={config.sourceLang} onChange={(v) => patch({ sourceLang: v })} />
                    <HelpLabel label="To" help="The language to translate into." />
                    <LanguageChipRack value={targetLang} onChange={setTargetLang} />
                  </>
                }
                contextHelp="Describe the podcast: host names, recurring guests, technical terms, niche vocabulary. Helps the AI produce better translations."
              />

              {activePreset !== "manual" && (
                <PipelineRunFooter
                  onRun={() => startMutation.mutate()}
                  isPending={startMutation.isPending}
                  mutationError={startMutation.isError ? startMutation.error : null}
                  hasExisting={hasTranslations}
                  initialLabel="Translate with AI"
                  rerunLabel="Run translation"
                  disabled={runDisabled}
                  disabledTitle={runDisabledTitle}
                />
              )}

              {activePreset === "manual" && (
                <div className="border-t border-border/50 pt-3">
                  <ManualModePanel
                    batchMinutes={config.batchMinutes}
                    generatePrompts={(batchMinutes) =>
                      getTranslateManualPrompts({
                        audio_path: audioPath!,
                        context: config.context,
                        source_lang: config.sourceLang,
                        target_lang: targetLang,
                        batch_minutes: batchMinutes,
                        source_version_id: sourceVersionId ?? undefined,
                      })
                    }
                    applyCorrections={(corrections) =>
                      applyTranslateManual({
                        audio_path: audioPath!,
                        lang: targetLang,
                        corrections,
                      })
                    }
                    onApplied={() => {
                      task.refreshQueries();
                      task.setExpanded(false);
                      setEditingLang(langKey(targetLang));
                    }}
                  />
                </div>
              )}
            </div>
          )}
        </>
      }
    >
      {hasTranslations && editingLang && !task.activeTaskId && !expanded && (
        <TranscriptViewer
          editorKey={`translate-${editingLang}`}
          audioPath={audioPath ?? undefined}
          loadSegments={() => getTranslateSegments(audioPath!, editingLang)}
          saveSegments={(segs) => saveTranslateSegments(audioPath!, editingLang, segs)}
          exportSource={`translated_${editingLang}`}
          exportFilename={episode.stem ? `${episode.stem}_${editingLang}` : undefined}
          showDelete
          showFlags={false}
          showSpeaker
          referenceSegments={referenceSegments}
          referenceLabel="Source text"
          speakers={showMeta?.speakers}
          loadVersions={() => getTranslateVersions(audioPath!, editingLang)}
          loadVersion={(id) => loadTranslateVersion(audioPath!, editingLang, id)}
          deleteVersion={(id) => deleteTranslateVersion(audioPath!, editingLang, id)}
        />
      )}
    </PipelinePanel>
  );
}
