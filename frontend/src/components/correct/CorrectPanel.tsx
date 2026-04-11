import { useMemo, useState } from "react";
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
  getCorrectManualPrompts,
  applyCorrectManual,
} from "@/api/client";
import { getAllVersions } from "@/api/search";
import { queryKeys } from "@/api/queryKeys";
import { errorMessage, formatDuration, selectClass, versionOption } from "@/lib/utils";
import { filterVersionsForStep } from "@/lib/pipelineInputs";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { useLLMConfig, buildLLMRequest } from "@/hooks/useLLMPipeline";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { useCapabilities } from "@/hooks/useCapabilities";
import { LLM_PRESETS, modeToPreset } from "@/stores/pipelineConfigStore";
import type { LLMConfig } from "@/components/common/LLMControls";
import TranscriptViewer from "@/components/editor/TranscriptViewer";
import PipelinePanel from "@/components/common/PipelinePanel";
import FormGrid from "@/components/common/FormGrid";
import HelpLabel from "@/components/common/HelpLabel";
import Segmented from "@/components/common/Segmented";
import MissingDependency from "@/components/common/MissingDependency";
import ManualModePanel from "@/components/common/ManualModePanel";
import { Button } from "@/components/ui/button";

// Stored verbatim into `sourceLang` and interpolated into the LLM prompt
// template — so the chip labels are English (what the prompt expects), not
// native names like `Français`.
const TOP_LANGS: readonly string[] = ["English", "French", "German", "Spanish", "Italian"];

export default function CorrectPanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const audioPath = useAudioPath();
  if (!episode) return null;

  const task = usePipelineTask(audioPath, "correct");
  const expanded = task.expanded || !episode.corrected;
  const [sourceVersionId, setSourceVersionId] = useState<string | null>(null);

  const [config, setConfig] = useLLMConfig(episode, showMeta);
  const patch = (p: Partial<LLMConfig>) => setConfig({ ...config, ...p });
  const activePreset = modeToPreset(config.mode);

  // "Other" mode is purely derived: any non-top-chip value (including the
  // empty string set when the user clicks Other) shows the free-text input.
  const langOther = !TOP_LANGS.includes(config.sourceLang);

  // Batch count derived from episode duration — more intuitive than raw
  // minutes because the episode length is fixed. Falls back to a plain
  // minutes input when duration is unknown.
  const episodeMinutes = episode.duration ? episode.duration / 60 : null;
  const batchCount = episodeMinutes && config.batchMinutes > 0
    ? Math.max(1, Math.round(episodeMinutes / config.batchMinutes))
    : 1;
  const setBatchCount = (count: number) => {
    const n = Math.max(1, Math.min(20, Math.floor(count) || 1));
    if (episodeMinutes) patch({ batchMinutes: Math.max(1, Math.ceil(episodeMinutes / n)) });
  };
  const minutesPerBatch = episodeMinutes ? Math.ceil(episodeMinutes / batchCount) : null;

  const { has: hasCap } = useCapabilities();
  const hasOllama = hasCap("ollama");
  const hasOpenAI = hasCap("openai");
  const hasLLM = hasOllama || hasOpenAI;

  const { apiProviders, getProviderInfo, detectedKeys } = useLLMProviders();
  const providerInfo = getProviderInfo(config.provider);

  // Reference transcript for the diff view in TranscriptViewer.
  const { data: transcriptSegments } = useQuery({
    queryKey: queryKeys.transcribeSegments(audioPath),
    queryFn: () => getSegments(audioPath!),
    enabled: !!audioPath && episode.transcribed,
  });

  const { data: allVersions } = useQuery({
    queryKey: queryKeys.allVersions(audioPath),
    queryFn: () => getAllVersions(audioPath),
    enabled: !!audioPath && episode.transcribed,
  });
  const inputVersions = useMemo(
    () => (allVersions ? filterVersionsForStep(allVersions, "correct") : undefined),
    [allVersions],
  );

  // Disabled state for the Run button — the active preset's backend may not
  // be installed. Manual mode skips the button entirely (clipboard-only).
  const backendMissing =
    (activePreset === "local" && !hasOllama) ||
    (activePreset === "cloud" && !hasOpenAI);
  const disabledTitle = backendMissing
    ? activePreset === "local"
      ? "Install Ollama to run locally — or switch to Cloud/Manual"
      : "Install the openai package to use cloud providers — or switch to Manual"
    : undefined;

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
      description="Use AI to fix spelling mistakes, punctuation, and other transcription errors. Runs locally or through a cloud service."
      prerequisite={!episode.transcribed ? "You need a transcript first. Go to the Transcribe tab to create one." : undefined}
      done={episode.corrected}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel="Re-run correction"
      settingsLabel="Correction settings"
      taskId={task.activeTaskId}
      onTaskComplete={() => { task.handleComplete(); }}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="No correction yet."
      controls={
        <div className="px-4 pt-3 pb-4 space-y-4">
          {!hasLLM && (
            <MissingDependency
              extra="pipeline"
              label="LLM libraries"
              description="Required for automatic AI processing. Manual mode works without them — it gives you prompts to paste into any chatbot."
            />
          )}

          <FormGrid>
            <HelpLabel label="Mode" help="Local runs Ollama on your computer (free, needs a GPU). Cloud calls a service like OpenAI or Mistral (requires an API key). Manual gives you prompts to paste into any chatbot yourself." />
            <Segmented
              value={activePreset}
              onChange={(key) => patch({ mode: LLM_PRESETS[key].mode })}
              options={[
                ["local", LLM_PRESETS.local.label, LLM_PRESETS.local.desc, hasOllama],
                ["cloud", LLM_PRESETS.cloud.label, LLM_PRESETS.cloud.desc],
                ["manual", LLM_PRESETS.manual.label, LLM_PRESETS.manual.desc],
              ]}
            />

            {inputVersions && inputVersions.length > 0 && (
              <>
                <HelpLabel label="Transcript" help="Which transcript version the AI should correct. Defaults to the latest." />
                <select
                  value={sourceVersionId ?? ""}
                  onChange={(e) => setSourceVersionId(e.target.value || null)}
                  className={`${selectClass} text-xs max-w-full min-w-0`}
                >
                  <option value="">Latest — {versionOption(inputVersions[0])}</option>
                  {inputVersions.map((v) => (
                    <option key={v.id} value={v.id}>{versionOption(v)}</option>
                  ))}
                </select>
              </>
            )}

            {activePreset === "cloud" && (
              <>
                <HelpLabel label="Provider" help="Which cloud AI service to use. Each provider requires its own API key, set as an environment variable (e.g. OPENAI_API_KEY for OpenAI)." />
                <select
                  value={config.provider}
                  onChange={(e) => patch({ provider: e.target.value, model: "" })}
                  className={`${selectClass} max-w-full min-w-0`}
                >
                  {apiProviders.length > 0
                    ? apiProviders.map(([key, spec]) => (
                        <option key={key} value={key}>{(spec as { label: string }).label}</option>
                      ))
                    : <option value={config.provider}>{config.provider}</option>
                  }
                </select>

                <HelpLabel label="Endpoint" help="Custom OpenAI-compatible API endpoint. Leave empty to use the provider's default." />
                <input
                  value={config.apiBaseUrl}
                  onChange={(e) => patch({ apiBaseUrl: e.target.value })}
                  placeholder={providerInfo?.url || "https://api.example.com/v1"}
                  className="input py-1 text-sm"
                />

                <HelpLabel label="API key" help={providerInfo?.env_var ? `Leave empty to read from the ${providerInfo.env_var} environment variable.` : "API key for your endpoint."} />
                <input
                  type="password"
                  value={config.apiKey}
                  onChange={(e) => patch({ apiKey: e.target.value })}
                  placeholder={
                    providerInfo?.env_var && detectedKeys[config.provider]
                      ? `••• (${detectedKeys[config.provider]})`
                      : providerInfo?.env_var
                        ? `from ${providerInfo.env_var}`
                        : "required"
                  }
                  className="input py-1 text-sm"
                />
              </>
            )}

            <HelpLabel label="Model" help={
              activePreset === "cloud"
                ? "The AI model to use (e.g. gpt-4o for OpenAI, mistral-large for Mistral). Leave empty for the provider's default."
                : activePreset === "local"
                  ? "The Ollama model to use (e.g. llama3, mistral). Leave empty for the default."
                  : "Optional — note which model you used, so you can track it later in provenance."
            } />
            <input
              value={config.model}
              onChange={(e) => patch({ model: e.target.value })}
              placeholder={
                activePreset === "cloud"
                  ? (providerInfo?.model || "default")
                  : activePreset === "manual"
                    ? "e.g. ChatGPT-4o, Claude 3.5…"
                    : "default"
              }
              className="input py-1 text-sm"
            />

            <HelpLabel label="Language" help="The language spoken in the podcast. Helps the AI produce better corrections." />
            <div className="flex flex-wrap gap-1.5">
              {TOP_LANGS.map((label) => {
                const selected = !langOther && config.sourceLang === label;
                return (
                  <button
                    key={label}
                    onClick={() => patch({ sourceLang: label })}
                    className={`px-2.5 py-1 text-xs rounded-md border transition ${selected ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
                  >
                    {label}
                  </button>
                );
              })}
              <button
                onClick={() => patch({ sourceLang: "" })}
                className={`px-2.5 py-1 text-xs rounded-md border transition ${langOther ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
              >
                Other
              </button>
              {langOther && (
                <input
                  value={config.sourceLang}
                  onChange={(e) => patch({ sourceLang: e.target.value })}
                  placeholder="e.g. Japanese, Arabic…"
                  className="input py-1 text-xs w-40"
                  autoFocus
                />
              )}
            </div>

            <HelpLabel label="Batches" help={episodeMinutes
              ? "How many chunks to split the episode into. More batches keep each request small enough for the model's context window, but take longer overall."
              : "Maximum audio duration per LLM request, in minutes. Larger batches are faster but risk exceeding the model's context window."} />
            {episodeMinutes ? (
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={batchCount}
                  onChange={(e) => setBatchCount(Number(e.target.value))}
                  min={1}
                  max={20}
                  className="input py-1 text-sm w-16"
                />
                {minutesPerBatch !== null && (
                  <span className="text-xs text-muted-foreground">
                    ≈ {minutesPerBatch} min each ({formatDuration(episode.duration)} total)
                  </span>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  value={config.batchMinutes}
                  onChange={(e) => patch({ batchMinutes: Number(e.target.value) })}
                  min={1}
                  step={5}
                  className="input py-1 text-sm w-20"
                />
                <span className="text-xs text-muted-foreground">min per batch</span>
              </div>
            )}
          </FormGrid>

          <div className="space-y-1.5">
            <HelpLabel label="Context" help="Describe the podcast: host names, recurring guests, technical terms, niche vocabulary. Helps the AI spell names correctly and understand jargon." />
            <textarea
              value={config.context}
              onChange={(e) => patch({ context: e.target.value })}
              placeholder="Describe the podcast, hosts, topics…"
              rows={4}
              className="input py-1 text-sm resize-y w-full"
            />
          </div>

          {activePreset !== "manual" && (
            <div className="flex items-baseline gap-3 flex-wrap pt-1">
              <Button
                onClick={() => startMutation.mutate()}
                disabled={startMutation.isPending || backendMissing}
                size="sm"
                title={disabledTitle}
              >
                {startMutation.isPending
                  ? "Starting…"
                  : episode.corrected
                    ? "Re-run correction"
                    : "Correct with AI"}
              </Button>
              {episode.corrected && (
                <span className="text-xs text-muted-foreground">Saves a new version — previous ones stay in History.</span>
              )}
              {startMutation.isError && (
                <p className="text-destructive text-xs w-full">{errorMessage(startMutation.error)}</p>
              )}
            </div>
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
