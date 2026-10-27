import { useId, useMemo } from "react";
import type { Episode, VersionEntry } from "@/api/types";
import type { LLMConfig, LLMPresetKey } from "@/stores/pipelineConfigStore";
import { LLM_PRESETS } from "@/stores/pipelineConfigStore";
import { formatDuration, selectClass } from "@/lib/utils";
import { modelPlaceholderFor, modelsFor } from "@/lib/providerModels";
import { useApiKeys } from "@/hooks/useApiKeys";
import { useProviderProfiles } from "@/hooks/useProviderProfiles";
import { useBatchCount } from "@/hooks/useLLMPipeline";
import { useCapabilities } from "@/hooks/useCapabilities";
import { useOllamaCheck } from "@/hooks/useOllamaCheck";
import FormGrid from "./FormGrid";
import HelpLabel from "./HelpLabel";
import OllamaStatus from "./OllamaStatus";
import Segmented from "./Segmented";
import VersionPicker from "./VersionPicker";
import { NumberInput } from "@/components/ui/number-input";

interface LLMControlsFormProps {
  /** The episode a single run targets; omitted for a batch, where the
   *  batch size is set in minutes rather than as a count per episode. */
  episode?: Episode;
  config: LLMConfig;
  patch: (p: Partial<LLMConfig>) => void;
  activePreset: LLMPresetKey;

  /** Upstream versions the user can pick as input, or undefined to hide the
   *  row (and with it the four source props). */
  inputVersions?: VersionEntry[];
  sourceVersionId?: string | null;
  onSourceVersionChange?: (id: string | null) => void;
  sourceLabel?: string;
  sourceHelp?: string;

  /** Slot for step-specific language rows (single "Language" or "From"/"To" pair). */
  languageRows: React.ReactNode;

  /** Help for the Context field, or undefined to hide it (a batch sends the
   *  show's own context, edited in the show settings). */
  contextHelp?: string;
  /** False hides the batch size row (the batch dialog's manual flow sets a
   *  count per episode instead). */
  showBatches?: boolean;
}

/**
 * Shared form body for LLM pipeline panels (correct, translate). Each panel
 * supplies its own `languageRows` slot plus the surrounding run footer and
 * manual-mode wiring.
 */
export default function LLMControlsForm({
  episode,
  config,
  patch,
  activePreset,
  inputVersions,
  sourceVersionId,
  onSourceVersionChange,
  sourceLabel,
  sourceHelp,
  languageRows,
  contextHelp,
  showBatches = true,
}: LLMControlsFormProps) {
  const { profiles, error: profilesError } = useProviderProfiles();
  const { keys, error: keysError } = useApiKeys();
  const apiProfiles = useMemo(() => profiles.filter((p) => p.type !== "ollama"), [profiles]);
  const { episodeMinutes, batchCount, setBatchCount, minutesPerBatch } = useBatchCount(episode, config, patch);
  const hasOllama = useCapabilities().has("ollama");
  const datalistId = useId();
  const modelSuggestions = activePreset === "cloud" ? modelsFor(config.providerProfile) : [];

  const isLocal = activePreset === "local";
  const ollama = useOllamaCheck(isLocal && hasOllama);

  const onPickKey = (name: string) => {
    if (!name) {
      patch({ keyName: "" });
      return;
    }
    const entry = keys.find((k) => k.name === name);
    if (entry?.suggested_provider && !config.providerProfile) {
      patch({ keyName: name, providerProfile: entry.suggested_provider });
    } else {
      patch({ keyName: name });
    }
  };

  return (
    <>
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

        {inputVersions && inputVersions.length > 0 && onSourceVersionChange && (
          <>
            <HelpLabel label={sourceLabel ?? "Source"} help={sourceHelp} />
            <VersionPicker
              versions={inputVersions}
              value={sourceVersionId ?? null}
              onChange={onSourceVersionChange}
            />
          </>
        )}

        {activePreset === "cloud" && (
          <>
            <HelpLabel label="Provider" htmlFor={`${datalistId}-0`} help="Pick a built-in profile (OpenAI, Anthropic, Mistral) or a custom one you added in Settings → Credentials. Custom profiles use OpenAI-compatible endpoints with your own base URL." />
            <select
              id={`${datalistId}-0`}
              value={config.providerProfile}
              onChange={(e) => patch({ providerProfile: e.target.value, model: "" })}
              className={`${selectClass} max-w-full min-w-0`}
            >
              <option value="">{profilesError ? "Could not load profiles" : "Pick a profile…"}</option>
              {apiProfiles.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}{p.builtin ? "" : " (custom)"}
                </option>
              ))}
            </select>

            <HelpLabel label="LLM API key" htmlFor={`${datalistId}-1`} help="Pick a key from your pool. Add or manage keys in Settings → Credentials." />
            <select
              id={`${datalistId}-1`}
              value={config.keyName}
              onChange={(e) => onPickKey(e.target.value)}
              className={`${selectClass} max-w-full min-w-0`}
            >
              <option value="">
                {keysError ? "Could not load keys" : keys.length === 0 ? "No keys, add one in Settings" : "Pick a key…"}
              </option>
              {keys.map((k) => (
                <option key={k.name} value={k.name}>
                  {k.name}
                  {k.suggested_provider ? `, ${k.suggested_provider}` : ""}
                </option>
              ))}
            </select>
          </>
        )}

        {isLocal && hasOllama && (
          <>
            <HelpLabel label="Connection" help="Status of the local Ollama daemon. The desktop app does not bundle Ollama itself; install it from ollama.com and start it before running Local mode." />
            <OllamaStatus
              data={ollama.data}
              isFetching={ollama.isFetching}
              onRefresh={() => ollama.refetch()}
            />
          </>
        )}

        <HelpLabel label="Model" htmlFor={`${datalistId}-model`} help={
          activePreset === "cloud"
            ? "The AI model to use (e.g. gpt-4o for OpenAI, mistral-large for Mistral). Leave empty for the provider's default."
            : isLocal
              ? "Pick from the models pulled into your local Ollama. If the dropdown is empty, run e.g. `ollama pull mistral` and click Refresh."
              : "Optional. Note which model you used, so you can track it later in provenance."
        } />
        {isLocal && ollama.data?.reachable && ollama.data.models.length > 0 ? (
          <select
            id={`${datalistId}-model`}
            value={config.model}
            onChange={(e) => patch({ model: e.target.value })}
            className={`${selectClass} max-w-full min-w-0`}
          >
            <option value="" disabled>Pick a model…</option>
            {!ollama.data.models.includes(config.model) && config.model && (
              <option value={config.model}>{config.model} (not pulled)</option>
            )}
            {ollama.data.models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        ) : (
          <input
            id={`${datalistId}-model`}
            value={config.model}
            onChange={(e) => patch({ model: e.target.value })}
            list={modelSuggestions.length > 0 ? datalistId : undefined}
            placeholder={
              activePreset === "cloud"
                ? modelPlaceholderFor(config.providerProfile)
                : activePreset === "manual"
                  ? "e.g. ChatGPT-4o, Claude 3.5…"
                  : "e.g. mistral, llama3"
            }
            className="input"
          />
        )}
        {modelSuggestions.length > 0 && (
          <datalist id={datalistId}>
            {modelSuggestions.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        )}

        {languageRows}

        {showBatches && <>
        <HelpLabel label="Batches" htmlFor={`${datalistId}-batches`} help={episodeMinutes
          ? "How many chunks to split the episode into. More batches keep each request small enough for the model's context window, but take longer overall."
          : "Maximum audio duration per LLM request, in minutes. Larger batches are faster but risk exceeding the model's context window."} />
        {episodeMinutes ? (
          <div className="flex items-center gap-2">
            <NumberInput
              id={`${datalistId}-batches`}
              value={batchCount}
              onChange={setBatchCount}
              min={1}
              className="input w-16"
            />
            {minutesPerBatch !== null && (
              <span className="text-xs text-muted-foreground">
                ≈ {minutesPerBatch} min each ({formatDuration(episode?.duration)} total)
              </span>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <NumberInput
              id={`${datalistId}-batches`}
              value={config.batchMinutes}
              onChange={(n) => patch({ batchMinutes: n })}
              min={1}
              step={5}
              className="input w-20"
            />
            <span className="text-xs text-muted-foreground">min per batch</span>
          </div>
        )}
        </>}
      </FormGrid>

      {contextHelp !== undefined && (
        <div className="space-y-1.5">
          <HelpLabel label="Context" htmlFor={`${datalistId}-2`} help={contextHelp} />
          <textarea
              id={`${datalistId}-2`}
            value={config.context}
            onChange={(e) => patch({ context: e.target.value })}
            placeholder="Describe the podcast, hosts, topics…"
            rows={4}
            className="input resize-y w-full"
          />
        </div>
      )}
    </>
  );
}
