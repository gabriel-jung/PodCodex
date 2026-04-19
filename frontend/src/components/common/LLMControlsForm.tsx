import type { Episode, VersionEntry } from "@/api/types";
import type { LLMConfig, LLMPresetKey } from "@/stores/pipelineConfigStore";
import { LLM_PRESETS } from "@/stores/pipelineConfigStore";
import { formatDuration, selectClass, versionOption } from "@/lib/utils";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { useBatchCount } from "@/hooks/useLLMPipeline";
import { useCapabilities } from "@/hooks/useCapabilities";
import FormGrid from "./FormGrid";
import HelpLabel from "./HelpLabel";
import Segmented from "./Segmented";
import { NumberInput } from "@/components/ui/number-input";

interface LLMControlsFormProps {
  episode: Episode;
  config: LLMConfig;
  patch: (p: Partial<LLMConfig>) => void;
  activePreset: LLMPresetKey;

  /** Upstream versions the user can pick as input, or undefined to hide the row. */
  inputVersions?: VersionEntry[];
  sourceVersionId: string | null;
  onSourceVersionChange: (id: string | null) => void;
  sourceLabel: string;
  sourceHelp: string;

  /** Slot for step-specific language rows (single "Language" or "From"/"To" pair). */
  languageRows: React.ReactNode;

  contextHelp: string;
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
}: LLMControlsFormProps) {
  const { apiProviders, getProviderInfo, detectedKeys } = useLLMProviders();
  const providerInfo = getProviderInfo(config.provider);
  const { episodeMinutes, batchCount, setBatchCount, minutesPerBatch } = useBatchCount(episode, config, patch);
  const hasOllama = useCapabilities().has("ollama");

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

        {inputVersions && inputVersions.length > 0 && (
          <>
            <HelpLabel label={sourceLabel} help={sourceHelp} />
            <select
              value={sourceVersionId ?? ""}
              onChange={(e) => onSourceVersionChange(e.target.value || null)}
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
              className="input"
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
              className="input"
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
          className="input"
        />

        {languageRows}

        <HelpLabel label="Batches" help={episodeMinutes
          ? "How many chunks to split the episode into. More batches keep each request small enough for the model's context window, but take longer overall."
          : "Maximum audio duration per LLM request, in minutes. Larger batches are faster but risk exceeding the model's context window."} />
        {episodeMinutes ? (
          <div className="flex items-center gap-2">
            <NumberInput
              value={batchCount}
              onChange={setBatchCount}
              min={1}
              max={20}
              className="input w-16"
            />
            {minutesPerBatch !== null && (
              <span className="text-xs text-muted-foreground">
                ≈ {minutesPerBatch} min each ({formatDuration(episode.duration)} total)
              </span>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <NumberInput
              value={config.batchMinutes}
              onChange={(n) => patch({ batchMinutes: n })}
              min={1}
              step={5}
              className="input w-20"
            />
            <span className="text-xs text-muted-foreground">min per batch</span>
          </div>
        )}
      </FormGrid>

      <div className="space-y-1.5">
        <HelpLabel label="Context" help={contextHelp} />
        <textarea
          value={config.context}
          onChange={(e) => patch({ context: e.target.value })}
          placeholder="Describe the podcast, hosts, topics…"
          rows={4}
          className="input resize-y w-full"
        />
      </div>
    </>
  );
}
