import { Button } from "@/components/ui/button";
import { selectClass } from "@/lib/utils";
import { useCapabilities } from "@/hooks/useCapabilities";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import FormGrid from "./FormGrid";
import HelpLabel from "./HelpLabel";
import MissingDependency from "./MissingDependency";
import ManualModePanel from "./ManualModePanel";

export type LLMMode = "api" | "ollama" | "manual";

export interface LLMConfig {
  mode: LLMMode;
  provider: string;
  model: string;
  context: string;
  sourceLang: string;
  batchMinutes: number;
  apiBaseUrl: string;
  apiKey: string;
}

interface LLMControlsProps {
  config: LLMConfig;
  onChange: (patch: Partial<LLMConfig>) => void;
  onRun: () => void;
  isPending: boolean;
  error?: string | null;
  runLabel?: string;
  extraFields?: React.ReactNode;
  manualPrompts?: {
    generate: (batchMinutes: number) => Promise<{ batch_index: number; prompt: string; segment_count: number }[]>;
    apply: (corrections: unknown[]) => Promise<unknown>;
    onApplied?: () => void;
  };
}

export default function LLMControls({
  config,
  onChange,
  onRun,
  isPending,
  error,
  runLabel = "Run",
  extraFields,
  manualPrompts,
}: LLMControlsProps) {
  const { has: hasCap } = useCapabilities();
  const hasOllama = hasCap("ollama");
  const hasOpenAI = hasCap("openai");
  const hasLLM = hasOllama || hasOpenAI;
  const modes: LLMMode[] = manualPrompts ? ["api", "ollama", "manual"] : ["api", "ollama"];

  const { apiProviders, getProviderInfo } = useLLMProviders();
  const providerInfo = getProviderInfo(config.provider);

  return (
    <div className="px-4 pb-3 space-y-3">
      {!hasLLM && (
        <MissingDependency
          extra="pipeline"
          label="LLM libraries"
          description="Required for automatic AI processing. You can also use manual mode, which gives you prompts to paste into any chatbot."
        />
      )}

      {/* ── Section 1: Mode + model ── */}
      <div className="text-sm">
        <FormGrid className="max-w-lg">
          <HelpLabel label="Mode" help="Ollama runs locally on your computer (free, needs a GPU). API calls a cloud service like OpenAI or Mistral (requires an API key). Manual generates prompts you can paste into any chatbot yourself." />
          <div className="flex gap-3">
            {modes.map((m) => (
              <label key={m} className="flex items-center gap-1 cursor-pointer">
                <input
                  type="radio"
                  checked={config.mode === m}
                  onChange={() => onChange({ mode: m })}
                  className="accent-primary"
                />
                <span>{m}</span>
              </label>
            ))}
          </div>

          {config.mode !== "api" && (
            <>
              <HelpLabel label="Model" help={config.mode === "manual" ? "Optional. Note which model you used, so you can track it later in provenance." : "The Ollama model to use (e.g. llama3, mistral). Leave empty to use the default."} />
              <input
                value={config.model}
                onChange={(e) => onChange({ model: e.target.value })}
                placeholder={config.mode === "manual" ? "e.g. ChatGPT-4o, Claude 3.5…" : "default"}
                className="input py-1 text-sm"
              />
            </>
          )}
        </FormGrid>

        {config.mode === "api" && (
          <FormGrid className="max-w-lg mt-2 pl-6 border-l-2 border-border/40">
            <HelpLabel label="Provider" help="Which cloud AI service to use. Each provider requires its own API key, set as an environment variable (e.g. OPENAI_API_KEY for OpenAI)." />
            <select
              value={config.provider}
              onChange={(e) => onChange({ provider: e.target.value, model: "" })}
              className={selectClass}
            >
              {apiProviders.length > 0
                ? apiProviders.map(([key, spec]) => (
                    <option key={key} value={key}>{spec.label}</option>
                  ))
                : <option value={config.provider}>{config.provider}</option>
              }
            </select>

            <HelpLabel label="Endpoint" help={providerInfo?.url ? "Custom API endpoint URL. Leave empty to use the provider's default endpoint." : "OpenAI-compatible API endpoint URL."} />
            <input
              value={config.apiBaseUrl}
              onChange={(e) => onChange({ apiBaseUrl: e.target.value })}
              placeholder={providerInfo?.url || "https://api.example.com/v1"}
              className="input py-1 text-sm"
            />

            <HelpLabel label="API key" help={providerInfo?.env_var ? `Leave empty to use the ${providerInfo.env_var} environment variable.` : "API key for your endpoint."} />
            <input
              type="password"
              value={config.apiKey}
              onChange={(e) => onChange({ apiKey: e.target.value })}
              placeholder={providerInfo?.env_var ? `from ${providerInfo.env_var}` : "required"}
              className="input py-1 text-sm"
            />

          </FormGrid>
        )}

        {config.mode === "api" && (
          <FormGrid className="max-w-lg mt-2">
            <HelpLabel label="Model" help="The AI model to use (e.g. gpt-4o for OpenAI, mistral-large for Mistral). Leave empty to use the provider's recommended default." />
            <input
              value={config.model}
              onChange={(e) => onChange({ model: e.target.value })}
              placeholder={providerInfo?.model || "default"}
              className="input py-1 text-sm"
            />
          </FormGrid>
        )}
      </div>

      {/* ── Section 2: Common settings ── */}
      <div className="border-t border-border/50 pt-3 text-sm space-y-3">
        <FormGrid className="max-w-lg">
          <HelpLabel label="Source language" help="The language spoken in the podcast. Helps the AI produce better corrections and translations." />
          <input
            value={config.sourceLang}
            onChange={(e) => onChange({ sourceLang: e.target.value })}
            className="input py-1 text-sm"
          />

          {extraFields}

          <HelpLabel label="Batch duration" help="Maximum audio duration per batch, in minutes. Larger batches are faster but risk exceeding the model's context window." />
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              value={config.batchMinutes}
              onChange={(e) => onChange({ batchMinutes: Number(e.target.value) })}
              min={1}
              step={5}
              className="input py-1 text-sm w-20"
            />
            <span className="text-xs text-muted-foreground">min</span>
          </div>
        </FormGrid>

        {/* Context — full width */}
        <div>
          <HelpLabel label="Context" help="Describe your podcast: host names, recurring guests, technical terms, niche vocabulary. Helps the AI spell names correctly and understand jargon." />
          <textarea
            value={config.context}
            onChange={(e) => onChange({ context: e.target.value })}
            placeholder="Describe the podcast, hosts, topics..."
            rows={Math.min(10, Math.max(3, (config.context || "").split("\n").length + 1, Math.ceil((config.context || "").length / 80)))}
            className="input py-1 text-sm resize-y w-full mt-1.5"
          />
        </div>
      </div>

      {/* ── Run / Manual ────────────────────────────────── */}
      {config.mode !== "manual" && (
        <div className="border-t border-border/50 pt-3">
          <Button onClick={onRun} disabled={isPending || !hasLLM} size="sm" title={!hasLLM ? "Install the pipeline extra to enable automatic processing" : undefined}>
            {isPending ? "Starting..." : runLabel}
          </Button>
          {error && <p className="text-destructive text-xs">{error}</p>}
        </div>
      )}

      {config.mode === "manual" && manualPrompts && (
        <div className="border-t border-border/50 pt-4">
          <ManualModePanel
            batchMinutes={config.batchMinutes}
            generatePrompts={manualPrompts.generate}
            applyCorrections={manualPrompts.apply}
            onApplied={manualPrompts.onApplied}
          />
        </div>
      )}
    </div>
  );
}
