import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { getPipelineConfig } from "@/api/client";
import { selectClass } from "@/lib/utils";
import { useCapabilities } from "@/hooks/useCapabilities";
import AdvancedToggle from "./AdvancedToggle";
import HelpLabel from "./HelpLabel";
import MissingDependency from "./MissingDependency";
import ManualModePanel from "./ManualModePanel";

export type LLMMode = "ollama" | "api" | "manual";

export interface LLMConfig {
  mode: LLMMode;
  provider: string;
  model: string;
  context: string;
  sourceLang: string;
  batchSize: number;
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
  const modes: LLMMode[] = manualPrompts ? ["ollama", "api", "manual"] : ["ollama", "api"];

  const { data: pipelineConfig } = useQuery({
    queryKey: ["pipeline-config"],
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  const apiProviders = pipelineConfig
    ? Object.entries(pipelineConfig.llm_providers).filter(([k]) => k !== "ollama")
    : [];

  // Current provider info (env var name, default model, etc.)
  const providerInfo = pipelineConfig?.llm_providers[config.provider] as
    | { url?: string; model?: string; label?: string; env_var?: string }
    | undefined;

  return (
    <div className="px-4 pb-3 space-y-3">
      {!hasLLM && (
        <MissingDependency
          extra="pipeline"
          label="LLM libraries"
          description="Required for automatic AI processing. You can use manual mode instead — it gives you prompts to paste into any chatbot."
        />
      )}

      {/* Standard parameters — 2-column layout on wide screens */}
      <div className="flex flex-col lg:flex-row gap-6 text-sm">
        {/* Left: form fields */}
        <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 max-w-md items-start sm:items-center">
          <HelpLabel label="Mode" help="Ollama = runs on your own computer (free, needs a GPU). API = uses a cloud service (OpenAI, Mistral, etc. — requires an API key). Manual = shows you the prompts so you can paste them into any chatbot yourself." />
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

          {config.mode !== "manual" && (
            <>
              {config.mode === "api" && (
                <>
                  <HelpLabel label="Provider" help="Which cloud AI service to use. You need an API key for the chosen provider — set it as an environment variable (e.g. OPENAI_API_KEY for OpenAI)." />
                  <select
                    value={config.provider}
                    onChange={(e) => onChange({ provider: e.target.value })}
                    className={selectClass}
                  >
                    {apiProviders.length > 0
                      ? apiProviders.map(([key, spec]) => (
                          <option key={key} value={key}>{spec.label}</option>
                        ))
                      : <option value={config.provider}>{config.provider}</option>
                    }
                  </select>
                </>
              )}

              <HelpLabel label="Model" help="The AI model to use (e.g. llama3 for Ollama, gpt-4o for OpenAI). Leave empty to use the provider's recommended default." />
              <input
                value={config.model}
                onChange={(e) => onChange({ model: e.target.value })}
                placeholder="auto from provider"
                className="input py-1 text-sm"
              />

              {config.mode === "api" && (
                <>
                  <HelpLabel label="Endpoint" help={providerInfo?.url ? "Custom API endpoint URL. Leave empty to use the provider's default endpoint. Useful for proxies." : "OpenAI-compatible API endpoint URL (required)."} />
                  <input
                    value={config.apiBaseUrl}
                    onChange={(e) => onChange({ apiBaseUrl: e.target.value })}
                    placeholder={providerInfo?.url || "https://api.example.com/v1"}
                    className="input py-1 text-sm"
                  />

                  <HelpLabel label="API key" help={providerInfo?.env_var ? `Authentication key. Leave empty to use the ${providerInfo.env_var} environment variable.` : "API key for your endpoint."} />
                  <input
                    type="password"
                    value={config.apiKey}
                    onChange={(e) => onChange({ apiKey: e.target.value })}
                    placeholder={providerInfo?.env_var ? `from ${providerInfo.env_var}` : "required"}
                    className="input py-1 text-sm"
                  />
                </>
              )}
            </>
          )}

          <HelpLabel label="Source language" help="The language spoken in the podcast. This helps the AI understand context and produce better corrections or translations." />
          <input
            value={config.sourceLang}
            onChange={(e) => onChange({ sourceLang: e.target.value })}
            className="input py-1 text-sm"
          />

          {extraFields}
        </div>

        {/* Right: context textarea */}
        <div className="flex flex-col gap-1.5 lg:flex-1 lg:border-l lg:border-border lg:pl-6">
          <HelpLabel label="Context" help="Tell the AI about your podcast: host names, recurring guests, technical terms, or niche vocabulary. This helps it spell names correctly and understand jargon." />
          <textarea
            value={config.context}
            onChange={(e) => onChange({ context: e.target.value })}
            placeholder="Describe the podcast, hosts, topics..."
            className="input py-1 text-sm resize-y flex-1 min-h-[5rem]"
          />
        </div>
      </div>

      {/* Advanced settings */}
      {config.mode !== "manual" && (
        <AdvancedToggle className="border-t border-border/50 pt-3 space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm pl-3 border-l-2 border-border">
            <HelpLabel label="Batch size" help="How many segments are processed at once. Smaller values (5-10) are safer but slower. Larger values (20-50) are faster but may fail if the AI can't handle that much text." />
            <input
              type="number"
              value={config.batchSize}
              onChange={(e) => onChange({ batchSize: Number(e.target.value) })}
              min={1}
              className="input py-1 text-sm w-20"
            />
          </div>
        </AdvancedToggle>
      )}

      {/* Run button */}
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
            generatePrompts={manualPrompts.generate}
            applyCorrections={manualPrompts.apply}
            onApplied={manualPrompts.onApplied}
          />
        </div>
      )}
    </div>
  );
}
