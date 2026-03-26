import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Settings2 } from "lucide-react";
import HelpLabel from "./HelpLabel";
import ManualModePanel from "./ManualModePanel";

export const PROVIDERS = ["ollama", "openai", "anthropic", "mistral", "groq"] as const;

export type LLMMode = "ollama" | "api" | "manual";

export interface LLMConfig {
  mode: LLMMode;
  provider: string;
  model: string;
  context: string;
  sourceLang: string;
  batchSize: number;
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
    generate: () => Promise<{ batch_index: number; prompt: string; segment_count: number }[]>;
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
  const [showAdvanced, setShowAdvanced] = useState(false);
  const modes: LLMMode[] = manualPrompts ? ["ollama", "api", "manual"] : ["ollama", "api"];

  return (
    <div className="px-4 pb-3 space-y-3">
      {/* Standard parameters */}
      <div className="grid grid-cols-2 gap-3 max-w-lg text-sm">
        <HelpLabel label="Mode" help="Ollama runs locally on your machine. API calls external providers (OpenAI, Anthropic, etc.). Manual lets you copy prompts to your own LLM." />
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
                <HelpLabel label="Provider" help="Which API provider to use. Each requires its own API key set as an environment variable (e.g. OPENAI_API_KEY)." />
                <select
                  value={config.provider}
                  onChange={(e) => onChange({ provider: e.target.value })}
                  className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
                >
                  {PROVIDERS.filter((p) => p !== "ollama").map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </>
            )}

            <HelpLabel label="Model" help="Model name (e.g. llama3, gpt-4o, claude-sonnet-4-20250514). Leave empty to use the provider's default." />
            <input
              value={config.model}
              onChange={(e) => onChange({ model: e.target.value })}
              placeholder="auto from provider"
              className="input py-1 text-sm"
            />
          </>
        )}

        <HelpLabel label="Source language" help="The language of the podcast audio. Used to tailor LLM prompts for correction or translation." />
        <input
          value={config.sourceLang}
          onChange={(e) => onChange({ sourceLang: e.target.value })}
          className="input py-1 text-sm"
        />

        {extraFields}

        <HelpLabel label="Context" help="Describe the podcast, hosts, recurring topics, or proper nouns. Helps the LLM make better corrections." />
        <textarea
          value={config.context}
          onChange={(e) => onChange({ context: e.target.value })}
          placeholder="Describe the podcast, hosts, topics..."
          className="input py-1 text-sm"
          rows={2}
        />
      </div>

      {/* Advanced settings */}
      {config.mode !== "manual" && (
        <div className="border-t border-border/50 pt-3 space-y-3">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
          >
            <Settings2 className="w-3 h-3" />
            <span className="font-semibold uppercase tracking-wide">{showAdvanced ? "Hide advanced" : "Advanced settings"}</span>
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-2 gap-3 max-w-lg text-sm pl-3 border-l-2 border-border">
              <HelpLabel label="Batch size" help="Number of segments sent to the LLM per request. Smaller batches are more reliable but slower. Increase for faster processing if your model handles long context well." />
              <input
                type="number"
                value={config.batchSize}
                onChange={(e) => onChange({ batchSize: Number(e.target.value) })}
                min={1}
                className="input py-1 text-sm w-20"
              />
            </div>
          )}
        </div>
      )}

      {/* Run button */}
      {config.mode !== "manual" && (
        <div className="border-t border-border/50 pt-3">
          <Button onClick={onRun} disabled={isPending} size="sm">
            {isPending ? "Starting..." : runLabel}
          </Button>
          {error && <p className="text-destructive text-xs">{error}</p>}
        </div>
      )}

      {config.mode === "manual" && manualPrompts && (
        <ManualModePanel
          generatePrompts={manualPrompts.generate}
          applyCorrections={manualPrompts.apply}
          onApplied={manualPrompts.onApplied}
        />
      )}
    </div>
  );
}
