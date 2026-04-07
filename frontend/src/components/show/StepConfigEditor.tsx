import { useState } from "react";
import type { Episode } from "@/api/types";
import { usePipelineConfig } from "@/hooks/usePipelineConfig";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import {
  TRANSCRIBE_PRESETS,
  LLM_PRESETS,
  INDEX_PRESETS,
  usePipelineConfigStore,
} from "@/stores/pipelineConfigStore";
import { Button } from "@/components/ui/button";
import { Mic, Sparkles, Languages, Database, ChevronDown, Play, Settings2 } from "lucide-react";
import { languageToISO } from "@/lib/utils";
import type { LLMConfig } from "@/components/common/LLMControls";

export type StepKey = "transcribe" | "polish" | "translate" | "index";

function llmModeLabel(llm: LLMConfig): string {
  if (llm.mode === "manual") return "manual";
  const prefix = llm.mode === "api" ? llm.provider : "ollama";
  return `${prefix}/${llm.model || "default"}`;
}

const STEPS = [
  { key: "transcribe", label: "Transcribe", icon: Mic },
  { key: "polish", label: "Polish", icon: Sparkles },
  { key: "translate", label: "Translate", icon: Languages },
  { key: "index", label: "Index", icon: Database },
] as const;

export { STEPS };

/** Reusable preset card grid. */
function PresetCards<K extends string>({
  presets,
  active,
  onSelect,
}: {
  presets: Record<K, { label: string; desc: string }>;
  active: string;
  onSelect: (key: K) => void;
}) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {(Object.entries(presets) as [K, { label: string; desc: string }][]).map(([key, p]) => (
        <button
          key={key}
          onClick={() => onSelect(key)}
          className={`rounded-lg border p-3 text-left transition ${
            active === key
              ? "border-primary bg-accent"
              : "border-border hover:border-primary/50"
          }`}
        >
          <div className="text-sm font-medium">{p.label}</div>
          <div className="text-xs text-muted-foreground mt-0.5">{p.desc}</div>
        </button>
      ))}
    </div>
  );
}

export default function StepConfigEditor({ step, episodes, showLanguage, onRun, onClose }: {
  step: StepKey;
  episodes: Episode[];
  showLanguage: string;
  onRun: () => void;
  onClose: () => void;
}) {
  const { tc, setTc, llm, setLLM, engine, setEngine, targetLang, setTargetLang } = usePipelineConfig();
  const { whisperModels, detectedKeys: detected, apiProviders } = useLLMProviders();
  const indexModel = usePipelineConfigStore((s) => s.indexModel);
  const setIndexModel = usePipelineConfigStore((s) => s.setIndexModel);
  const transcribePreset = usePipelineConfigStore((s) => s.transcribePreset);
  const setTranscribePreset = usePipelineConfigStore((s) => s.setTranscribePreset);
  const llmPreset = usePipelineConfigStore((s) => s.llmPreset);
  const setLLMPreset = usePipelineConfigStore((s) => s.setLLMPreset);
  const indexPreset = usePipelineConfigStore((s) => s.indexPreset);
  const setIndexPreset = usePipelineConfigStore((s) => s.setIndexPreset);
  const [advanced, setAdvanced] = useState(false);

  const selectClass = "bg-secondary text-secondary-foreground rounded px-2 py-1.5 border border-border text-sm w-full";
  const inputFieldClass = "input py-1.5 text-sm w-full";

  const stepInfo = STEPS.find((s) => s.key === step)!;
  const Icon = stepInfo.icon;

  const selectTranscribePreset = (key: string) => {
    setTranscribePreset(key);
    const p = TRANSCRIBE_PRESETS[key as keyof typeof TRANSCRIBE_PRESETS];
    if (p) setTc({ modelSize: p.modelSize, diarize: p.diarize });
    setAdvanced(false);
  };

  const selectLLMPreset = (key: string) => {
    setLLMPreset(key);
    const p = LLM_PRESETS[key as keyof typeof LLM_PRESETS];
    if (p) setLLM({ mode: p.mode });
    setAdvanced(false);
  };

  const selectIndexPreset = (key: string) => {
    setIndexPreset(key);
    const p = INDEX_PRESETS[key as keyof typeof INDEX_PRESETS];
    if (p) setIndexModel(p.model);
    setAdvanced(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-background border border-border rounded-lg shadow-xl w-full max-w-md mx-4" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border">
          <Icon className="w-4 h-4" />
          <span className="text-sm font-semibold">{stepInfo.label}</span>
          <span className="text-xs text-muted-foreground">
            {episodes.length} episode{episodes.length !== 1 ? "s" : ""}
          </span>
          <div className="flex-1" />
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-lg leading-none" aria-label="Close">&times;</button>
        </div>

        <div className="px-5 py-4 space-y-4 max-h-[70vh] overflow-y-auto">
          {/* ── Preset cards ── */}
          {!advanced && (
            <>
              {step === "transcribe" && (
                <PresetCards presets={TRANSCRIBE_PRESETS} active={transcribePreset} onSelect={selectTranscribePreset} />
              )}
              {(step === "polish" || step === "translate") && (
                <PresetCards presets={LLM_PRESETS} active={llmPreset} onSelect={selectLLMPreset} />
              )}
              {step === "index" && (
                <PresetCards presets={INDEX_PRESETS} active={indexPreset} onSelect={selectIndexPreset} />
              )}
            </>
          )}

          {/* ── Quick summary (non-advanced) ── */}
          {!advanced && (
            <div className="flex flex-wrap gap-1.5">
              {step === "transcribe" && (
                <>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">{tc.modelSize}</span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">{tc.diarize ? "diarize" : "no diarize"}</span>
                </>
              )}
              {step === "polish" && (
                <>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">{engine}</span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">
                    {llmModeLabel(llm)}
                  </span>
                </>
              )}
              {step === "translate" && (
                <>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">
                    {llm.sourceLang || "auto"} &rarr; {targetLang || "?"}
                  </span>
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">
                    {llmModeLabel(llm)}
                  </span>
                </>
              )}
              {step === "index" && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground font-mono">{indexModel}</span>
              )}
            </div>
          )}

          {/* Advanced toggle */}
          <button
            onClick={() => setAdvanced(!advanced)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
          >
            {advanced ? (
              <><ChevronDown className="w-3 h-3 rotate-180 transition-transform" /> Hide advanced</>
            ) : (
              <><Settings2 className="w-3 h-3" /> Advanced settings</>
            )}
          </button>

          {/* ── Advanced settings ── */}
          {advanced && (
            <div className="space-y-4 border-t border-border pt-4">
              {step === "transcribe" && (
                <>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Model</label>
                    <select value={tc.modelSize} onChange={(e) => setTc({ modelSize: e.target.value })} className={selectClass}>
                      {Object.keys(whisperModels).length > 0
                        ? Object.entries(whisperModels).map(([key, label]) => (
                            <option key={key} value={key}>{key} &mdash; {label}</option>
                          ))
                        : <option value={tc.modelSize}>{tc.modelSize}</option>
                      }
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Language</label>
                    <div className="flex items-center gap-2 text-sm">
                      <span className="font-mono">{languageToISO(showLanguage) || "auto-detect"}</span>
                      {showLanguage && <span className="text-muted-foreground text-xs">({showLanguage})</span>}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-sm font-medium">Batch size</label>
                      <input type="number" value={tc.batchSize} onChange={(e) => setTc({ batchSize: Number(e.target.value) })} min={1} className={inputFieldClass} />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-sm font-medium">Speakers</label>
                      <input type="number" value={tc.numSpeakers} onChange={(e) => setTc({ numSpeakers: e.target.value })} placeholder="auto" min={1} className={inputFieldClass} />
                    </div>
                  </div>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" checked={tc.diarize} onChange={(e) => setTc({ diarize: e.target.checked })} className="accent-primary" />
                    <span className="text-sm">Diarize (detect speakers)</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" checked={tc.clean} onChange={(e) => setTc({ clean: e.target.checked })} className="accent-primary" />
                    <span className="text-sm">Clean transcript (remove hallucinations)</span>
                  </label>
                  {tc.diarize && (
                    <div className="space-y-1.5">
                      <label className="text-sm font-medium">HuggingFace token</label>
                      <input type="password" value={tc.hfToken} onChange={(e) => setTc({ hfToken: e.target.value })} placeholder={detected.hf_token || "from env"} className={inputFieldClass} />
                    </div>
                  )}
                </>
              )}

              {step === "polish" && (
                <>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Engine</label>
                    <select value={engine} onChange={(e) => setEngine(e.target.value)} className={selectClass}>
                      <option value="Whisper">Whisper</option>
                      <option value="Voxtral">Voxtral</option>
                    </select>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Source language</label>
                    <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className={inputFieldClass} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Context</label>
                    <textarea value={llm.context} onChange={(e) => setLLM({ context: e.target.value })} placeholder="Describe the podcast, hosts, topics..." className="input py-1.5 text-sm w-full resize-y min-h-[3rem]" />
                  </div>
                </>
              )}

              {step === "translate" && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <label className="text-sm font-medium">Source language</label>
                      <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className={inputFieldClass} />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-sm font-medium">Target language</label>
                      <input value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className={inputFieldClass} />
                    </div>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Context</label>
                    <textarea value={llm.context} onChange={(e) => setLLM({ context: e.target.value })} placeholder="Describe the podcast, hosts, topics..." className="input py-1.5 text-sm w-full resize-y min-h-[3rem]" />
                  </div>
                </>
              )}

              {step === "index" && (
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Embedding model</label>
                  <input value={indexModel} onChange={(e) => setIndexModel(e.target.value)} className={inputFieldClass} />
                </div>
              )}

              {/* LLM settings — shared by polish & translate */}
              {(step === "polish" || step === "translate") && (
                <>
                  <div className="border-t border-border pt-4">
                    <h4 className="text-sm font-semibold mb-3">LLM Settings</h4>
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Mode</label>
                    <div className="flex gap-4">
                      {(["ollama", "api", "manual"] as const).map((m) => (
                        <label key={m} className="flex items-center gap-1.5 cursor-pointer text-sm">
                          <input type="radio" checked={llm.mode === m} onChange={() => setLLM({ mode: m })} className="accent-primary" />
                          {m}
                        </label>
                      ))}
                    </div>
                  </div>
                  {llm.mode === "api" && (
                    <div className="space-y-1.5">
                      <label className="text-sm font-medium">Provider</label>
                      <select value={llm.provider} onChange={(e) => setLLM({ provider: e.target.value })} className={selectClass}>
                        {apiProviders.length > 0
                          ? apiProviders.map(([key, spec]) => (
                              <option key={key} value={key}>{spec.label}</option>
                            ))
                          : <option value={llm.provider}>{llm.provider}</option>
                        }
                      </select>
                    </div>
                  )}
                  {llm.mode !== "manual" && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <label className="text-sm font-medium">Model</label>
                        <input value={llm.model} onChange={(e) => setLLM({ model: e.target.value })} placeholder="default" className={inputFieldClass} />
                      </div>
                      <div className="space-y-1.5">
                        <label className="text-sm font-medium">Batch (min)</label>
                        <input type="number" value={llm.batchMinutes} onChange={(e) => setLLM({ batchMinutes: Number(e.target.value) })} min={1} step={5} className={inputFieldClass} />
                      </div>
                    </div>
                  )}
                  {llm.mode === "api" && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <label className="text-sm font-medium">Endpoint</label>
                        <input value={llm.apiBaseUrl} onChange={(e) => setLLM({ apiBaseUrl: e.target.value })} placeholder="default" className={inputFieldClass} />
                      </div>
                      <div className="space-y-1.5">
                        <label className="text-sm font-medium">API key</label>
                        <input type="password" value={llm.apiKey} onChange={(e) => setLLM({ apiKey: e.target.value })} placeholder={detected[llm.provider] || "from env"} className={inputFieldClass} />
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-5 py-3 border-t border-border">
          <Button onClick={onClose} variant="ghost" size="sm">Cancel</Button>
          <Button onClick={onRun} size="sm">
            <Play className="w-3.5 h-3.5 mr-1" />
            {stepInfo.label} {episodes.length} episode{episodes.length !== 1 ? "s" : ""}
          </Button>
        </div>
      </div>
    </div>
  );
}
