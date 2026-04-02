import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getPipelineConfig } from "@/api/client";
import type { Episode } from "@/api/types";
import { usePipelineConfig } from "@/hooks/usePipelineConfig";
import { Button } from "@/components/ui/button";
import { Mic, Sparkles, Languages, Database } from "lucide-react";
import { languageToISO } from "@/lib/utils";

export type StepKey = "transcribe" | "polish" | "translate" | "index";

const STEPS = [
  { key: "transcribe", label: "Transcribe", icon: Mic },
  { key: "polish", label: "Polish", icon: Sparkles },
  { key: "translate", label: "Translate", icon: Languages },
  { key: "index", label: "Index", icon: Database },
] as const;

export { STEPS };

export default function StepConfigEditor({ step, episodes, showLanguage, onRun, onClose }: { step: StepKey; episodes: Episode[]; showLanguage: string; onRun: () => void; onClose: () => void }) {
  const { tc, setTc, llm, setLLM, engine, setEngine, targetLang, setTargetLang } = usePipelineConfig();

  const { data: pipelineConfig } = useQuery({
    queryKey: ["pipeline-config"],
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  const whisperModels = pipelineConfig?.whisper_models ?? {};
  const detected = pipelineConfig?.detected_keys ?? {};
  const apiProviders = pipelineConfig
    ? Object.entries(pipelineConfig.llm_providers).filter(([k]) => k !== "ollama")
    : [];

  const selClass = "bg-secondary text-secondary-foreground rounded px-2 py-1.5 border border-border text-sm w-full";
  const inputClass = "input py-1.5 text-sm w-full";

  const stepInfo = STEPS.find((s) => s.key === step)!;
  const Icon = stepInfo.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div className="bg-background border border-border rounded-lg shadow-xl w-full max-w-md mx-4" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center gap-2 px-5 py-4 border-b border-border">
          <Icon className="w-4 h-4" />
          <h3 className="text-base font-semibold">{stepInfo.label}</h3>
          <div className="flex-1" />
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-lg">×</button>
        </div>

        {/* Body */}
        <div className="px-5 py-4 space-y-4 max-h-[60vh] overflow-y-auto">
          {step === "transcribe" && (
            <>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Model</label>
                <select value={tc.modelSize} onChange={(e) => setTc({ modelSize: e.target.value })} className={selClass}>
                  {Object.keys(whisperModels).length > 0
                    ? Object.entries(whisperModels).map(([key, label]) => (
                        <option key={key} value={key}>{key} — {label}</option>
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
                  <input type="number" value={tc.batchSize} onChange={(e) => setTc({ batchSize: Number(e.target.value) })} min={1} className={inputClass} />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Speakers</label>
                  <input type="number" value={tc.numSpeakers} onChange={(e) => setTc({ numSpeakers: e.target.value })} placeholder="auto" min={1} className={inputClass} />
                </div>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={tc.diarize} onChange={(e) => setTc({ diarize: e.target.checked })} className="accent-primary" />
                <span className="text-sm">Diarize (detect speakers)</span>
              </label>
              {tc.diarize && (
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">HuggingFace token</label>
                  <input type="password" value={tc.hfToken} onChange={(e) => setTc({ hfToken: e.target.value })} placeholder={detected.hf_token || "from env"} className={inputClass} />
                </div>
              )}
            </>
          )}

          {step === "polish" && (
            <>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Engine</label>
                <select value={engine} onChange={(e) => setEngine(e.target.value)} className={selClass}>
                  <option value="Whisper">Whisper</option>
                  <option value="Voxtral">Voxtral</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Source language</label>
                <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className={inputClass} />
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
                  <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className={inputClass} />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Target language</label>
                  <input value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className={inputClass} />
                </div>
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Context</label>
                <textarea value={llm.context} onChange={(e) => setLLM({ context: e.target.value })} placeholder="Describe the podcast, hosts, topics..." className="input py-1.5 text-sm w-full resize-y min-h-[3rem]" />
              </div>
            </>
          )}

          {step === "index" && (
            <p className="text-sm text-muted-foreground">No additional configuration needed. Episodes will be indexed using default settings.</p>
          )}

          {/* LLM settings — shared by polish & translate */}
          {(step === "polish" || step === "translate") && (
            <>
              <div className="border-t border-border pt-4 mt-4">
                <h4 className="text-sm font-semibold mb-3">LLM Settings</h4>
              </div>
              <div className="space-y-1.5">
                <label className="text-sm font-medium">Mode</label>
                <div className="flex gap-4">
                  {(["ollama", "api"] as const).map((m) => (
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
                  <select value={llm.provider} onChange={(e) => setLLM({ provider: e.target.value })} className={selClass}>
                    {apiProviders.length > 0
                      ? apiProviders.map(([key, spec]) => (
                          <option key={key} value={key}>{spec.label}</option>
                        ))
                      : <option value={llm.provider}>{llm.provider}</option>
                    }
                  </select>
                </div>
              )}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Model</label>
                  <input value={llm.model} onChange={(e) => setLLM({ model: e.target.value })} placeholder="default" className={inputClass} />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Batch (min)</label>
                  <input type="number" value={llm.batchMinutes} onChange={(e) => setLLM({ batchMinutes: Number(e.target.value) })} min={1} step={5} className={inputClass} />
                </div>
              </div>
              {llm.mode === "api" && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">Endpoint</label>
                    <input value={llm.apiBaseUrl} onChange={(e) => setLLM({ apiBaseUrl: e.target.value })} placeholder="default" className={inputClass} />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium">API key</label>
                    <input type="password" value={llm.apiKey} onChange={(e) => setLLM({ apiKey: e.target.value })} placeholder={detected[llm.provider] || "from env"} className={inputClass} />
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Episodes summary */}
        <div className="px-5 py-3 border-t border-border">
          <p className="text-xs font-medium text-muted-foreground mb-1.5">{episodes.length} episode{episodes.length !== 1 ? "s" : ""} selected</p>
          <div className="max-h-24 overflow-y-auto space-y-0.5">
            {episodes.map((ep) => (
              <p key={ep.id} className="text-xs text-muted-foreground truncate">{ep.title}</p>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-5 py-4 border-t border-border">
          <Button onClick={onClose} variant="ghost" size="sm">Cancel</Button>
          <Button onClick={onRun} size="sm">{stepInfo.label} {episodes.length} episode{episodes.length !== 1 ? "s" : ""}</Button>
        </div>
      </div>
    </div>
  );
}
