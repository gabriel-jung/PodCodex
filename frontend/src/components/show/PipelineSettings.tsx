import { usePipelineConfig } from "@/hooks/usePipelineConfig";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { SettingRow, SettingSection } from "@/components/ui/setting-row";
import { languageToISO, selectClass } from "@/lib/utils";

interface PipelineSettingsProps {
  language: string;
}

export default function PipelineSettings({ language }: PipelineSettingsProps) {
  const { tc, setTc, llm, setLLM, targetLang, setTargetLang } = usePipelineConfig();

  const { whisperModels, detectedKeys: detected, apiProviders } = useLLMProviders();

  return (
    <>
      {/* ── Transcription ── */}
      <SettingSection title="Transcription" description="Whisper model and diarization settings.">
        <SettingRow label="Language" help="Derived from the show language above. This ISO code is passed to WhisperX for transcription and alignment.">
          <span className="text-sm font-mono">
            {languageToISO(language) || <span className="text-muted-foreground italic">auto-detect</span>}
          </span>
        </SettingRow>
        <SettingRow label="Model" help="The speech recognition model. Bigger = better but slower.">
          <select value={tc.modelSize} onChange={(e) => setTc({ modelSize: e.target.value })} className={selectClass}>
            {Object.keys(whisperModels).length > 0
              ? Object.entries(whisperModels).map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))
              : <option value={tc.modelSize}>{tc.modelSize}</option>
            }
          </select>
        </SettingRow>
        <SettingRow label="Diarize" help="Detect and label different speakers.">
          <input type="checkbox" checked={tc.diarize} onChange={(e) => setTc({ diarize: e.target.checked })} className="accent-primary" />
        </SettingRow>
        {tc.diarize && (
          <>
            <SettingRow label="HF token" help="HuggingFace token for pyannote speaker model.">
              <input type="password" value={tc.hfToken} onChange={(e) => setTc({ hfToken: e.target.value })} placeholder={detected.hf_token || "from env"} className="input py-1 text-sm w-32" />
            </SettingRow>
            <SettingRow label="Speakers" help="Expected number of speakers (empty = auto).">
              <input type="number" value={tc.numSpeakers} onChange={(e) => setTc({ numSpeakers: e.target.value })} placeholder="auto" min={1} className="input py-1 text-sm w-16" />
            </SettingRow>
          </>
        )}
        <SettingRow label="Batch size" help="GPU batch size for transcription.">
          <input type="number" value={tc.batchSize} onChange={(e) => setTc({ batchSize: Number(e.target.value) })} min={1} className="input py-1 text-sm w-16" />
        </SettingRow>
      </SettingSection>

      {/* ── LLM Settings ── */}
      <SettingSection title="LLM" description="AI model configuration for Correct and Translate steps.">
        <SettingRow label="Mode" help="Ollama = local GPU. API = cloud service.">
          <div className="flex gap-3">
            {(["ollama", "api"] as const).map((m) => (
              <label key={m} className="flex items-center gap-1 cursor-pointer text-sm">
                <input type="radio" checked={llm.mode === m} onChange={() => setLLM({ mode: m })} className="accent-primary" />
                <span>{m}</span>
              </label>
            ))}
          </div>
        </SettingRow>

        {llm.mode === "api" && (
          <SettingRow label="Provider" help="Cloud AI service to use.">
            <select value={llm.provider} onChange={(e) => setLLM({ provider: e.target.value })} className={selectClass}>
              {apiProviders.length > 0
                ? apiProviders.map(([key, spec]) => (
                    <option key={key} value={key}>{spec.label}</option>
                  ))
                : <option value={llm.provider}>{llm.provider}</option>
              }
            </select>
          </SettingRow>
        )}

        <SettingRow label="Model" help="AI model name (empty = provider default).">
          <input value={llm.model} onChange={(e) => setLLM({ model: e.target.value })} placeholder="auto" className="input py-1 text-sm w-32" />
        </SettingRow>

        {llm.mode === "api" && (
          <>
            <SettingRow label="Endpoint" help="Custom API endpoint URL.">
              <input value={llm.apiBaseUrl} onChange={(e) => setLLM({ apiBaseUrl: e.target.value })} placeholder="default" className="input py-1 text-sm w-40" />
            </SettingRow>
            <SettingRow label="API key" help="Authentication key.">
              <input type="password" value={llm.apiKey} onChange={(e) => setLLM({ apiKey: e.target.value })} placeholder={detected[llm.provider] || "from env"} className="input py-1 text-sm w-32" />
            </SettingRow>
          </>
        )}

        <SettingRow label="Batch duration" help="Maximum audio duration (minutes) per LLM request.">
          <div className="flex items-center gap-1.5">
            <input type="number" value={llm.batchMinutes} onChange={(e) => setLLM({ batchMinutes: Number(e.target.value) })} min={1} step={5} className="input py-1 text-sm w-16" />
            <span className="text-xs text-muted-foreground">min</span>
          </div>
        </SettingRow>

        <SettingRow label="Source language" help="Language spoken in the podcast.">
          <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className="input py-1 text-sm w-24" />
        </SettingRow>

        <SettingRow
          label="Context"
          help="Describe the podcast, hosts, topics for better AI results."
          below={
            <textarea
              value={llm.context}
              onChange={(e) => setLLM({ context: e.target.value })}
              placeholder="Describe the podcast, hosts, topics..."
              className="input py-1 text-sm resize-y w-full min-h-[4rem]"
            />
          }
        >
          <span />
        </SettingRow>
      </SettingSection>

      {/* ── Translation ── */}
      <SettingSection title="Translation" description="Target language for AI translation.">
        <SettingRow label="Target language" help="Language to translate into.">
          <input value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className="input py-1 text-sm w-24" />
        </SettingRow>
      </SettingSection>
    </>
  );
}
