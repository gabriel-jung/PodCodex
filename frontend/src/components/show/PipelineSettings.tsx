import { usePipelineConfig } from "@/hooks/usePipelineConfig";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { SettingRow, SettingSection } from "@/components/ui/setting-row";
import { AdvancedFieldset } from "@/components/ui/advanced-fieldset";
import { languageToISO, selectClass, inputWidth } from "@/lib/utils";

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
        <SettingRow label="Language" help="Derived from the show language above. Passed to WhisperX as an ISO code.">
          <span className="text-sm font-mono">
            {languageToISO(language) || <span className="text-muted-foreground italic">auto-detect</span>}
          </span>
        </SettingRow>
        <SettingRow label="Model" help="Speech recognition model. Larger = better quality, slower.">
          <select value={tc.modelSize} onChange={(e) => setTc({ modelSize: e.target.value })} className={selectClass}>
            {Object.keys(whisperModels).length > 0
              ? Object.entries(whisperModels).map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))
              : <option value={tc.modelSize}>{tc.modelSize}</option>
            }
          </select>
        </SettingRow>
        <SettingRow label="Diarize" help="Detect and label different speakers (requires HuggingFace token).">
          <input type="checkbox" checked={tc.diarize} onChange={(e) => setTc({ diarize: e.target.checked })} className="accent-primary" />
        </SettingRow>
        {tc.diarize && (
          <>
            <SettingRow label="HF token" help="HuggingFace access token. Needed to download the pyannote speaker model.">
              <input type="password" value={tc.hfToken} onChange={(e) => setTc({ hfToken: e.target.value })} placeholder={detected.hf_token || "from env"} className={`input ${inputWidth.short}`} />
            </SettingRow>
            <SettingRow label="Speakers" help="Expected speaker count. Empty = auto-detect.">
              <input type="number" value={tc.numSpeakers} onChange={(e) => setTc({ numSpeakers: e.target.value })} placeholder="auto" min={1} className={`input ${inputWidth.numeric}`} />
            </SettingRow>
          </>
        )}
      </SettingSection>

      {/* ── LLM Settings ── */}
      <SettingSection title="LLM" description="AI used for Correct and Translate steps.">
        <SettingRow label="Mode" help="Where the AI runs: Ollama (local) or a cloud API.">
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
          <SettingRow label="Provider" help="Cloud AI service.">
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

        <SettingRow label="Model" help="Specific model name. Empty = provider default.">
          <input value={llm.model} onChange={(e) => setLLM({ model: e.target.value })} placeholder="auto" className={`input ${inputWidth.short}`} />
        </SettingRow>

        {llm.mode === "api" && (
          <SettingRow label="API key" help="Authentication key. Empty = read from environment.">
            <input type="password" value={llm.apiKey} onChange={(e) => setLLM({ apiKey: e.target.value })} placeholder={detected[llm.provider] || "from env"} className={`input ${inputWidth.short}`} />
          </SettingRow>
        )}
      </SettingSection>

      {/* ── Translation ── */}
      <SettingSection title="Translation" description="Target language for AI translation.">
        <SettingRow label="Target language" help="Language to translate into.">
          <input value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className={`input ${inputWidth.short}`} />
        </SettingRow>
      </SettingSection>

      {/* ── Advanced ── */}
      <AdvancedFieldset
        legend="Advanced"
        description="Batch sizes, custom endpoints, and extra context. Most users leave these alone."
      >
        <SettingRow label="Batch size" help="GPU batch size for transcription. Larger = faster, more VRAM.">
          <input type="number" value={tc.batchSize} onChange={(e) => setTc({ batchSize: Number(e.target.value) })} min={1} className={`input ${inputWidth.numeric}`} />
        </SettingRow>
        {llm.mode === "api" && (
          <SettingRow label="Endpoint" help="Custom API base URL. Empty = provider default.">
            <input value={llm.apiBaseUrl} onChange={(e) => setLLM({ apiBaseUrl: e.target.value })} placeholder="default" className={`input ${inputWidth.medium}`} />
          </SettingRow>
        )}
        <SettingRow label="Batch duration" help="Max audio duration per LLM request.">
          <div className="flex items-center gap-1.5">
            <input type="number" value={llm.batchMinutes} onChange={(e) => setLLM({ batchMinutes: Number(e.target.value) })} min={1} step={5} className={`input ${inputWidth.numeric}`} />
            <span className="text-xs text-muted-foreground">min</span>
          </div>
        </SettingRow>
        <SettingRow label="Source language" help="Spoken language. Normally inferred from the show.">
          <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className={`input ${inputWidth.short}`} />
        </SettingRow>
        <SettingRow
          label="Context"
          help="Describe the show, hosts, topics. Improves AI accuracy."
          below={
            <textarea
              value={llm.context}
              onChange={(e) => setLLM({ context: e.target.value })}
              placeholder="Describe the show, hosts, recurring topics..."
              className={`input resize-y ${inputWidth.full} min-h-[4rem]`}
            />
          }
        >
          <span />
        </SettingRow>
      </AdvancedFieldset>
    </>
  );
}
