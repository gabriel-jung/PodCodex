import { useState, useEffect, useRef, useCallback } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ShowMeta } from "@/api/types";
import { updateShowMeta, syncToQdrant, getPipelineConfig } from "@/api/client";
import { usePipelineConfigStore, useConfigStore } from "@/stores";
import { Button } from "@/components/ui/button";
import { SettingRow, SettingSection } from "@/components/ui/setting-row";
import { errorMessage, selectClass } from "@/lib/utils";
import SectionHeader from "@/components/common/SectionHeader";
import ProgressBar from "@/components/editor/ProgressBar";

interface ShowSettingsProps {
  folder: string;
  meta: ShowMeta;
  hasIndex: boolean;
}

export default function ShowSettings({ folder, meta, hasIndex }: ShowSettingsProps) {
  const queryClient = useQueryClient();

  // ── Show info ──
  const [name, setName] = useState(meta.name);
  const [language, setLanguage] = useState(meta.language);
  const [rssUrl, setRssUrl] = useState(meta.rss_url);
  const [artworkUrl, setArtworkUrl] = useState(meta.artwork_url);
  const [syncTaskId, setSyncTaskId] = useState<string | null>(null);
  const [overwrite, setOverwrite] = useState(false);

  useEffect(() => {
    setName(meta.name);
    setLanguage(meta.language);
    setRssUrl(meta.rss_url);
    setArtworkUrl(meta.artwork_url);
  }, [meta]);

  const isDirty =
    name !== meta.name ||
    language !== meta.language ||
    rssUrl !== meta.rss_url ||
    artworkUrl !== meta.artwork_url;

  const saveMutation = useMutation({
    mutationFn: () =>
      updateShowMeta(folder, {
        name,
        language,
        rss_url: rssUrl,
        speakers: meta.speakers,
        artwork_url: artworkUrl,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["showMeta", folder] });
      queryClient.invalidateQueries({ queryKey: ["shows"] });
    },
  });

  const saveTimer = useRef<ReturnType<typeof setTimeout>>();
  const autoSave = useCallback(() => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => saveMutation.mutate(), 1500);
  }, [saveMutation]);

  useEffect(() => {
    if (isDirty) autoSave();
    return () => { if (saveTimer.current) clearTimeout(saveTimer.current); };
  }, [name, language, rssUrl, artworkUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  const syncMutation = useMutation({
    mutationFn: () =>
      syncToQdrant({ folder, show: meta.name || name, overwrite }),
    onSuccess: (data) => setSyncTaskId(data.task_id),
  });

  // ── Episode filters ──
  const {
    minDurationMinutes, setMinDurationMinutes,
    maxDurationMinutes, setMaxDurationMinutes,
    titleInclude, setTitleInclude,
    titleExclude, setTitleExclude,
  } = useConfigStore();

  // ── Pipeline config (shared store) ──
  const tc = usePipelineConfigStore((s) => s.transcribe);
  const setTc = usePipelineConfigStore((s) => s.setTranscribe);
  const llm = usePipelineConfigStore((s) => s.llm);
  const setLLM = usePipelineConfigStore((s) => s.setLLM);
  const engine = usePipelineConfigStore((s) => s.engine);
  const setEngine = usePipelineConfigStore((s) => s.setEngine);
  const targetLang = usePipelineConfigStore((s) => s.targetLang);
  const setTargetLang = usePipelineConfigStore((s) => s.setTargetLang);

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

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-8 max-w-2xl">
      {/* ── Show Info ── */}
      <SettingSection title="Show Info" description="Basic metadata for this podcast.">
        <SettingRow label="Name" help="Display name for this podcast.">
          <input value={name} onChange={(e) => setName(e.target.value)} className="input py-1 text-sm w-48" />
        </SettingRow>
        <SettingRow label="Language" help="Primary spoken language (e.g. French, English).">
          <input value={language} onChange={(e) => setLanguage(e.target.value)} className="input py-1 text-sm w-32" />
        </SettingRow>
        <SettingRow label="RSS URL" help="The podcast's RSS feed URL.">
          <input value={rssUrl} onChange={(e) => setRssUrl(e.target.value)} placeholder="https://..." className="input py-1 text-sm w-64" />
        </SettingRow>
        <SettingRow label="Artwork" help="URL to the podcast cover image.">
          <div className="flex items-center gap-2">
            <input value={artworkUrl} onChange={(e) => setArtworkUrl(e.target.value)} placeholder="https://..." className="input py-1 text-sm w-48" />
            {artworkUrl && (
              <img src={artworkUrl} alt="" className="w-7 h-7 rounded object-cover shrink-0" onError={(e) => (e.currentTarget.style.display = "none")} />
            )}
          </div>
        </SettingRow>
      </SettingSection>

      {/* Save status */}
      {(isDirty || saveMutation.isSuccess || saveMutation.isError) && (
        <div className="flex items-center gap-3 text-xs -mt-4">
          {isDirty && <span className="text-yellow-400">Saving...</span>}
          {saveMutation.isSuccess && !isDirty && <span className="text-green-400">Saved</span>}
          {saveMutation.isError && <span className="text-destructive">{errorMessage(saveMutation.error)}</span>}
        </div>
      )}

      {/* ── Episode Filters ── */}
      <SettingSection title="Episode Filters" description="Filter which episodes are shown in the list.">
        <SettingRow label="Min duration" help="Hide episodes shorter than this (minutes). 0 = show all.">
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              min={0}
              step={5}
              value={minDurationMinutes || ""}
              onChange={(e) => setMinDurationMinutes(Math.max(0, Number(e.target.value)))}
              placeholder="0"
              className="input w-16 py-1 text-sm text-center"
            />
            <span className="text-xs text-muted-foreground">min</span>
          </div>
        </SettingRow>
        <SettingRow label="Max duration" help="Hide episodes longer than this (minutes). 0 = no limit.">
          <div className="flex items-center gap-1.5">
            <input
              type="number"
              min={0}
              step={5}
              value={maxDurationMinutes || ""}
              onChange={(e) => setMaxDurationMinutes(Math.max(0, Number(e.target.value)))}
              placeholder="0"
              className="input w-16 py-1 text-sm text-center"
            />
            <span className="text-xs text-muted-foreground">min</span>
          </div>
        </SettingRow>
        <SettingRow label="Title contains" help="Only show episodes whose title contains this text.">
          <input
            value={titleInclude}
            onChange={(e) => setTitleInclude(e.target.value)}
            placeholder="filter..."
            className="input py-1 text-sm w-40"
          />
        </SettingRow>
        <SettingRow label="Title excludes" help="Hide episodes whose title contains this text.">
          <input
            value={titleExclude}
            onChange={(e) => setTitleExclude(e.target.value)}
            placeholder="exclude..."
            className="input py-1 text-sm w-40"
          />
        </SettingRow>
      </SettingSection>

      {/* ── Transcription ── */}
      <SettingSection title="Transcription" description="Whisper model and diarization settings for transcription.">
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
      <SettingSection title="LLM Settings" description="AI model configuration shared by Polish and Translate.">
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

        <SettingRow label="Source language" help="Language spoken in the podcast.">
          <input value={llm.sourceLang} onChange={(e) => setLLM({ sourceLang: e.target.value })} className="input py-1 text-sm w-24" />
        </SettingRow>

        <SettingRow label="Batch size" help="Segments per LLM request.">
          <input type="number" value={llm.batchSize} onChange={(e) => setLLM({ batchSize: Number(e.target.value) })} min={1} className="input py-1 text-sm w-16" />
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

      {/* ── Polish ── */}
      <SettingSection title="Polish" description="AI correction engine for transcription errors.">
        <SettingRow label="Engine" help="Which engine to use for correction.">
          <select value={engine} onChange={(e) => setEngine(e.target.value)} className={selectClass}>
            <option value="Whisper">Whisper</option>
            <option value="Voxtral">Voxtral</option>
          </select>
        </SettingRow>
      </SettingSection>

      {/* ── Translate ── */}
      <SettingSection title="Translation" description="AI translation to another language.">
        <SettingRow label="Target language" help="Language to translate into.">
          <input value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className="input py-1 text-sm w-24" />
        </SettingRow>
      </SettingSection>

      {/* ── Qdrant Sync ── */}
      {hasIndex && (
        <div className="border-t border-border pt-6 space-y-3">
          <SectionHeader>Qdrant Sync</SectionHeader>
          <p className="text-xs text-muted-foreground">
            Push indexed episodes from the local database to Qdrant for faster search across large collections.
          </p>

          {syncTaskId ? (
            <ProgressBar taskId={syncTaskId} onComplete={() => setSyncTaskId(null)} />
          ) : (
            <div className="flex items-center gap-3">
              <Button
                onClick={() => syncMutation.mutate()}
                disabled={syncMutation.isPending}
                variant="outline"
                size="sm"
              >
                {syncMutation.isPending ? "Starting..." : "Sync to Qdrant"}
              </Button>
              <label className="flex items-center gap-1.5 cursor-pointer text-xs text-muted-foreground">
                <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} className="accent-primary" />
                Overwrite existing
              </label>
              {syncMutation.isError && (
                <span className="text-xs text-destructive">{errorMessage(syncMutation.error)}</span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
