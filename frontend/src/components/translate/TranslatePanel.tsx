import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Episode, ShowMeta } from "@/api/types";
import {
  getTranslateSegments,
  getTranslateSegmentsRaw,
  getTranslateVersionInfo,
  saveTranslateSegments,
  getTranslateLanguages,
  getPolishSegments,
  getSegments,
  startTranslate,
  getTranslateManualPrompts,
  applyTranslateManual,
} from "@/api/client";
import { ChevronDown, ChevronRight } from "lucide-react";
import SegmentEditor from "@/components/editor/SegmentEditor";
import ProgressBar from "@/components/editor/ProgressBar";
import LLMControls, { type LLMConfig } from "@/components/common/LLMControls";

interface TranslatePanelProps {
  episode: Episode;
  showMeta?: ShowMeta | null;
}

export default function TranslatePanel({ episode, showMeta }: TranslatePanelProps) {
  const queryClient = useQueryClient();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(episode.translations.length === 0);

  const [config, setConfig] = useState<LLMConfig>({
    mode: "ollama",
    provider: "openai",
    model: "",
    context: "",
    sourceLang: showMeta?.language || "French",
    batchSize: 10,
  });
  const [targetLang, setTargetLang] = useState("English");
  const [selectedLang, setSelectedLang] = useState(episode.translations[0] || "");

  const { data: languages } = useQuery({
    queryKey: ["translate", "languages", episode.audio_path],
    queryFn: () => getTranslateLanguages(episode.audio_path!),
    enabled: !!episode.audio_path,
  });

  const { data: referenceSegments } = useQuery({
    queryKey: ["source-for-translate", episode.audio_path],
    queryFn: async () => {
      if (!episode.audio_path) return [];
      try {
        if (episode.polished) return await getPolishSegments(episode.audio_path);
      } catch { /* fall through */ }
      return getSegments(episode.audio_path);
    },
    enabled: !!episode.audio_path && episode.transcribed,
  });

  const startMutation = useMutation({
    mutationFn: () =>
      startTranslate({
        audio_path: episode.audio_path!,
        mode: config.mode === "api" ? "api" : "ollama",
        provider: config.mode === "api" ? config.provider : undefined,
        model: config.model,
        context: config.context,
        source_lang: config.sourceLang,
        target_lang: targetLang,
        batch_size: config.batchSize,
      }),
    onSuccess: (data) => setTaskId(data.task_id),
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["translate"] });
    queryClient.invalidateQueries({ queryKey: ["episodes"] });
  };

  const handleComplete = () => {
    invalidate();
    setTaskId(null);
    setExpanded(false);
    setSelectedLang(targetLang.toLowerCase().replace(/\s+/g, "_"));
  };

  if (!episode.transcribed) {
    return (
      <div className="p-6 text-muted-foreground">
        Transcription required before translation. Go to the Transcribe tab first.
      </div>
    );
  }

  const hasTranslations = (languages?.length ?? 0) > 0;

  return (
    <div className="flex flex-col h-full">
      {/* Step header */}
      <div className="px-4 pt-3 pb-2 border-b border-border">
        <h3 className="text-sm font-semibold">Translate</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Translate segments to another language using an LLM. Uses polished text when available, otherwise the raw transcript.
        </p>
      </div>

      {/* Controls — collapsible when translations exist */}
      {!taskId && (
        <div className="border-b border-border">
          {hasTranslations ? (
            <div className="flex items-center">
              <button
                onClick={() => setExpanded(!expanded)}
                className="px-4 py-2 flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition"
              >
                {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                <span className="font-medium">New translation</span>
              </button>
              {!expanded && (
                <select
                  value={selectedLang}
                  onChange={(e) => setSelectedLang(e.target.value)}
                  className="ml-auto mr-4 bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
                >
                  {languages!.map((lang) => (
                    <option key={lang} value={lang}>{lang}</option>
                  ))}
                </select>
              )}
            </div>
          ) : (
            <div className="px-4 pt-3 pb-1">
              <h4 className="text-sm font-medium">Translation settings</h4>
            </div>
          )}

          {expanded && (
            <LLMControls
              config={config}
              onChange={(patch) => setConfig({ ...config, ...patch })}
              onRun={() => startMutation.mutate()}
              isPending={startMutation.isPending}
              error={startMutation.isError ? (startMutation.error as Error).message : null}
              runLabel="Run translation"
              extraFields={
                <>
                  <label className="text-muted-foreground">Target language</label>
                  <input
                    value={targetLang}
                    onChange={(e) => setTargetLang(e.target.value)}
                    className="input py-1 text-sm"
                  />
                </>
              }
              manualPrompts={{
                generate: () =>
                  getTranslateManualPrompts({
                    audio_path: episode.audio_path!,
                    context: config.context,
                    source_lang: config.sourceLang,
                    target_lang: targetLang,
                  }),
                apply: (corrections) =>
                  applyTranslateManual({
                    audio_path: episode.audio_path!,
                    lang: targetLang,
                    corrections,
                  }),
                onApplied: () => {
                  invalidate();
                  setExpanded(false);
                  setSelectedLang(targetLang.toLowerCase().replace(/\s+/g, "_"));
                },
              }}
            />
          )}
        </div>
      )}

      {taskId && <ProgressBar taskId={taskId} onComplete={handleComplete} />}

      {hasTranslations && selectedLang && !taskId && !expanded && (
        <SegmentEditor
          editorKey={`translate-${selectedLang}`}
          audioPath={episode.audio_path ?? undefined}
          episodeDuration={episode.duration}
          loadSegments={() => getTranslateSegments(episode.audio_path!, selectedLang)}
          loadRawSegments={() => getTranslateSegmentsRaw(episode.audio_path!, selectedLang)}
          loadVersionInfo={() => getTranslateVersionInfo(episode.audio_path!, selectedLang)}
          saveSegments={(segs) => saveTranslateSegments(episode.audio_path!, selectedLang, segs)}
          showDelete
          showFlags={false}
          showSpeaker
          referenceSegments={referenceSegments}
          referenceLabel="Source text"
          speakers={showMeta?.speakers}
        />
      )}

      {!hasTranslations && !expanded && !taskId && (
        <div className="p-6 text-muted-foreground">
          No translations yet.
        </div>
      )}
    </div>
  );
}
