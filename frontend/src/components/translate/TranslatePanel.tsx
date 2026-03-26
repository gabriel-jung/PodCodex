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
import { buildDefaultContext } from "@/lib/utils";
import HelpLabel from "@/components/common/HelpLabel";
import SegmentEditor from "@/components/editor/SegmentEditor";
import PipelinePanel from "@/components/common/PipelinePanel";
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
    context: buildDefaultContext(episode, showMeta),
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

  const hasTranslations = (languages?.length ?? 0) > 0;

  return (
    <PipelinePanel
      title="Translate"
      description="Translate the transcript into another language using AI. If you polished the text first, the translation will use that improved version."
      prerequisite={!episode.transcribed ? "You need a transcript first. Go to the Transcribe tab to create one." : undefined}
      done={hasTranslations}
      expanded={expanded}
      onToggle={() => setExpanded(!expanded)}
      rerunLabel="New translation"
      taskId={taskId}
      onTaskComplete={handleComplete}
      emptyMessage="No translations yet."
      controls={
        <>
          {/* Language selector in the toggle row (only when collapsed) */}
          {hasTranslations && !expanded && (
            <div className="px-4 pb-2 flex justify-end">
              <select
                value={selectedLang}
                onChange={(e) => setSelectedLang(e.target.value)}
                className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
              >
                {languages!.map((lang) => (
                  <option key={lang} value={lang}>{lang}</option>
                ))}
              </select>
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
                  <HelpLabel label="Target language" help="The language to translate into (e.g. English, Spanish, German)." />
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
        </>
      }
    >
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
    </PipelinePanel>
  );
}
