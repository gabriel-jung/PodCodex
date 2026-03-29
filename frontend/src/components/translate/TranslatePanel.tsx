import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEpisodeStore } from "@/stores";
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
import { buildDefaultContext, errorMessage, selectClass } from "@/lib/utils";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import HelpLabel from "@/components/common/HelpLabel";
import SegmentEditor from "@/components/editor/SegmentEditor";
import PipelinePanel from "@/components/common/PipelinePanel";
import LLMControls, { type LLMConfig } from "@/components/common/LLMControls";

export default function TranslatePanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  if (!episode) return null;
  const [targetLang, setTargetLang] = useState("English");
  const [editingLang, setEditingLang] = useState(episode.translations[0] || "");

  const task = usePipelineTask(episode.audio_path, "translate", {
    onComplete: () => setEditingLang(targetLang.toLowerCase().replace(/\s+/g, "_")),
  });
  const expanded = task.expanded || episode.translations.length === 0;

  const [config, setConfig] = useState<LLMConfig>({
    mode: "ollama",
    provider: "openai",
    model: "",
    context: buildDefaultContext(episode, showMeta),
    sourceLang: showMeta?.language || "French",
    batchSize: 10,
    apiBaseUrl: "",
    apiKey: "",
  });

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
        provider: config.mode === "api" && config.provider !== "custom" ? config.provider : undefined,
        model: config.model,
        context: config.context,
        source_lang: config.sourceLang,
        target_lang: targetLang,
        batch_size: config.batchSize,
        api_base_url: config.apiBaseUrl || undefined,
        api_key: config.apiKey || undefined,
      }),
    onSuccess: (data) => task.startTask(data.task_id),
  });

  const hasTranslations = (languages?.length ?? 0) > 0;

  return (
    <PipelinePanel
      title="Translate"
      description="Translate the transcript into another language using AI. If you polished the text first, the translation will use that improved version."
      prerequisite={!episode.transcribed ? "You need a transcript first. Go to the Transcribe tab to create one." : undefined}
      done={hasTranslations}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel="New translation"
      taskId={task.activeTaskId}
      onTaskComplete={task.handleComplete}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="No translations yet."
      controls={
        <>
          {hasTranslations && !expanded && (
            <div className="px-4 pb-2 flex justify-end">
              <select
                value={editingLang}
                onChange={(e) => setEditingLang(e.target.value)}
                className={selectClass}
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
              error={startMutation.isError ? errorMessage(startMutation.error) : null}
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
                generate: (batchMinutes) =>
                  getTranslateManualPrompts({
                    audio_path: episode.audio_path!,
                    context: config.context,
                    source_lang: config.sourceLang,
                    target_lang: targetLang,
                    batch_minutes: batchMinutes,
                  }),
                apply: (corrections) =>
                  applyTranslateManual({
                    audio_path: episode.audio_path!,
                    lang: targetLang,
                    corrections,
                  }),
                onApplied: () => {
                  task.refreshQueries();
                  task.setExpanded(false);
                  setEditingLang(targetLang.toLowerCase().replace(/\s+/g, "_"));
                },
              }}
            />
          )}
        </>
      }
    >
      {hasTranslations && editingLang && !task.activeTaskId && !expanded && (
        <SegmentEditor
          editorKey={`translate-${editingLang}`}
          audioPath={episode.audio_path ?? undefined}
          episodeDuration={episode.duration}
          loadSegments={() => getTranslateSegments(episode.audio_path!, editingLang)}
          loadRawSegments={() => getTranslateSegmentsRaw(episode.audio_path!, editingLang)}
          loadVersionInfo={() => getTranslateVersionInfo(episode.audio_path!, editingLang)}
          saveSegments={(segs) => saveTranslateSegments(episode.audio_path!, editingLang, segs)}
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
