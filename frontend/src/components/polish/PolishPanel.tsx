import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Episode, ShowMeta } from "@/api/types";
import {
  getPolishSegments,
  getPolishSegmentsRaw,
  getPolishVersionInfo,
  savePolishSegments,
  getSegments,
  startPolish,
  getPolishManualPrompts,
  applyPolishManual,
} from "@/api/client";
import { ChevronDown, ChevronRight } from "lucide-react";
import SegmentEditor from "@/components/editor/SegmentEditor";
import ProgressBar from "@/components/editor/ProgressBar";
import LLMControls, { type LLMConfig } from "@/components/common/LLMControls";

interface PolishPanelProps {
  episode: Episode;
  showMeta?: ShowMeta | null;
}

export default function PolishPanel({ episode, showMeta }: PolishPanelProps) {
  const queryClient = useQueryClient();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(!episode.polished);

  const [config, setConfig] = useState<LLMConfig>({
    mode: "ollama",
    provider: "openai",
    model: "",
    context: "",
    sourceLang: showMeta?.language || "French",
    batchSize: 10,
  });
  const [engine, setEngine] = useState("Whisper");

  const { data: referenceSegments } = useQuery({
    queryKey: ["transcribe", "segments", episode.audio_path],
    queryFn: () => getSegments(episode.audio_path!),
    enabled: !!episode.audio_path && episode.transcribed,
  });

  const startMutation = useMutation({
    mutationFn: () =>
      startPolish({
        audio_path: episode.audio_path!,
        mode: config.mode === "api" ? "api" : "ollama",
        provider: config.mode === "api" ? config.provider : undefined,
        model: config.model,
        context: config.context,
        source_lang: config.sourceLang,
        batch_size: config.batchSize,
        engine,
      }),
    onSuccess: (data) => setTaskId(data.task_id),
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["polish"] });
    queryClient.invalidateQueries({ queryKey: ["episodes"] });
  };

  const handleComplete = () => {
    invalidate();
    setTaskId(null);
    setExpanded(false);
  };

  if (!episode.transcribed) {
    return (
      <div className="p-6 text-muted-foreground">
        Transcription required before polishing. Go to the Transcribe tab first.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Step header */}
      <div className="px-4 pt-3 pb-2 border-b border-border">
        <h3 className="text-sm font-semibold">Polish</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Fix transcription errors and improve readability using an LLM — locally with Ollama or via API.
        </p>
      </div>

      {/* Controls — collapsible when already polished */}
      {!taskId && (
        <div className="border-b border-border">
          {episode.polished ? (
            <button
              onClick={() => setExpanded(!expanded)}
              className="w-full px-4 py-2 flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition"
            >
              {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
              <span className="font-medium">Re-run polish</span>
            </button>
          ) : (
            <div className="px-4 pt-3 pb-1">
              <h4 className="text-sm font-medium">Polish settings</h4>
            </div>
          )}

          {expanded && (
            <LLMControls
              config={config}
              onChange={(patch) => setConfig({ ...config, ...patch })}
              onRun={() => startMutation.mutate()}
              isPending={startMutation.isPending}
              error={startMutation.isError ? (startMutation.error as Error).message : null}
              runLabel="Run polish"
              extraFields={
                <>
                  <label className="text-muted-foreground">Engine</label>
                  <select
                    value={engine}
                    onChange={(e) => setEngine(e.target.value)}
                    className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
                  >
                    <option value="Whisper">Whisper</option>
                    <option value="Voxtral">Voxtral</option>
                  </select>
                </>
              }
              manualPrompts={{
                generate: () =>
                  getPolishManualPrompts({
                    audio_path: episode.audio_path!,
                    context: config.context,
                    source_lang: config.sourceLang,
                    engine,
                  }),
                apply: (corrections) =>
                  applyPolishManual({ audio_path: episode.audio_path!, corrections }),
                onApplied: () => {
                  invalidate();
                  setExpanded(false);
                },
              }}
            />
          )}
        </div>
      )}

      {taskId && <ProgressBar taskId={taskId} onComplete={handleComplete} />}

      {episode.polished && !taskId && (
        <SegmentEditor
          editorKey="polish"
          audioPath={episode.audio_path ?? undefined}
          episodeDuration={episode.duration}
          loadSegments={() => getPolishSegments(episode.audio_path!)}
          loadRawSegments={() => getPolishSegmentsRaw(episode.audio_path!)}
          loadVersionInfo={() => getPolishVersionInfo(episode.audio_path!)}
          saveSegments={(segs) => savePolishSegments(episode.audio_path!, segs)}
          showDelete
          showFlags={false}
          showSpeaker
          referenceSegments={referenceSegments}
          referenceLabel="Original transcript"
          speakers={showMeta?.speakers}
        />
      )}

      {!episode.polished && !expanded && !taskId && (
        <div className="p-6 text-muted-foreground">
          Polish pipeline not yet run for this episode.
        </div>
      )}
    </div>
  );
}
