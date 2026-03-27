import { useState, useCallback } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Episode, ShowMeta } from "@/api/types";
import {
  getSynthesisStatus,
  startExtractVoices,
  getVoiceSamples,
  startGenerateTTS,
  getGeneratedSegments,
  assembleEpisode,
  audioFileUrl,
  getPipelineConfig,
} from "@/api/client";
import { Button } from "@/components/ui/button";
import { Settings2 } from "lucide-react";
import HelpLabel from "@/components/common/HelpLabel";
import ProgressBar from "@/components/editor/ProgressBar";
import PipelinePanel from "@/components/common/PipelinePanel";
import { useAppStore } from "@/store";

interface SynthesizePanelProps {
  episode: Episode;
  showMeta?: ShowMeta | null;
}

export default function SynthesizePanel({ episode, showMeta }: SynthesizePanelProps) {
  const queryClient = useQueryClient();
  const { playAudio } = useAppStore();

  const [extractTaskId, setExtractTaskId] = useState<string | null>(null);
  const [generateTaskId, setGenerateTaskId] = useState<string | null>(null);
  const [language, setLanguage] = useState(showMeta?.language || "English");
  const [modelSize, setModelSize] = useState("1.7B");
  const [sourceLang, setSourceLang] = useState("");
  const [maxChunkDuration, setMaxChunkDuration] = useState(20);
  const [assembleStrategy, setAssembleStrategy] = useState("original_timing");
  const [expanded, setExpanded] = useState(!episode.synthesized);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { data: pipelineConfig } = useQuery({
    queryKey: ["pipeline-config"],
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ["synthesize", "status", episode.audio_path],
    queryFn: () => getSynthesisStatus(episode.audio_path!),
    enabled: !!episode.audio_path,
  });

  const { data: voiceSamples } = useQuery({
    queryKey: ["synthesize", "voices", episode.audio_path],
    queryFn: () => getVoiceSamples(episode.audio_path!),
    enabled: !!episode.audio_path && !!status?.voice_samples_extracted,
  });

  const { data: generatedSegments } = useQuery({
    queryKey: ["synthesize", "generated", episode.audio_path],
    queryFn: () => getGeneratedSegments(episode.audio_path!),
    enabled: !!episode.audio_path && !!status?.tts_segments_generated,
  });

  const refreshQueries =useCallback(() => {
    refetchStatus();
    queryClient.invalidateQueries({ queryKey: ["synthesize"] });
    queryClient.invalidateQueries({ queryKey: ["episodes"] });
  }, [queryClient, refetchStatus]);

  const extractMutation = useMutation({
    mutationFn: () =>
      startExtractVoices({ audio_path: episode.audio_path! }),
    onSuccess: (data) => setExtractTaskId(data.task_id),
  });

  const generateMutation = useMutation({
    mutationFn: () =>
      startGenerateTTS({
        audio_path: episode.audio_path!,
        model_size: modelSize,
        language,
        source_lang: sourceLang || undefined,
        max_chunk_duration: maxChunkDuration,
      }),
    onSuccess: (data) => setGenerateTaskId(data.task_id),
  });

  const assembleMutation = useMutation({
    mutationFn: () =>
      assembleEpisode({
        audio_path: episode.audio_path!,
        strategy: assembleStrategy,
      }),
    onSuccess: () => {
      refreshQueries();
      setExpanded(false);
    },
  });

  const prereq = !episode.audio_path
    ? "Download the audio file first before synthesizing."
    : !episode.transcribed
      ? "You need a transcript first. Go to the Transcribe tab to create one."
      : undefined;

  const isRunning = !!extractTaskId || !!generateTaskId;

  return (
    <PipelinePanel
      title="Synthesize"
      description="Re-create the episode with cloned voices — extract voice samples, generate speech for each segment, then assemble the final audio."
      prerequisite={prereq}
      done={episode.synthesized}
      expanded={expanded && !isRunning}
      onToggle={() => setExpanded(!expanded)}
      rerunLabel="Re-run synthesis"
      settingsLabel="Synthesis pipeline"
      taskId={null}
      emptyMessage="Synthesis pipeline not yet run for this episode."
      controls={!isRunning ? (
        <div className="px-4 pb-3 space-y-4">
          {/* ── Step 1: Voice extraction ────────── */}
          <section className="space-y-3">
            <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              1. Extract Voice Samples
            </h5>
            <div className="flex items-center gap-3">
              <Button
                onClick={() => extractMutation.mutate()}
                disabled={extractMutation.isPending}
                size="sm"
              >
                {status?.voice_samples_extracted ? "Re-extract" : "Extract Voices"}
              </Button>
              {status?.voice_samples_extracted && (
                <span className="text-xs text-green-400">Samples extracted</span>
              )}
              {extractMutation.isError && (
                <span className="text-xs text-destructive">
                  {(extractMutation.error as Error).message}
                </span>
              )}
            </div>

            {voiceSamples && Object.keys(voiceSamples).length > 0 && (
              <div className="space-y-2">
                {Object.entries(voiceSamples).map(([speaker, samples]) => (
                  <div key={speaker} className="text-sm">
                    <span className="font-medium">{speaker}</span>
                    <span className="text-muted-foreground ml-2">
                      ({samples.length} sample{samples.length !== 1 ? "s" : ""})
                    </span>
                    <div className="flex gap-2 mt-1 flex-wrap">
                      {samples.map((s, i) => (
                        <button
                          key={i}
                          onClick={() => {
                            const audio = new Audio(audioFileUrl(s.file));
                            audio.play();
                          }}
                          className="text-xs px-2 py-1 rounded bg-secondary hover:bg-accent transition border border-border"
                          title={s.text || `Sample ${i + 1}`}
                        >
                          {s.duration.toFixed(1)}s
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* ── Step 2: TTS generation ─────────── */}
          <section className="space-y-3 border-t border-border/50 pt-3">
            <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              2. Generate TTS Segments
            </h5>

            <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 items-center text-sm">
              <HelpLabel label="Language" help="The language the generated speech should sound like." />
              <input
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="input py-1 text-sm"
              />
              <HelpLabel label="Model size" help="Larger model produces more natural-sounding speech but needs more GPU memory." />
              <select
                value={modelSize}
                onChange={(e) => setModelSize(e.target.value)}
                className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
              >
                {pipelineConfig
                  ? Object.entries(pipelineConfig.tts_model_sizes).map(([key, desc]) => (
                      <option key={key} value={key}>{key} — {desc}</option>
                    ))
                  : <option value={modelSize}>{modelSize}</option>
                }
              </select>
              {episode.translations.length > 0 && (
                <>
                  <HelpLabel label="Text source" help="Which text to synthesize. 'Best available' picks the translation if one exists, otherwise the polished or raw transcript." />
                  <select
                    value={sourceLang}
                    onChange={(e) => setSourceLang(e.target.value)}
                    className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
                  >
                    <option value="">Best available</option>
                    {episode.translations.map((lang) => (
                      <option key={lang} value={lang}>{lang}</option>
                    ))}
                  </select>
                </>
              )}
            </div>

            {/* Advanced TTS settings */}
            <div className="space-y-3">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
              >
                <Settings2 className="w-3 h-3" />
                <span className="font-semibold uppercase tracking-wide">
                  {showAdvanced ? "Hide advanced" : "Advanced settings"}
                </span>
              </button>
              {showAdvanced && (
                <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 items-center text-sm pl-3 border-l-2 border-border">
                  <HelpLabel label="Max chunk (s)" help="Maximum duration in seconds for each TTS chunk. Shorter chunks are more stable, longer ones sound more natural." />
                  <input
                    type="number"
                    value={maxChunkDuration}
                    onChange={(e) => setMaxChunkDuration(Number(e.target.value))}
                    className="input py-1 text-sm w-20"
                    min={5}
                    max={60}
                  />
                </div>
              )}
            </div>

            <div className="flex items-center gap-3">
              <Button
                onClick={() => generateMutation.mutate()}
                disabled={!status?.voice_samples_extracted || generateMutation.isPending}
                size="sm"
              >
                {status?.tts_segments_generated ? "Re-generate" : "Generate"}
              </Button>
              {!status?.voice_samples_extracted && (
                <span className="text-xs text-muted-foreground">Extract voices first</span>
              )}
              {status?.tts_segments_generated && (
                <span className="text-xs text-green-400">
                  {generatedSegments?.length ?? "?"} segments generated
                </span>
              )}
              {generateMutation.isError && (
                <span className="text-xs text-destructive">
                  {(generateMutation.error as Error).message}
                </span>
              )}
            </div>
          </section>

          {/* ── Step 3: Assembly ───────────────── */}
          <section className="space-y-3 border-t border-border/50 pt-3">
            <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              3. Assemble Episode
            </h5>

            <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 items-center text-sm">
              <HelpLabel label="Strategy" help="How to handle pauses between segments in the final audio." />
              <select
                value={assembleStrategy}
                onChange={(e) => setAssembleStrategy(e.target.value)}
                className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
              >
                {pipelineConfig
                  ? Object.entries(pipelineConfig.assemble_strategies).map(([key, desc]) => (
                      <option key={key} value={key} title={desc}>{desc}</option>
                    ))
                  : <option value={assembleStrategy}>{assembleStrategy}</option>
                }
              </select>
            </div>

            <div className="flex items-center gap-3">
              <Button
                onClick={() => assembleMutation.mutate()}
                disabled={!status?.tts_segments_generated || assembleMutation.isPending}
                size="sm"
              >
                {assembleMutation.isPending ? "Assembling..." : "Assemble"}
              </Button>
              {assembleMutation.isError && (
                <span className="text-xs text-destructive">
                  {(assembleMutation.error as Error).message}
                </span>
              )}
            </div>
          </section>
        </div>
      ) : undefined}
    >
      {/* Progress bars (when running) */}
      {extractTaskId && (
        <ProgressBar taskId={extractTaskId} onComplete={() => { refreshQueries(); setExtractTaskId(null); }} />
      )}
      {generateTaskId && (
        <ProgressBar taskId={generateTaskId} onComplete={() => { refreshQueries(); setGenerateTaskId(null); }} />
      )}

      {/* Result — shown when synthesized and controls collapsed */}
      {status?.synthesized && !expanded && !isRunning && (
        <div className="p-4 space-y-3">
          {assembleMutation.data && (
            <p className="text-xs text-green-400">
              Assembled ({(assembleMutation.data.duration / 60).toFixed(1)} min)
            </p>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              const synthPath = episode.audio_path!.replace(/\.[^.]+$/, ".synthesized.wav");
              playAudio(
                assembleMutation.data?.path || synthPath,
                `${episode.title} (Synthesized)`,
                "",
                showMeta?.name,
              );
            }}
          >
            Play Synthesized
          </Button>
        </div>
      )}
    </PipelinePanel>
  );
}
