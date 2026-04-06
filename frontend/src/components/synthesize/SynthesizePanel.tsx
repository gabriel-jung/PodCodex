import { useState, useCallback, useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Segment } from "@/api/types";
import { useEpisodeStore, useAudioStore } from "@/stores";
import {
  getSynthesisStatus,
  getVoiceSamples,
  extractSelectedSamples,
  uploadVoiceSample,
  startGenerateTTS,
  getGeneratedSegments,
  assembleEpisode,
  getPipelineConfig,
  getSegments,
  getPolishSegments,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { useCapabilities } from "@/hooks/useCapabilities";
import MissingDependency from "@/components/common/MissingDependency";
import ProgressBar from "@/components/editor/ProgressBar";
import PipelinePanel from "@/components/common/PipelinePanel";
import VoiceExtractionSection, { segKey } from "./VoiceExtractionSection";
import TTSGenerationSection from "./TTSGenerationSection";
import AssemblySection from "./AssemblySection";

export default function SynthesizePanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  if (!episode) return null;
  const queryClient = useQueryClient();
  const { seekTo, setAudioMeta } = useAudioStore();

  const [extractTaskId, setExtractTaskId] = useState<string | null>(null);
  const [generateTaskId, setGenerateTaskId] = useState<string | null>(null);
  const [language, setLanguage] = useState(showMeta?.language || "English");
  const [modelSize, setModelSize] = useState("1.7B");
  const [sourceLang, setSourceLang] = useState("");
  const [maxChunkDuration, setMaxChunkDuration] = useState(20);
  const [assembleStrategy, setAssembleStrategy] = useState("original_timing");
  const [expanded, setExpanded] = useState(!episode.synthesized);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [expandedSeg, setExpandedSeg] = useState<string | null>(null);
  const [showCount, setShowCount] = useState<Record<string, number>>({});
  const [timeFrom, setTimeFrom] = useState("");
  const [timeTo, setTimeTo] = useState("");
  const [speakerOverrides, setSpeakerOverrides] = useState<Record<string, string>>({});

  const { data: pipelineConfig } = useQuery({
    queryKey: queryKeys.pipelineConfig(),
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: queryKeys.synthesizeStatus(episode.audio_path),
    queryFn: () => getSynthesisStatus(episode.audio_path!),
    enabled: !!episode.audio_path,
  });

  // Load transcript segments for speaker browsing
  const { data: transcriptSegments } = useQuery({
    queryKey: queryKeys.synthSourceSegments(episode.audio_path),
    queryFn: async () => {
      if (!episode.audio_path) return [];
      try {
        if (episode.polished) return await getPolishSegments(episode.audio_path);
      } catch { /* fall through */ }
      return getSegments(episode.audio_path);
    },
    enabled: !!episode.audio_path && episode.transcribed,
  });

  /** Parse "mm:ss" or "hh:mm:ss" or plain seconds to seconds. */
  const parseTime = (v: string): number | null => {
    if (!v.trim()) return null;
    const parts = v.trim().split(":").map(Number);
    if (parts.some(isNaN)) return null;
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    return parts[0];
  };

  const fromSec = parseTime(timeFrom);
  const toSec = parseTime(timeTo);

  // All speakers (for reassignment dropdown)
  const allSpeakers = useMemo(() => {
    if (!transcriptSegments) return [];
    const set = new Set<string>();
    for (const seg of transcriptSegments) {
      const sp = seg.speaker || "";
      if (sp && sp !== "[BREAK]" && sp !== "UNKNOWN" && sp !== "UNK") set.add(sp);
    }
    return [...set].sort();
  }, [transcriptSegments]);

  // Group segments by speaker (excluding breaks/unknown), applying time range + overrides
  const segmentsBySpeaker = useMemo(() => {
    if (!transcriptSegments) return {};
    const grouped: Record<string, Segment[]> = {};
    for (const seg of transcriptSegments) {
      const sp = seg.speaker || "";
      if (!sp || sp === "[BREAK]" || sp === "UNKNOWN" || sp === "UNK") continue;
      // Time range filter
      if (fromSec != null && seg.end < fromSec) continue;
      if (toSec != null && seg.start > toSec) continue;
      // Apply speaker override for display grouping
      const effectiveSpeaker = speakerOverrides[segKey(seg)] || sp;
      (grouped[effectiveSpeaker] ??= []).push(seg);
    }
    return grouped;
  }, [transcriptSegments, fromSec, toSec, speakerOverrides]);

  const { data: voiceSamples, refetch: refetchVoiceSamples } = useQuery({
    queryKey: queryKeys.synthesizeVoices(episode.audio_path),
    queryFn: () => getVoiceSamples(episode.audio_path!),
    enabled: !!episode.audio_path && !!status?.voice_samples_extracted,
  });

  const { data: generatedSegments } = useQuery({
    queryKey: queryKeys.synthesizeGenerated(episode.audio_path),
    queryFn: () => getGeneratedSegments(episode.audio_path!),
    enabled: !!episode.audio_path && !!status?.tts_segments_generated,
  });

  const refreshQueries = useCallback(() => {
    refetchStatus();
    queryClient.invalidateQueries({ queryKey: queryKeys.synthesizeAll() });
    queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
  }, [queryClient, refetchStatus]);

  const extractMutation = useMutation({
    mutationFn: () => {
      const selections = (transcriptSegments || [])
        .filter((seg) => selected.has(segKey(seg)))
        .map((seg) => ({
          speaker: speakerOverrides[segKey(seg)] || seg.speaker || "",
          start: seg.start,
          end: seg.end,
          text: seg.text,
        }));
      return extractSelectedSamples(episode.audio_path!, selections);
    },
    onSuccess: () => {
      refreshQueries();
      refetchVoiceSamples();
    },
  });

  const uploadMutation = useMutation({
    mutationFn: ({ speaker, file }: { speaker: string; file: File }) =>
      uploadVoiceSample(episode.audio_path!, speaker, file),
    onSuccess: () => {
      refreshQueries();
      refetchVoiceSamples();
    },
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

  const handleRetry = () => {
    setExtractTaskId(null);
    setGenerateTaskId(null);
    setExpanded(true);
  };
  const handleDismiss = () => {
    setExtractTaskId(null);
    setGenerateTaskId(null);
  };

  const { has: hasCap } = useCapabilities();
  const hasTTS = hasCap("tts") && hasCap("soundfile");

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
      blocker={!prereq && !hasTTS ? (
        <MissingDependency
          extra="pipeline"
          label="Synthesis libraries"
          description="Voice cloning and TTS generation require soundfile, qwen-tts, and other dependencies from the pipeline extra."
        />
      ) : undefined}
      done={episode.synthesized}
      expanded={expanded && !isRunning}
      onToggle={() => setExpanded(!expanded)}
      rerunLabel="Re-run synthesis"
      settingsLabel="Synthesis pipeline"
      taskId={null}
      onRetry={handleRetry}
      onDismiss={handleDismiss}
      emptyMessage="Synthesis pipeline not yet run for this episode."
      controls={!isRunning ? (
        <div className="px-4 pb-3 space-y-4">
          <VoiceExtractionSection
            segmentsBySpeaker={segmentsBySpeaker}
            allSpeakers={allSpeakers}
            selected={selected}
            setSelected={setSelected}
            expandedSeg={expandedSeg}
            setExpandedSeg={setExpandedSeg}
            showCount={showCount}
            setShowCount={setShowCount}
            timeFrom={timeFrom}
            setTimeFrom={setTimeFrom}
            timeTo={timeTo}
            setTimeTo={setTimeTo}
            speakerOverrides={speakerOverrides}
            setSpeakerOverrides={setSpeakerOverrides}
            extractMutation={extractMutation}
            uploadMutation={uploadMutation}
            status={status}
            voiceSamples={voiceSamples}
            seekTo={seekTo}
            audioPath={episode.audio_path!}
          />

          <TTSGenerationSection
            language={language}
            setLanguage={setLanguage}
            modelSize={modelSize}
            setModelSize={setModelSize}
            sourceLang={sourceLang}
            setSourceLang={setSourceLang}
            maxChunkDuration={maxChunkDuration}
            setMaxChunkDuration={setMaxChunkDuration}
            translations={episode.translations}
            pipelineConfig={pipelineConfig}
            status={status}
            generatedSegments={generatedSegments}
            generateMutation={generateMutation}
          />

          <AssemblySection
            assembleStrategy={assembleStrategy}
            setAssembleStrategy={setAssembleStrategy}
            pipelineConfig={pipelineConfig}
            status={status}
            assembleMutation={assembleMutation}
          />
        </div>
      ) : undefined}
    >
      {/* Progress bars (when running) */}
      {extractTaskId && (
        <ProgressBar
          taskId={extractTaskId}
          onComplete={() => { refreshQueries(); setExtractTaskId(null); }}
          onRetry={() => { setExtractTaskId(null); extractMutation.mutate(); }}
          onDismiss={() => setExtractTaskId(null)}
        />
      )}
      {generateTaskId && (
        <ProgressBar
          taskId={generateTaskId}
          onComplete={() => { refreshQueries(); setGenerateTaskId(null); }}
          onRetry={() => { setGenerateTaskId(null); generateMutation.mutate(); }}
          onDismiss={() => setGenerateTaskId(null)}
        />
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
              const synthPath = assembleMutation.data?.path || episode.audio_path!.replace(/\.[^.]+$/, ".synthesized.wav");
              setAudioMeta(synthPath, { title: `${episode.title} (Synthesized)`, showName: showMeta?.name });
              seekTo(synthPath, 0);
            }}
          >
            Play Synthesized
          </Button>
        </div>
      )}
    </PipelinePanel>
  );
}
