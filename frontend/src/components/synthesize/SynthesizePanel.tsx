import { useCallback, useMemo, useState } from "react";
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
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { useCapabilities } from "@/hooks/useCapabilities";
import MissingDependency from "@/components/common/MissingDependency";
import ProgressBar from "@/components/editor/ProgressBar";
import PipelinePanel from "@/components/common/PipelinePanel";
import { segKey } from "@/lib/segKey";
import SourceSegmentPicker, { type ResolvedSource } from "./SourceSegmentPicker";
import VoiceExtractionSection from "./VoiceExtractionSection";
import TTSGenerationSection from "./TTSGenerationSection";
import AssemblySection from "./AssemblySection";

export default function SynthesizePanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const queryClient = useQueryClient();
  const seekTo = useAudioStore((s) => s.seekTo);
  const setAudioMeta = useAudioStore((s) => s.setAudioMeta);

  const [extractTaskId, setExtractTaskId] = useState<string | null>(null);
  const [generateTaskId, setGenerateTaskId] = useState<string | null>(null);
  const [language, setLanguage] = useState(showMeta?.language || "English");
  const [modelSize, setModelSize] = useState("1.7B");
  const [maxChunkDuration, setMaxChunkDuration] = useState(20);
  const [force, setForce] = useState(false);
  const [onlySpeakers, setOnlySpeakers] = useState<string[]>([]);
  const [assembleStrategy, setAssembleStrategy] = useState("original_timing");
  const [silenceDuration, setSilenceDuration] = useState(0.5);
  const [expanded, setExpanded] = useState(!episode?.synthesized);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [expandedSeg, setExpandedSeg] = useState<string | null>(null);
  const [showCount, setShowCount] = useState<Record<string, number>>({});
  const [speakerOverrides, setSpeakerOverrides] = useState<Record<string, string>>({});

  // Source picker: null == "latest valid version". Resolved step/lang
  // come back via onResolvedSourceChange so the generate request knows
  // exactly what to send.
  const [sourceVersionId, setSourceVersionId] = useState<string | null>(null);
  const [resolvedSource, setResolvedSource] = useState<ResolvedSource>({
    step: "transcript",
    lang: "",
    sourceLang: undefined,
    sourceVersionId: null,
  });
  const [sourceSelection, setSourceSelection] = useState<Set<string>>(() => new Set());
  const [sourceSegments, setSourceSegments] = useState<Segment[]>([]);

  const { data: pipelineConfig } = useQuery({
    queryKey: queryKeys.pipelineConfig(),
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: queryKeys.synthesizeStatus(episode?.audio_path),
    queryFn: () => getSynthesisStatus(episode!.audio_path!),
    enabled: !!episode?.audio_path,
  });

  // All speakers that appear in the currently loaded source version.
  const allSpeakers = useMemo(() => {
    const set = new Set<string>();
    for (const seg of sourceSegments) {
      const sp = seg.speaker || "";
      if (sp && sp !== "[BREAK]" && sp !== "UNKNOWN" && sp !== "UNK") set.add(sp);
    }
    return [...set].sort();
  }, [sourceSegments]);

  // Effective working set: only segments the user kept checked in the
  // source picker. Unchecked rows are dropped from voice sampling AND
  // from the generated output.
  const workingSegments = useMemo(
    () => sourceSegments.filter((seg) => sourceSelection.has(segKey(seg))),
    [sourceSegments, sourceSelection],
  );

  // Group the working set by OUTPUT speaker (after applying reassignments).
  const segmentsBySpeaker = useMemo(() => {
    const grouped: Record<string, Segment[]> = {};
    for (const seg of workingSegments) {
      const sp = seg.speaker || "";
      if (!sp || sp === "[BREAK]" || sp === "UNKNOWN" || sp === "UNK") continue;
      const effectiveSpeaker = speakerOverrides[segKey(seg)] || sp;
      (grouped[effectiveSpeaker] ??= []).push(seg);
    }
    return grouped;
  }, [workingSegments, speakerOverrides]);

  const { data: voiceSamples, refetch: refetchVoiceSamples } = useQuery({
    queryKey: queryKeys.synthesizeVoices(episode?.audio_path),
    queryFn: () => getVoiceSamples(episode!.audio_path!),
    enabled: !!episode?.audio_path && !!status?.voice_samples_extracted,
  });

  const { data: generatedSegments } = useQuery({
    queryKey: queryKeys.synthesizeGenerated(episode?.audio_path),
    queryFn: () => getGeneratedSegments(episode!.audio_path!),
    enabled: !!episode?.audio_path && !!status?.tts_segments_generated,
  });

  const refreshQueries = useCallback(() => {
    refetchStatus();
    queryClient.invalidateQueries({ queryKey: queryKeys.synthesizeAll() });
    queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
  }, [queryClient, refetchStatus]);

  const extractMutation = useMutation({
    mutationFn: () => {
      const selections = workingSegments
        .filter((seg) => selected.has(segKey(seg)))
        .map((seg) => ({
          speaker: speakerOverrides[segKey(seg)] || seg.speaker || "",
          start: seg.start,
          end: seg.end,
          text: seg.text,
        }));
      return extractSelectedSamples(episode!.audio_path!, selections);
    },
    onSuccess: () => {
      refreshQueries();
      refetchVoiceSamples();
    },
  });

  const uploadMutation = useMutation({
    mutationFn: ({ speaker, file }: { speaker: string; file: File }) =>
      uploadVoiceSample(episode!.audio_path!, speaker, file),
    onSuccess: () => {
      refreshQueries();
      refetchVoiceSamples();
    },
  });

  const generateMutation = useMutation({
    mutationFn: () =>
      startGenerateTTS({
        audio_path: episode!.audio_path!,
        model_size: modelSize,
        language,
        source_lang: resolvedSource.sourceLang,
        source_version_id: resolvedSource.sourceVersionId ?? undefined,
        max_chunk_duration: maxChunkDuration,
        force,
        only_speakers: onlySpeakers.length > 0 ? onlySpeakers : undefined,
        keep_segment_keys: Array.from(sourceSelection),
      }),
    onSuccess: (data) => setGenerateTaskId(data.task_id),
  });

  const assembleMutation = useMutation({
    mutationFn: () =>
      assembleEpisode({
        audio_path: episode!.audio_path!,
        strategy: assembleStrategy,
        silence_duration: silenceDuration,
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

  if (!episode) return null;

  const prereq = !episode.audio_path
    ? "Download the audio file first before synthesizing."
    : !episode.transcribed
      ? "You need a transcript first. Go to the Transcribe tab to create one."
      : undefined;

  const isRunning = !!extractTaskId || !!generateTaskId;

  const sourceSummary =
    resolvedSource.step === "translate"
      ? `${resolvedSource.lang} translation`
      : resolvedSource.step === "corrected"
        ? "corrected transcript"
        : "raw transcript";

  return (
    <PipelinePanel
      title="Synthesize"
      description="Re-create the episode with cloned voices."
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
      taskId={null}
      onRetry={handleRetry}
      onDismiss={handleDismiss}
      emptyMessage="Synthesis pipeline not yet run for this episode."
      controls={!isRunning ? (
        <div className="px-4 pb-3 space-y-4">
          <SourceSegmentPicker
            audioPath={episode.audio_path!}
            episode={episode}
            sourceVersionId={sourceVersionId}
            setSourceVersionId={setSourceVersionId}
            onResolvedSourceChange={setResolvedSource}
            onSegmentsChange={setSourceSegments}
            selectedKeys={sourceSelection}
            setSelectedKeys={setSourceSelection}
            seekTo={seekTo}
          />

          <VoiceExtractionSection
            segmentsBySpeaker={segmentsBySpeaker}
            allSpeakers={allSpeakers}
            selected={selected}
            setSelected={setSelected}
            expandedSeg={expandedSeg}
            setExpandedSeg={setExpandedSeg}
            showCount={showCount}
            setShowCount={setShowCount}
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
            maxChunkDuration={maxChunkDuration}
            setMaxChunkDuration={setMaxChunkDuration}
            force={force}
            setForce={setForce}
            onlySpeakers={onlySpeakers}
            setOnlySpeakers={setOnlySpeakers}
            allSpeakers={allSpeakers}
            sourceSummary={sourceSummary}
            pipelineConfig={pipelineConfig}
            status={status}
            generatedSegments={generatedSegments}
            generateMutation={generateMutation}
          />

          <AssemblySection
            assembleStrategy={assembleStrategy}
            setAssembleStrategy={setAssembleStrategy}
            silenceDuration={silenceDuration}
            setSilenceDuration={setSilenceDuration}
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
            <p className="text-xs text-success">
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
