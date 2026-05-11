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
  cancelTask,
  getSynthesizeVersions,
  getSynthesizeVersionPath,
  deleteSynthesizeVersion,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { useCapabilities } from "@/hooks/useCapabilities";
import MissingDependency from "@/components/common/MissingDependency";
import ProgressBar from "@/components/editor/ProgressBar";
import PipelinePanel from "@/components/common/PipelinePanel";
import { segKey } from "@/lib/segKey";
import { versionDate } from "@/lib/utils";
import SourceSegmentPicker, { type ResolvedSource } from "./SourceSegmentPicker";
import VoiceExtractionSection from "./VoiceExtractionSection";
import TTSGenerationSection from "./TTSGenerationSection";
import AssemblySection from "./AssemblySection";
import { plainStatus } from "@/lib/stepStatus";

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
  const [silenceDuration, setSilenceDuration] = useState(0.2);
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
  const [selectedSynthVersionId, setSelectedSynthVersionId] = useState<string | null>(null);

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

  // Group the FULL source by OUTPUT speaker (after applying reassignments)
  // for voice sample extraction / clip browsing. Independent of the source
  // picker's checkbox scope — narrowing synthesis shouldn't starve the
  // cloned voice of source clips. The narrowed scope is sent to the backend
  // separately as `keep_segment_keys` for the generation step only.
  const segmentsBySpeaker = useMemo(() => {
    const grouped: Record<string, Segment[]> = {};
    for (const seg of sourceSegments) {
      const sp = seg.speaker || "";
      if (!sp || sp === "[BREAK]" || sp === "UNKNOWN" || sp === "UNK") continue;
      const effectiveSpeaker = speakerOverrides[segKey(seg)] || sp;
      (grouped[effectiveSpeaker] ??= []).push(seg);
    }
    return grouped;
  }, [sourceSegments, speakerOverrides]);

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

  const { data: synthVersions } = useQuery({
    queryKey: queryKeys.synthesizeVersions(episode?.audio_path ?? episode?.output_dir),
    queryFn: () => getSynthesizeVersions(episode?.audio_path, episode?.output_dir),
    enabled: (!!episode?.audio_path || !!episode?.output_dir) && !!status?.synthesized,
  });

  const activeSynthVersion = synthVersions?.find((v) => v.id === selectedSynthVersionId) ?? synthVersions?.[0];

  const deleteSynthVersionMutation = useMutation({
    mutationFn: (versionId: string) =>
      deleteSynthesizeVersion(episode?.audio_path, versionId, episode?.output_dir),
    onSuccess: () => {
      setSelectedSynthVersionId(null);
      refreshQueries();
    },
  });

  const refreshQueries = useCallback(() => {
    refetchStatus();
    queryClient.invalidateQueries({ queryKey: queryKeys.synthesizeAll() });
    queryClient.invalidateQueries({ queryKey: queryKeys.synthesizeVersions(episode?.audio_path) });
    queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
  }, [queryClient, refetchStatus, episode?.audio_path]);

  const extractMutation = useMutation({
    mutationFn: () => {
      const selections = sourceSegments
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
        language,
        model_size: modelSize,
        source_version_id: resolvedSource.sourceVersionId ?? undefined,
      }),
    onSuccess: (data) => {
      refreshQueries();
      setExpanded(false);
      setSelectedSynthVersionId(data.version_id);
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
      status={plainStatus(!!episode.synthesized || !!status?.synthesized)}
      expanded={expanded && !isRunning}
      onToggle={() => setExpanded(!expanded)}
      rerunLabel="Re-run synthesis"
      taskId={null}
      onRetry={handleRetry}
      onDismiss={handleDismiss}
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
          onDismiss={() => { cancelTask(extractTaskId).catch(() => {}); setExtractTaskId(null); }}
          onCancel={() => { cancelTask(extractTaskId).catch(() => {}); setExtractTaskId(null); }}
        />
      )}
      {generateTaskId && (
        <ProgressBar
          taskId={generateTaskId}
          onComplete={() => { refreshQueries(); setGenerateTaskId(null); }}
          onRetry={() => { setGenerateTaskId(null); generateMutation.mutate(); }}
          onDismiss={() => { cancelTask(generateTaskId).catch(() => {}); setGenerateTaskId(null); }}
          onCancel={() => { cancelTask(generateTaskId).catch(() => {}); setGenerateTaskId(null); }}
        />
      )}

      {/* Result — visible whenever synthesized, regardless of whether
          re-run controls are expanded above. */}
      {status?.synthesized && !isRunning && (
        <div className="p-4 space-y-3">
          {synthVersions && synthVersions.length > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Version</span>
              <select
                value={activeSynthVersion?.id ?? ""}
                onChange={(e) => setSelectedSynthVersionId(e.target.value)}
                className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border max-w-[18rem]"
              >
                {synthVersions.map((v, i) => {
                  const params = (v.params ?? {}) as { strategy?: string; language?: string; duration_s?: number };
                  const parts = [
                    i === 0 ? "Latest" : null,
                    versionDate(v),
                    params.language,
                    params.strategy,
                    params.duration_s ? `${(params.duration_s / 60).toFixed(1)}m` : null,
                  ].filter(Boolean);
                  return (
                    <option key={v.id} value={v.id}>
                      {parts.join(" · ")}
                    </option>
                  );
                })}
              </select>
              {activeSynthVersion && synthVersions.length > 1 && (
                <button
                  type="button"
                  onClick={() => {
                    if (confirm("Delete this synthesized version?")) {
                      deleteSynthVersionMutation.mutate(activeSynthVersion.id);
                    }
                  }}
                  className="text-xs text-muted-foreground hover:text-destructive transition px-1"
                  title="Delete this version"
                  disabled={deleteSynthVersionMutation.isPending}
                >
                  Delete
                </button>
              )}
            </div>
          )}
          <Button
            variant="outline"
            size="sm"
            disabled={!activeSynthVersion && !assembleMutation.data}
            onClick={async () => {
              let synthPath: string | null = null;
              if (activeSynthVersion) {
                try {
                  const v = await getSynthesizeVersionPath(episode.audio_path, activeSynthVersion.id, episode.output_dir);
                  synthPath = v.path;
                } catch {
                  return;
                }
              } else {
                synthPath = assembleMutation.data?.path ?? episode.audio_path!.replace(/\.[^.]+$/, ".synthesized.wav");
              }
              if (!synthPath) return;
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
