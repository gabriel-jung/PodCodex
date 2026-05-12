import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { AssembleStrategy, Segment } from "@/api/types";
import { useEpisodeStore, useAudioStore, useTaskStore } from "@/stores";
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
import { resolveSynthSpeaker } from "@/lib/speakers";
import { versionDate } from "@/lib/utils";
import SourceSegmentPicker, { type ResolvedSource } from "./SourceSegmentPicker";
import VoiceExtractionSection from "./VoiceExtractionSection";
import TTSGenerationSection from "./TTSGenerationSection";
import AssemblySection from "./AssemblySection";
import { plainStatus } from "@/lib/stepStatus";
import { getEpisodeSourceRef, getEpisodeStem } from "@/lib/episodeRef";

export default function SynthesizePanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const folder = useEpisodeStore((s) => s.folder);
  const setEpisodeTask = useTaskStore((s) => s.setEpisodeTask);
  const queryClient = useQueryClient();
  const seekTo = useAudioStore((s) => s.seekTo);
  const setAudioMeta = useAudioStore((s) => s.setAudioMeta);

  const [extractTaskId, setExtractTaskId] = useState<string | null>(null);
  const [generateTaskId, setGenerateTaskId] = useState<string | null>(null);
  // Seed language from showMeta on first render; if showMeta arrives later
  // (panel mounts before the meta query resolves), sync via the effect below.
  const [language, setLanguage] = useState(showMeta?.language || "English");
  const languageInitRef = useRef(!!showMeta?.language);
  useEffect(() => {
    if (!languageInitRef.current && showMeta?.language) {
      setLanguage(showMeta.language);
      languageInitRef.current = true;
    }
  }, [showMeta?.language]);
  const [modelSize, setModelSize] = useState("1.7B");
  const [maxChunkDuration, setMaxChunkDuration] = useState(20);
  const [force, setForce] = useState(false);
  const [onlySpeakers, setOnlySpeakers] = useState<string[]>([]);
  const [assembleStrategy, setAssembleStrategy] = useState<AssembleStrategy>("silence");
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
  const sourceSelectionStampRef = useRef<string>("");
  const [sourceSegments, setSourceSegments] = useState<Segment[]>([]);
  const [selectedSynthVersionId, setSelectedSynthVersionId] = useState<string | null>(null);

  // Selection / overrides / per-speaker UI state are keyed by segKey, which
  // depends on speaker + timestamps of the CURRENT source. Switching the
  // source (different version or step) silently invalidates every key —
  // wipe them so stale entries don't disable the Extract button or attribute
  // segments to dead speakers.
  const sourceFingerprint = `${resolvedSource.step}|${resolvedSource.lang}|${resolvedSource.sourceVersionId ?? "latest"}`;
  const lastSourceFingerprintRef = useRef<string>(sourceFingerprint);
  useEffect(() => {
    if (lastSourceFingerprintRef.current === sourceFingerprint) return;
    lastSourceFingerprintRef.current = sourceFingerprint;
    setSelected(new Set());
    setSpeakerOverrides({});
    setShowCount({});
    setExpandedSeg(null);
  }, [sourceFingerprint]);

  const { data: pipelineConfig } = useQuery({
    queryKey: queryKeys.pipelineConfig(),
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  const { audioPath, outputDir, sourceRef, hasSourceRef, noAudio } = getEpisodeSourceRef(episode);

  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: queryKeys.synthesizeStatus(sourceRef),
    queryFn: () => getSynthesisStatus(audioPath, outputDir),
    enabled: hasSourceRef,
  });

  // Single pass over sourceSegments. Empty/unknown speakers route to
  // NARRATOR_SPEAKER so legacy transcripts can still upload one voice
  // sample without renaming in the editor first.
  //   - allSpeakers: pre-override names (drives the reassignment dropdown
  //     options and TTS "only-speakers" filter)
  //   - segmentsBySpeaker: post-override grouping (drives the voice-sample
  //     UI — narrowing synthesis must not starve voice clips, so this is
  //     independent of the source picker's checkbox scope)
  const { allSpeakers, segmentsBySpeaker } = useMemo(() => {
    const grouped: Record<string, Segment[]> = {};
    const original = new Set<string>();
    const hasOverrides = Object.keys(speakerOverrides).length > 0;
    for (const seg of sourceSegments) {
      const sp = resolveSynthSpeaker(seg.speaker ?? "");
      if (!sp) continue;
      original.add(sp);
      const effectiveSpeaker = hasOverrides ? speakerOverrides[segKey(seg)] || sp : sp;
      (grouped[effectiveSpeaker] ??= []).push(seg);
    }
    return { allSpeakers: [...original].sort(), segmentsBySpeaker: grouped };
  }, [sourceSegments, speakerOverrides]);

  const resolvedVersionId = resolvedSource.sourceVersionId ?? null;

  const { data: voiceSamples } = useQuery({
    queryKey: queryKeys.synthesizeVoices(sourceRef, resolvedVersionId),
    queryFn: () => getVoiceSamples(audioPath, outputDir, resolvedVersionId),
    enabled: hasSourceRef && !!status?.voice_samples_extracted,
  });

  const { data: generatedSegments } = useQuery({
    queryKey: queryKeys.synthesizeGenerated(sourceRef, resolvedVersionId),
    queryFn: () => getGeneratedSegments(audioPath, outputDir, resolvedVersionId),
    enabled: hasSourceRef && !!status?.tts_segments_generated,
  });

  // Speakers in the active scope that lack any voice sample on disk.
  // Honors `onlySpeakers` if the user narrowed generation. Generation
  // would silently skip these segments without a sample to clone from.
  const speakersMissingSamples = useMemo(() => {
    const active = onlySpeakers.length > 0 ? onlySpeakers : allSpeakers;
    const have = voiceSamples ?? {};
    return active.filter((sp) => !(sp in have)).sort();
  }, [allSpeakers, onlySpeakers, voiceSamples]);

  // Segments in the current scope that have no TTS audio yet. Drives the
  // assemble-time warning: partial generations (only_speakers, errors,
  // cancellation) would otherwise produce a silently-shortened podcast.
  const missingGeneratedCount = useMemo(() => {
    if (sourceSelection.size === 0) return 0;
    return Math.max(0, sourceSelection.size - (generatedSegments?.length ?? 0));
  }, [sourceSelection, generatedSegments]);

  const { data: synthVersions } = useQuery({
    queryKey: queryKeys.synthesizeVersions(sourceRef),
    queryFn: () => getSynthesizeVersions(audioPath, outputDir),
    enabled: hasSourceRef && !!status?.synthesized,
  });

  const activeSynthVersion = useMemo(
    () => synthVersions?.find((v) => v.id === selectedSynthVersionId) ?? synthVersions?.[0],
    [synthVersions, selectedSynthVersionId],
  );

  const synthVersionOptions = useMemo(() => {
    if (!synthVersions) return [];
    return synthVersions.map((v, i) => {
      const params = (v.params ?? {}) as { strategy?: string; language?: string; duration_s?: number };
      const parts = [
        i === 0 ? "Latest" : null,
        versionDate(v),
        params.language,
        params.strategy,
        params.duration_s ? `${(params.duration_s / 60).toFixed(1)}m` : null,
      ].filter(Boolean);
      return { id: v.id, label: parts.join(" · ") };
    });
  }, [synthVersions]);

  const deleteSynthVersionMutation = useMutation({
    mutationFn: (versionId: string) =>
      deleteSynthesizeVersion(audioPath, versionId, outputDir),
    onSuccess: () => {
      setSelectedSynthVersionId(null);
      refreshQueries();
    },
  });

  const refreshQueries = useCallback(() => {
    refetchStatus();
    queryClient.invalidateQueries({ queryKey: queryKeys.synthesizeAll() });
    queryClient.invalidateQueries({ queryKey: queryKeys.synthesizeVersions(sourceRef) });
    queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
  }, [queryClient, refetchStatus, sourceRef]);

  const extractMutation = useMutation({
    mutationFn: () => {
      const selections = sourceSegments
        .filter((seg) => selected.has(segKey(seg)))
        .map((seg) => {
          const override = speakerOverrides[segKey(seg)];
          const resolved = override || resolveSynthSpeaker(seg.speaker ?? "") || "";
          return {
            speaker: resolved,
            start: seg.start,
            end: seg.end,
            text: seg.text,
          };
        });
      return extractSelectedSamples(audioPath, selections, outputDir);
    },
    onSuccess: () => {
      refreshQueries();
    },
  });

  const uploadMutation = useMutation({
    mutationFn: ({ speaker, file }: { speaker: string; file: File }) =>
      uploadVoiceSample(audioPath, speaker, file, outputDir),
    onSuccess: () => {
      refreshQueries();
    },
  });

  const generateMutation = useMutation({
    mutationFn: () =>
      startGenerateTTS({
        audio_path: audioPath ?? undefined,
        output_dir: outputDir ?? undefined,
        model_size: modelSize,
        language,
        source_lang: resolvedSource.sourceLang,
        source_version_id: resolvedSource.sourceVersionId ?? undefined,
        max_chunk_duration: maxChunkDuration,
        force,
        only_speakers: onlySpeakers.length > 0 ? onlySpeakers : undefined,
        keep_segment_keys: Array.from(sourceSelection),
      }),
    onSuccess: (data) => {
      setGenerateTaskId(data.task_id);
      if (episode && folder) {
        setEpisodeTask(data.task_id, {
          stem: getEpisodeStem(episode),
          folder,
          title: episode.title,
          step: "synthesize",
        });
      }
    },
  });

  const assembleMutation = useMutation({
    mutationFn: () =>
      assembleEpisode({
        audio_path: audioPath ?? undefined,
        output_dir: outputDir ?? undefined,
        strategy: assembleStrategy,
        silence_duration: silenceDuration,
        language,
        model_size: modelSize,
        source_version_id: resolvedSource.sourceVersionId ?? undefined,
        keep_segment_keys: Array.from(sourceSelection),
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
    setEpisodeTask(null);
    setExpanded(true);
  };
  const handleDismiss = () => {
    setExtractTaskId(null);
    setGenerateTaskId(null);
    setEpisodeTask(null);
  };

  const { has: hasCap } = useCapabilities();
  const hasTTS = hasCap("tts") && hasCap("soundfile");

  if (!episode) return null;

  const prereq = !hasSourceRef
    ? "No episode source on disk yet."
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
            audioPath={audioPath}
            outputDir={outputDir}
            episode={episode}
            sourceVersionId={sourceVersionId}
            setSourceVersionId={setSourceVersionId}
            onResolvedSourceChange={setResolvedSource}
            onSegmentsChange={setSourceSegments}
            selectedKeys={sourceSelection}
            setSelectedKeys={setSourceSelection}
            selectionStampRef={sourceSelectionStampRef}
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
            audioPath={audioPath}
            noAudio={noAudio}
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
            speakersMissingSamples={speakersMissingSamples}
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
            missingGeneratedCount={missingGeneratedCount}
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
          onComplete={() => {
            refreshQueries();
            setGenerateTaskId(null);
            setEpisodeTask(null);
            // Transient run flags — clear so the next click doesn't silently
            // re-apply the previous run's overrides.
            setForce(false);
            setOnlySpeakers([]);
          }}
          onRetry={() => { setGenerateTaskId(null); setEpisodeTask(null); generateMutation.mutate(); }}
          onDismiss={() => { cancelTask(generateTaskId).catch(() => {}); setGenerateTaskId(null); setEpisodeTask(null); }}
          onCancel={() => { cancelTask(generateTaskId).catch(() => {}); setGenerateTaskId(null); setEpisodeTask(null); }}
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
                {synthVersionOptions.map(({ id, label }) => (
                  <option key={id} value={id}>{label}</option>
                ))}
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
                  const v = await getSynthesizeVersionPath(audioPath, activeSynthVersion.id, outputDir);
                  synthPath = v.path;
                } catch {
                  return;
                }
              } else {
                synthPath = assembleMutation.data?.path ?? null;
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
