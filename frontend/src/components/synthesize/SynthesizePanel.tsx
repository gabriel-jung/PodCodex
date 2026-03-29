import { useState, useCallback, useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Segment } from "@/api/types";
import { useEpisodeStore } from "@/stores";
import {
  getSynthesisStatus,
  getVoiceSamples,
  extractSelectedSamples,
  uploadVoiceSample,
  startGenerateTTS,
  getGeneratedSegments,
  assembleEpisode,
  audioFileUrl,
  getPipelineConfig,
  getSegments,
  getPolishSegments,
} from "@/api/client";
import { Button } from "@/components/ui/button";
import { Play, Check, Upload } from "lucide-react";
import { useCapabilities } from "@/hooks/useCapabilities";
import AdvancedToggle from "@/components/common/AdvancedToggle";
import HelpLabel from "@/components/common/HelpLabel";
import MissingDependency from "@/components/common/MissingDependency";
import SectionHeader from "@/components/common/SectionHeader";
import ProgressBar from "@/components/editor/ProgressBar";
import PipelinePanel from "@/components/common/PipelinePanel";
import { useAudioStore } from "@/stores";
import { errorMessage, selectClass } from "@/lib/utils";

/** Key for a segment selection — "speaker:start:end" */
function segKey(seg: { speaker?: string; start: number; end: number }) {
  return `${seg.speaker || ""}:${seg.start}:${seg.end}`;
}

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
    queryKey: ["pipeline-config"],
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  const { data: status, refetch: refetchStatus } = useQuery({
    queryKey: ["synthesize", "status", episode.audio_path],
    queryFn: () => getSynthesisStatus(episode.audio_path!),
    enabled: !!episode.audio_path,
  });

  // Load transcript segments for speaker browsing
  const { data: transcriptSegments } = useQuery({
    queryKey: ["synth-source-segments", episode.audio_path],
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
          {/* ── Step 1: Voice extraction ────────── */}
          <section className="space-y-3">
            <SectionHeader>1. Select Voice Samples</SectionHeader>
            <p className="text-xs text-muted-foreground">
              Pick the best segments for each speaker to use as voice cloning references.
              Play segments to audition them, check the ones you want, then extract.
            </p>

            {/* Time range filter */}
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Focus:</span>
              <input
                value={timeFrom}
                onChange={(e) => setTimeFrom(e.target.value)}
                placeholder="from (mm:ss)"
                className="input py-0.5 text-xs w-24"
              />
              <span className="text-muted-foreground">to</span>
              <input
                value={timeTo}
                onChange={(e) => setTimeTo(e.target.value)}
                placeholder="to (mm:ss)"
                className="input py-0.5 text-xs w-24"
              />
              {(timeFrom || timeTo) && (
                <button
                  onClick={() => { setTimeFrom(""); setTimeTo(""); }}
                  className="text-muted-foreground hover:text-foreground"
                >
                  Clear
                </button>
              )}
            </div>

            {Object.keys(segmentsBySpeaker).length > 0 ? (
              <div className="space-y-3">
                {Object.entries(segmentsBySpeaker).map(([speaker, segs]) => {
                  const speakerSelected = segs.filter((s) => selected.has(segKey(s))).length;
                  return (
                    <div key={speaker} className="text-sm">
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="font-medium">{speaker}</span>
                        <span className="text-muted-foreground text-xs">
                          {segs.length} segment{segs.length !== 1 ? "s" : ""}
                          {speakerSelected > 0 && ` · ${speakerSelected} selected`}
                        </span>
                      </div>
                      <div className="flex flex-col gap-1 max-h-64 overflow-y-auto">
                        {(() => {
                          const filtered = segs
                            .filter((s) => (s.end - s.start) >= 2)
                            .sort((a, b) => (b.end - b.start) - (a.end - a.start));
                          const limit = showCount[speaker] || 10;
                          const visible = filtered.slice(0, limit);
                          const hasMore = filtered.length > limit;
                          return (
                            <>
                              {visible.map((seg) => {
                                const key = segKey(seg);
                                const isSelected = selected.has(key);
                                const isExpanded = expandedSeg === key;
                                const dur = seg.end - seg.start;
                                return (
                                  <div
                                    key={key}
                                    className={`flex flex-col rounded text-xs transition cursor-pointer ${
                                      isSelected
                                        ? "bg-primary/15 border border-primary/30"
                                        : "bg-secondary/50 border border-transparent hover:bg-secondary"
                                    }`}
                                    onClick={() => {
                                      const next = new Set(selected);
                                      if (isSelected) next.delete(key);
                                      else next.add(key);
                                      setSelected(next);
                                      setExpandedSeg(isExpanded ? null : key);
                                    }}
                                  >
                                    <div className="flex items-center gap-2 px-2 py-1">
                                      <div className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 ${
                                        isSelected ? "bg-primary border-primary" : "border-border"
                                      }`}>
                                        {isSelected && <Check className="w-3 h-3 text-primary-foreground" />}
                                      </div>
                                      <button
                                        onClick={(e) => {
                                          e.stopPropagation();
                                          seekTo(episode.audio_path!, seg.start);
                                        }}
                                        className="shrink-0 p-0.5 rounded hover:bg-accent transition"
                                        title="Play this segment"
                                      >
                                        <Play className="w-3 h-3" />
                                      </button>
                                      <span className="text-muted-foreground tabular-nums shrink-0 w-12">{dur.toFixed(1)}s</span>
                                      <span className={isExpanded ? "flex-1" : "truncate flex-1"}>{seg.text}</span>
                                      {/* Speaker reassignment */}
                                      {allSpeakers.length > 1 && (
                                        <select
                                          value={speakerOverrides[key] || seg.speaker || ""}
                                          onClick={(e) => e.stopPropagation()}
                                          onChange={(e) => {
                                            e.stopPropagation();
                                            const next = { ...speakerOverrides };
                                            if (e.target.value === seg.speaker) delete next[key];
                                            else next[key] = e.target.value;
                                            setSpeakerOverrides(next);
                                          }}
                                          className="shrink-0 bg-transparent border border-border rounded px-1 py-0 text-[10px] w-20 text-muted-foreground"
                                          title="Reassign speaker"
                                        >
                                          {allSpeakers.map((sp) => (
                                            <option key={sp} value={sp}>{sp}</option>
                                          ))}
                                        </select>
                                      )}
                                    </div>
                                    {isExpanded && seg.text.length > 60 && (
                                      <div className="px-2 pb-1.5 pl-[4.5rem] text-muted-foreground leading-relaxed">
                                        {seg.text}
                                      </div>
                                    )}
                                  </div>
                                );
                              })}
                              {hasMore && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setShowCount({ ...showCount, [speaker]: limit + 20 });
                                  }}
                                  className="text-xs text-muted-foreground hover:text-foreground transition py-1 text-center"
                                >
                                  Show more ({filtered.length - limit} remaining)
                                </button>
                              )}
                            </>
                          );
                        })()}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground italic">No transcript segments available.</p>
            )}

            <div className="flex items-center gap-3">
              <Button
                onClick={() => extractMutation.mutate()}
                disabled={selected.size === 0 || extractMutation.isPending}
                size="sm"
              >
                {extractMutation.isPending
                  ? "Extracting..."
                  : `Extract ${selected.size} sample${selected.size !== 1 ? "s" : ""}`}
              </Button>
              {status?.voice_samples_extracted && (
                <span className="text-xs text-green-400">Samples on disk</span>
              )}
              {extractMutation.isError && (
                <span className="text-xs text-destructive">
                  {errorMessage(extractMutation.error)}
                </span>
              )}
            </div>

            {/* Extracted + uploaded samples */}
            <div className="space-y-2 border-t border-border/50 pt-3">
              <div className="flex items-center gap-3">
                <span className="text-xs text-muted-foreground font-medium">
                  {voiceSamples && Object.keys(voiceSamples).length > 0
                    ? "Extracted samples:"
                    : "No samples yet"}
                </span>
              </div>
              {voiceSamples && Object.entries(voiceSamples).map(([speaker, samples]) => (
                <div key={speaker} className="text-sm">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{speaker}</span>
                    <span className="text-muted-foreground text-xs">
                      ({samples.length} sample{samples.length !== 1 ? "s" : ""})
                    </span>
                    <label className="cursor-pointer text-muted-foreground hover:text-foreground transition" title={`Upload audio for ${speaker}`}>
                      <Upload className="w-3 h-3" />
                      <input
                        type="file"
                        accept="audio/*"
                        className="hidden"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) uploadMutation.mutate({ speaker, file });
                          e.target.value = "";
                        }}
                      />
                    </label>
                  </div>
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

              {/* Upload for a new/any speaker */}
              {allSpeakers.length > 0 && (
                <div className="flex items-center gap-2 text-xs pt-1">
                  <label className="flex items-center gap-1.5 cursor-pointer text-muted-foreground hover:text-foreground transition">
                    <Upload className="w-3.5 h-3.5" />
                    <span>Upload external sample</span>
                    <input
                      type="file"
                      accept="audio/*"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          const speaker = allSpeakers[0];
                          const choice = prompt(`Upload for which speaker?\n${allSpeakers.join(", ")}`, speaker);
                          if (choice && allSpeakers.includes(choice)) {
                            uploadMutation.mutate({ speaker: choice, file });
                          }
                        }
                        e.target.value = "";
                      }}
                    />
                  </label>
                  {uploadMutation.isPending && <span className="text-muted-foreground">Uploading...</span>}
                  {uploadMutation.isError && <span className="text-destructive">{errorMessage(uploadMutation.error)}</span>}
                </div>
              )}
            </div>
          </section>

          {/* ── Step 2: TTS generation ─────────── */}
          <section className="space-y-3 border-t border-border/50 pt-3">
            <SectionHeader>2. Generate TTS Segments</SectionHeader>

            <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm">
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
                className={selectClass}
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
                    className={selectClass}
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
            <AdvancedToggle className="space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm pl-3 border-l-2 border-border">
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
            </AdvancedToggle>

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
                  {errorMessage(generateMutation.error)}
                </span>
              )}
            </div>
          </section>

          {/* ── Step 3: Assembly ───────────────── */}
          <section className="space-y-3 border-t border-border/50 pt-3">
            <SectionHeader>3. Assemble Episode</SectionHeader>

            <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm">
              <HelpLabel label="Strategy" help="How to handle pauses between segments in the final audio." />
              <select
                value={assembleStrategy}
                onChange={(e) => setAssembleStrategy(e.target.value)}
                className={selectClass}
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
                  {errorMessage(assembleMutation.error)}
                </span>
              )}
            </div>
          </section>
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
