import { useState, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEpisodeStore, usePipelineConfigStore } from "@/stores";
import {
  getSegments,
  getSegmentsRaw,
  getPipelineConfig,
  getTranscribeVersionInfo,
  getTranscribeVersions,
  loadTranscribeVersion,
  saveSegments,
  startTranscribe,
  uploadTranscript,
} from "@/api/client";
import { Button } from "@/components/ui/button";
import { Upload } from "lucide-react";
import { errorMessage, languageToISO, selectClass } from "@/lib/utils";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { useCapabilities } from "@/hooks/useCapabilities";
import AdvancedToggle from "@/components/common/AdvancedToggle";
import HelpLabel from "@/components/common/HelpLabel";
import MissingDependency from "@/components/common/MissingDependency";
import SectionHeader from "@/components/common/SectionHeader";
import SegmentEditor from "@/components/editor/SegmentEditor";
import PipelinePanel from "@/components/common/PipelinePanel";
import SpeakerMapEditor from "./SpeakerMapEditor";

export default function TranscribePanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  if (!episode) return null;
  const queryClient = useQueryClient();
  const { has: hasCap } = useCapabilities();
  const hasWhisperX = hasCap("whisperx");
  const task = usePipelineTask(episode.audio_path, "transcribe");
  const expanded = task.expanded || !episode.transcribed;
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data: pipelineConfig } = useQuery({
    queryKey: ["pipeline-config"],
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  // Form state — shared with batch modal via store
  const tc = usePipelineConfigStore((s) => s.transcribe);
  const setTc = usePipelineConfigStore((s) => s.setTranscribe);
  const [language, setLanguage] = useState(languageToISO(showMeta?.language || ""));

  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadTranscript(episode.audio_path!, file),
    onSuccess: () => {
      task.refreshQueries();
      task.setExpanded(false);
    },
  });

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) uploadMutation.mutate(file);
    e.target.value = "";
  };

  const startMutation = useMutation({
    mutationFn: () =>
      startTranscribe({
        audio_path: episode.audio_path!,
        model_size: tc.modelSize,
        language: language || undefined,
        batch_size: tc.batchSize,
        force: episode.transcribed,
        diarize: tc.diarize,
        hf_token: tc.hfToken || undefined,
        num_speakers: tc.numSpeakers ? Number(tc.numSpeakers) : undefined,
        show: showMeta?.name || "",
        episode: episode.title,
      }),
    onSuccess: (data) => task.startTask(data.task_id),
  });

  return (
    <PipelinePanel
      title="Transcribe"
      description="Turn audio into text and identify who is speaking."
      prerequisite={!episode.audio_path ? "Download the audio file first before transcribing." : undefined}
      done={episode.transcribed}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel="Re-run transcription"
      settingsLabel="Transcription settings"
      taskId={task.activeTaskId}
      onTaskComplete={task.handleComplete}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="No transcript yet. Run the transcription pipeline to get started."
      controls={
        <>
          <TranscribeForm
            modelSize={tc.modelSize} setModelSize={(v) => setTc({ modelSize: v })}
            language={language} setLanguage={setLanguage}
            batchSize={tc.batchSize} setBatchSize={(v) => setTc({ batchSize: v })}
            diarize={tc.diarize} setDiarize={(v) => setTc({ diarize: v })}
            hfToken={tc.hfToken} setHfToken={(v) => setTc({ hfToken: v })}
            numSpeakers={tc.numSpeakers} setNumSpeakers={(v) => setTc({ numSpeakers: v })}
            whisperModels={pipelineConfig?.whisper_models}
            hasWhisperX={hasWhisperX}
            onRun={() => startMutation.mutate()}
            onUpload={() => fileInputRef.current?.click()}
            isPending={startMutation.isPending}
            isUploading={uploadMutation.isPending}
            error={startMutation.isError ? errorMessage(startMutation.error) : uploadMutation.isError ? errorMessage(uploadMutation.error) : null}
            showOverwriteWarning={episode.transcribed}
          />
          <input ref={fileInputRef} type="file" accept=".json" onChange={handleFileUpload} className="hidden" />
        </>
      }
    >
      {/* Speaker map */}
      {episode.transcribed && !task.activeTaskId && (
        <SpeakerMapEditor
          audioPath={episode.audio_path}
          onSaved={() => {
            queryClient.invalidateQueries({ queryKey: ["transcribe", "segments", episode.audio_path] });
          }}
        />
      )}

      {/* Segment editor */}
      {episode.transcribed && !task.activeTaskId && (
        <TranscribeEditor audioPath={episode.audio_path!} duration={episode.duration} speakers={showMeta?.speakers} />
      )}
    </PipelinePanel>
  );
}

function TranscribeForm({
  modelSize, setModelSize,
  language, setLanguage,
  batchSize, setBatchSize,
  diarize, setDiarize,
  hfToken, setHfToken,
  numSpeakers, setNumSpeakers,
  whisperModels,
  hasWhisperX,
  onRun, onUpload, isPending, isUploading, error, showOverwriteWarning,
}: {
  modelSize: string; setModelSize: (v: string) => void;
  whisperModels?: Record<string, string>;
  hasWhisperX: boolean;
  language: string; setLanguage: (v: string) => void;
  batchSize: number; setBatchSize: (v: number) => void;
  diarize: boolean; setDiarize: (v: boolean) => void;
  hfToken: string; setHfToken: (v: string) => void;
  numSpeakers: string; setNumSpeakers: (v: string) => void;
  onRun: () => void; onUpload: () => void;
  isPending: boolean; isUploading: boolean; error: string | null;
  showOverwriteWarning: boolean;
}) {
  return (
    <div className="px-4 pb-3 space-y-4">
      {!hasWhisperX && (
        <MissingDependency
          extra="pipeline"
          label="WhisperX"
          description="Required for automatic transcription. You can still upload a transcript file manually."
        />
      )}

      <div className="flex flex-col lg:flex-row gap-6">
        <div className="space-y-3 flex-1">
          <SectionHeader>Transcription</SectionHeader>
          <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 text-sm items-start sm:items-center">
            <HelpLabel label="Model" help="The speech recognition model. Bigger models understand speech better (fewer mistakes) but take longer and need a more powerful GPU." />
            <select
              value={modelSize}
              onChange={(e) => setModelSize(e.target.value)}
              className={selectClass}
            >
              {whisperModels
                ? Object.entries(whisperModels).map(([key, desc]) => (
                    <option key={key} value={key} title={desc}>{key} — {desc}</option>
                  ))
                : <option value={modelSize}>{modelSize}</option>
              }
            </select>

            <HelpLabel label="Language" help="ISO code of the spoken language (e.g. fr, en, de). Leave empty to auto-detect. Setting it improves accuracy and enables word-level alignment." />
            <input
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              placeholder="e.g. fr, en, de"
              className="input py-1 text-sm"
            />
          </div>
        </div>

        <div className="space-y-3 flex-1 lg:border-l lg:border-border lg:pl-6">
          <div className="flex items-center gap-3">
            <SectionHeader>Speaker identification</SectionHeader>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={diarize}
                onChange={(e) => setDiarize(e.target.checked)}
                className="accent-primary"
              />
              <span className="text-xs text-muted-foreground">Enabled</span>
            </label>
          </div>

          {diarize && (
            <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 text-sm items-start sm:items-center">
              <HelpLabel label="Num speakers" help="How many different people are talking (e.g. 2 for an interview). Leave empty and it will guess. Filling this in helps it tell speakers apart more reliably." />
              <input
                value={numSpeakers}
                onChange={(e) => setNumSpeakers(e.target.value)}
                placeholder="auto-detect"
                className="input py-1 text-sm w-20"
              />
            </div>
          )}
        </div>
      </div>

      <AdvancedToggle className="border-t border-border/50 pt-3 space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm pl-3 border-l-2 border-border">
          <HelpLabel label="Batch size" help="How many audio chunks are processed at the same time. Higher = faster but uses more GPU memory. Lower this if you get out-of-memory errors." />
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
            min={1}
            className="input py-1 text-sm w-20"
          />

          {diarize && (
            <>
              <HelpLabel label="HF token" help="A HuggingFace access token, needed to download the speaker detection model. Get one free at huggingface.co/settings/tokens. If left empty, it looks for a HF_TOKEN environment variable." />
              <input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="from env if empty"
                className="input py-1 text-sm"
              />
            </>
          )}
        </div>
      </AdvancedToggle>

      <div className="flex items-center gap-3 border-t border-border/50 pt-3">
        <Button onClick={onRun} disabled={isPending || isUploading || !hasWhisperX} size="sm" title={!hasWhisperX ? "Install the pipeline extra to enable automatic transcription" : undefined}>
          {isPending ? "Starting..." : "Run"}
        </Button>
        <span className="text-xs text-muted-foreground">or</span>
        <Button onClick={onUpload} disabled={isUploading || isPending} variant="outline" size="sm">
          <Upload className="w-3.5 h-3.5 mr-1.5" />
          {isUploading ? "Uploading..." : "Upload file"}
        </Button>
      </div>

      {showOverwriteWarning && (
        <p className="text-xs text-muted-foreground">This will overwrite the existing transcript.</p>
      )}
      {error && <p className="text-destructive text-xs">{error}</p>}
    </div>
  );
}

function TranscribeEditor({ audioPath, duration, speakers }: { audioPath: string; duration: number; speakers?: string[] }) {
  const { data: referenceSegments } = useQuery({
    queryKey: ["transcribe", "segments-raw", audioPath],
    queryFn: () => getSegmentsRaw(audioPath),
    enabled: !!audioPath,
  });

  return (
    <SegmentEditor
      editorKey="transcribe"
      audioPath={audioPath}
      episodeDuration={duration}
      loadSegments={() => getSegments(audioPath)}
      loadRawSegments={() => getSegmentsRaw(audioPath)}
      loadVersionInfo={() => getTranscribeVersionInfo(audioPath)}
      saveSegments={(segs) => saveSegments(audioPath, segs)}
      showDelete
      showFlags
      showSpeaker
      referenceSegments={referenceSegments}
      referenceLabel="Original"
      speakers={speakers}
      loadVersions={() => getTranscribeVersions(audioPath)}
      loadVersion={(id) => loadTranscribeVersion(audioPath, id)}
    />
  );
}
