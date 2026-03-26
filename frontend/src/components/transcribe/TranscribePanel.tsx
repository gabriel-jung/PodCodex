import { useState, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Episode, ShowMeta } from "@/api/types";
import {
  getSegments,
  getSegmentsRaw,
  getPipelineConfig,
  getTranscribeVersionInfo,
  saveSegments,
  startTranscribe,
  uploadTranscript,
} from "@/api/client";
import { Button } from "@/components/ui/button";
import { Upload, Settings2 } from "lucide-react";
import HelpLabel from "@/components/common/HelpLabel";
import SegmentEditor from "@/components/editor/SegmentEditor";
import PipelinePanel from "@/components/common/PipelinePanel";
import SpeakerMapEditor from "./SpeakerMapEditor";

interface TranscribePanelProps {
  episode: Episode;
  showMeta?: ShowMeta | null;
}

export default function TranscribePanel({ episode, showMeta }: TranscribePanelProps) {
  const queryClient = useQueryClient();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(!episode.transcribed);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data: pipelineConfig } = useQuery({
    queryKey: ["pipeline-config"],
    queryFn: getPipelineConfig,
    staleTime: Infinity,
  });

  // Form state
  const [modelSize, setModelSize] = useState("large-v3");
  const [language, setLanguage] = useState(showMeta?.language || "");
  const [batchSize, setBatchSize] = useState(16);
  const [diarize, setDiarize] = useState(true);
  const [hfToken, setHfToken] = useState("");
  const [numSpeakers, setNumSpeakers] = useState<string>("");

  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadTranscript(episode.audio_path!, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["transcribe"] });
      queryClient.invalidateQueries({ queryKey: ["episodes"] });
      setExpanded(false);
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
        model_size: modelSize,
        language: language || undefined,
        batch_size: batchSize,
        force: episode.transcribed,
        diarize,
        hf_token: hfToken || undefined,
        num_speakers: numSpeakers ? Number(numSpeakers) : undefined,
        show: showMeta?.name || "",
        episode: episode.title,
      }),
    onSuccess: (data) => setTaskId(data.task_id),
  });

  const handleComplete = () => {
    queryClient.invalidateQueries({ queryKey: ["transcribe"] });
    queryClient.invalidateQueries({ queryKey: ["episodes"] });
    setTaskId(null);
    setExpanded(false);
  };

  return (
    <PipelinePanel
      title="Transcribe"
      description="Turn audio into text and identify who is speaking."
      prerequisite={!episode.audio_path ? "Download the audio file first before transcribing." : undefined}
      done={episode.transcribed}
      expanded={expanded}
      onToggle={() => setExpanded(!expanded)}
      rerunLabel="Re-run transcription"
      settingsLabel="Transcription settings"
      taskId={taskId}
      onTaskComplete={handleComplete}
      emptyMessage="No transcript yet. Run the transcription pipeline to get started."
      controls={
        <>
          <TranscribeForm
            modelSize={modelSize} setModelSize={setModelSize}
            language={language} setLanguage={setLanguage}
            batchSize={batchSize} setBatchSize={setBatchSize}
            diarize={diarize} setDiarize={setDiarize}
            hfToken={hfToken} setHfToken={setHfToken}
            numSpeakers={numSpeakers} setNumSpeakers={setNumSpeakers}
            whisperModels={pipelineConfig?.whisper_models}
            onRun={() => startMutation.mutate()}
            onUpload={() => fileInputRef.current?.click()}
            isPending={startMutation.isPending}
            isUploading={uploadMutation.isPending}
            error={startMutation.isError ? (startMutation.error as Error).message : uploadMutation.isError ? (uploadMutation.error as Error).message : null}
            showOverwriteWarning={episode.transcribed}
          />
          <input ref={fileInputRef} type="file" accept=".json" onChange={handleFileUpload} className="hidden" />
        </>
      }
    >
      {/* Speaker map */}
      {episode.transcribed && !taskId && (
        <SpeakerMapEditor
          audioPath={episode.audio_path}
          onSaved={() => {
            queryClient.invalidateQueries({ queryKey: ["transcribe", "segments", episode.audio_path] });
          }}
        />
      )}

      {/* Segment editor */}
      {episode.transcribed && !taskId && (
        <SegmentEditor
          editorKey="transcribe"
          audioPath={episode.audio_path}
          episodeDuration={episode.duration}
          loadSegments={() => getSegments(episode.audio_path!)}
          loadRawSegments={() => getSegmentsRaw(episode.audio_path!)}
          loadVersionInfo={() => getTranscribeVersionInfo(episode.audio_path!)}
          saveSegments={(segs) => saveSegments(episode.audio_path!, segs)}
          showDelete
          showFlags
          showSpeaker
          speakers={showMeta?.speakers}
        />
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
  onRun, onUpload, isPending, isUploading, error, showOverwriteWarning,
}: {
  modelSize: string; setModelSize: (v: string) => void;
  whisperModels?: Record<string, string>;
  language: string; setLanguage: (v: string) => void;
  batchSize: number; setBatchSize: (v: number) => void;
  diarize: boolean; setDiarize: (v: boolean) => void;
  hfToken: string; setHfToken: (v: string) => void;
  numSpeakers: string; setNumSpeakers: (v: string) => void;
  onRun: () => void; onUpload: () => void;
  isPending: boolean; isUploading: boolean; error: string | null;
  showOverwriteWarning: boolean;
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <div className="px-4 pb-3 space-y-4">
      {/* ── Transcription + Speaker identification — side by side on wide screens ── */}
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left: Transcription */}
        <div className="space-y-3 flex-1">
          <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Transcription</h5>
          <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 text-sm items-center">
            <HelpLabel label="Model" help="The speech recognition model. Bigger models understand speech better (fewer mistakes) but take longer and need a more powerful GPU." />
            <select
              value={modelSize}
              onChange={(e) => setModelSize(e.target.value)}
              className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
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

        {/* Right: Speaker identification */}
        <div className="space-y-3 flex-1 lg:border-l lg:border-border lg:pl-6">
          <div className="flex items-center gap-3">
            <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Speaker identification</h5>
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
            <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 text-sm items-center">
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

      {/* ── Advanced ── */}
      <div className="border-t border-border/50 pt-3 space-y-3">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
        >
          <Settings2 className="w-3 h-3" />
          <span className="font-semibold uppercase tracking-wide">{showAdvanced ? "Hide advanced" : "Advanced settings"}</span>
        </button>

        {showAdvanced && (
          <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 items-center text-sm pl-3 border-l-2 border-border">
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
        )}
      </div>

      {/* ── Actions ── */}
      <div className="flex items-center gap-3 border-t border-border/50 pt-3">
        <Button onClick={onRun} disabled={isPending || isUploading} size="sm">
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
