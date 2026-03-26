import { useState, useRef } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import type { Episode, ShowMeta } from "@/api/types";
import {
  getSegments,
  getSegmentsRaw,
  getTranscribeVersionInfo,
  saveSegments,
  startTranscribe,
  uploadTranscript,
} from "@/api/client";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronRight, Upload, Settings2 } from "lucide-react";
import HelpLabel from "@/components/common/HelpLabel";
import SegmentEditor from "@/components/editor/SegmentEditor";
import ProgressBar from "@/components/editor/ProgressBar";
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

  if (!episode.audio_path) {
    return (
      <div className="p-6 text-muted-foreground">
        Episode not downloaded yet. Download the audio file to start transcription.
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Step header */}
      <div className="px-4 pt-3 pb-2 border-b border-border">
        <h3 className="text-sm font-semibold">Transcribe</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Convert audio to text using Whisper with optional speaker diarization.
        </p>
      </div>

      {/* Transcription controls — collapsible when already transcribed */}
      {!taskId && (
        <div className="border-b border-border">
          {episode.transcribed ? (
            <button
              onClick={() => setExpanded(!expanded)}
              className="w-full px-4 py-2 flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition"
            >
              {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
              <span className="font-medium">Re-run transcription</span>
            </button>
          ) : (
            <div className="px-4 pt-3 pb-1">
              <h4 className="text-sm font-medium">Transcription settings</h4>
            </div>
          )}

          {expanded && (
            <TranscribeForm
              modelSize={modelSize} setModelSize={setModelSize}
              language={language} setLanguage={setLanguage}
              batchSize={batchSize} setBatchSize={setBatchSize}
              diarize={diarize} setDiarize={setDiarize}
              hfToken={hfToken} setHfToken={setHfToken}
              numSpeakers={numSpeakers} setNumSpeakers={setNumSpeakers}
              onRun={() => startMutation.mutate()}
              onUpload={() => fileInputRef.current?.click()}
              isPending={startMutation.isPending}
              isUploading={uploadMutation.isPending}
              error={startMutation.isError ? (startMutation.error as Error).message : uploadMutation.isError ? (uploadMutation.error as Error).message : null}
              showOverwriteWarning={episode.transcribed}
            />
          )}
          <input ref={fileInputRef} type="file" accept=".json" onChange={handleFileUpload} className="hidden" />
        </div>
      )}

      {/* Progress */}
      {taskId && <ProgressBar taskId={taskId} onComplete={handleComplete} />}

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

      {/* Not transcribed yet, controls collapsed */}
      {!episode.transcribed && !expanded && !taskId && (
        <div className="p-6 text-muted-foreground">
          No transcript yet. Run the transcription pipeline to get started.
        </div>
      )}
    </div>
  );
}

function TranscribeForm({
  modelSize, setModelSize,
  language, setLanguage,
  batchSize, setBatchSize,
  diarize, setDiarize,
  hfToken, setHfToken,
  numSpeakers, setNumSpeakers,
  onRun, onUpload, isPending, isUploading, error, showOverwriteWarning,
}: {
  modelSize: string; setModelSize: (v: string) => void;
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
      {/* ── Transcription ── */}
      <div className="space-y-3">
        <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Transcription</h5>
        <div className="grid grid-cols-2 gap-3 max-w-lg text-sm">
          <HelpLabel label="Model" help="Whisper model size. Larger models are more accurate but slower. large-v3 is recommended for best quality." />
          <select
            value={modelSize}
            onChange={(e) => setModelSize(e.target.value)}
            className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
          >
            {["large-v3", "medium", "small", "base", "tiny"].map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>

          <HelpLabel label="Language" help="Audio language. Leave empty to auto-detect. Specifying it improves accuracy and speed." />
          <input
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            placeholder="auto-detect"
            className="input py-1 text-sm"
          />
        </div>
      </div>

      {/* ── Speaker diarization ── */}
      <div className="space-y-3 border-t border-border/50 pt-3">
        <div className="flex items-center gap-3">
          <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Speaker diarization</h5>
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
          <div className="grid grid-cols-2 gap-3 max-w-lg text-sm">
            <HelpLabel label="Num speakers" help="Expected number of speakers. Leave empty to auto-detect. Setting this improves diarization when you know the count." />
            <input
              value={numSpeakers}
              onChange={(e) => setNumSpeakers(e.target.value)}
              placeholder="auto-detect"
              className="input py-1 text-sm w-20"
            />
          </div>
        )}
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
          <div className="grid grid-cols-2 gap-3 max-w-lg text-sm pl-3 border-l-2 border-border">
            <HelpLabel label="Batch size" help="Number of audio chunks processed in parallel. Higher values use more VRAM but are faster. Reduce if you get out-of-memory errors." />
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              min={1}
              className="input py-1 text-sm w-20"
            />

            {diarize && (
              <>
                <HelpLabel label="HF token" help="HuggingFace API token for pyannote diarization models. Uses the HF_TOKEN environment variable if left empty." />
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
