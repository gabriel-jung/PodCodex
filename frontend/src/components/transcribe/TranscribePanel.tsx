import { useState, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { useEpisodeStore, useAudioPath, usePipelineConfigStore } from "@/stores";
import {
  deleteTranscribeVersion,
  getSegments,
  getTranscribeVersions,
  importTranscript,
  loadTranscribeVersion,
  saveSegments,
  startTranscribe,
  uploadTranscript,
} from "@/api/client";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { Button } from "@/components/ui/button";
import { FileText, Upload } from "lucide-react";
import { errorMessage, languageToISO, selectClass } from "@/lib/utils";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { useCapabilities } from "@/hooks/useCapabilities";
import AdvancedToggle from "@/components/common/AdvancedToggle";
import FormGrid from "@/components/common/FormGrid";
import HelpLabel from "@/components/common/HelpLabel";
import MissingDependency from "@/components/common/MissingDependency";
import SectionHeader from "@/components/common/SectionHeader";
import TranscriptViewer from "@/components/editor/TranscriptViewer";
import PipelinePanel from "@/components/common/PipelinePanel";

export default function TranscribePanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const folder = useEpisodeStore((s) => s.folder);
  const audioPath = useAudioPath();
  if (!episode) return null;
  const hasRealAudio = !!episode.audio_path;

  const { has: hasCap } = useCapabilities();
  const hasWhisperX = hasCap("whisperx");
  const task = usePipelineTask(audioPath, "transcribe");
  const expanded = task.expanded || !episode.transcribed;
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { whisperModels: whisperModelsMap } = useLLMProviders();

  // Form state
  const tc = usePipelineConfigStore((s) => s.transcribe);
  const setTc = usePipelineConfigStore((s) => s.setTranscribe);
  const [language, setLanguage] = useState(languageToISO(showMeta?.language || ""));

  // Existing subtitle files for reimport controls
  const subtitleFiles = (episode.files ?? []).filter(
    (f) => f.endsWith(".vtt") || f.endsWith(".srt"),
  );

  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadTranscript(audioPath!, file),
    onSuccess: () => {
      task.refreshQueries();
      task.setExpanded(false);
    },
  });

  const importFileMutation = useMutation({
    mutationFn: (filePath: string) => importTranscript(audioPath!, filePath),
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
        audio_path: audioPath!,
        model_size: tc.modelSize,
        language: language || undefined,
        batch_size: tc.batchSize,
        force: episode.transcribed,
        diarize: tc.diarize,
        clean: tc.clean,
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
      description="Transcribe audio or import subtitles."
      prerequisite={!audioPath ? "Download the audio file or import subtitles first." : undefined}
      done={episode.transcribed}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel={hasRealAudio ? "Re-run transcription" : "Reimport transcript"}
      settingsLabel={hasRealAudio ? "Transcription settings" : "Import transcript"}
      taskId={task.activeTaskId}
      onTaskComplete={() => { task.handleComplete(); }}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="No transcript yet."
      controls={
        <>
          {hasRealAudio ? (
            <TranscribeForm
              modelSize={tc.modelSize} setModelSize={(v) => setTc({ modelSize: v })}
              language={language} setLanguage={setLanguage}
              batchSize={tc.batchSize} setBatchSize={(v) => setTc({ batchSize: v })}
              diarize={tc.diarize} setDiarize={(v) => setTc({ diarize: v })}
              clean={tc.clean} setClean={(v) => setTc({ clean: v })}
              hfToken={tc.hfToken} setHfToken={(v) => setTc({ hfToken: v })}
              numSpeakers={tc.numSpeakers} setNumSpeakers={(v) => setTc({ numSpeakers: v })}
              whisperModels={whisperModelsMap}
              hasWhisperX={hasWhisperX}
              hasAudio={hasRealAudio}
              onRun={() => startMutation.mutate()}
              onUpload={() => fileInputRef.current?.click()}
              isPending={startMutation.isPending}
              isUploading={uploadMutation.isPending}
              error={startMutation.isError ? errorMessage(startMutation.error) : uploadMutation.isError ? errorMessage(uploadMutation.error) : null}
              showOverwriteWarning={episode.transcribed}
            />
          ) : (
            <div className="px-4 pb-3 space-y-3">
              {subtitleFiles.length > 0 ? (
                <div className="space-y-1.5">
                  <p className="text-xs text-muted-foreground">Import from existing file:</p>
                  <div className="flex flex-wrap gap-1.5">
                    {subtitleFiles.map((f) => (
                      <Button
                        key={f}
                        variant="outline"
                        size="sm"
                        disabled={importFileMutation.isPending}
                        onClick={() => importFileMutation.mutate(`${folder}/${f}`)}
                      >
                        <FileText className="w-3.5 h-3.5 mr-1" />
                        {f.split("/").pop()}
                      </Button>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  No audio file. Upload a transcript or subtitle file (JSON, SRT, VTT).
                </p>
              )}
              <div className="flex items-center gap-3">
                {subtitleFiles.length > 0 && <span className="text-xs text-muted-foreground">or</span>}
                <Button onClick={() => fileInputRef.current?.click()} disabled={uploadMutation.isPending} variant="outline" size="sm">
                  <Upload className="w-3.5 h-3.5 mr-1.5" />
                  {uploadMutation.isPending ? "Uploading..." : "Upload file"}
                </Button>
              </div>
              {(uploadMutation.isError || importFileMutation.isError) && (
                <p className="text-destructive text-xs">{errorMessage(uploadMutation.error || importFileMutation.error)}</p>
              )}
            </div>
          )}
          <input ref={fileInputRef} type="file" accept=".json,.srt,.vtt" onChange={handleFileUpload} className="hidden" />
        </>
      }
    >
      {episode.transcribed && !task.activeTaskId && (
        <TranscriptViewer
          editorKey="transcribe"
          audioPath={audioPath ?? undefined}
          episodeDuration={episode.duration}
          loadSegments={() => getSegments(audioPath!)}
          saveSegments={(segs) => saveSegments(audioPath!, segs)}
          showDelete
          showFlags
          showSpeaker
          speakers={showMeta?.speakers}
          sourceLabel={transcriptSourceLabel(episode.provenance)}
          exportSource="transcript"
          loadVersions={() => getTranscribeVersions(audioPath!)}
          loadVersion={(id) => loadTranscribeVersion(audioPath!, id)}
          deleteVersion={(id) => deleteTranscribeVersion(audioPath!, id)}
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
  clean, setClean,
  hfToken, setHfToken,
  numSpeakers, setNumSpeakers,
  whisperModels,
  hasWhisperX,
  hasAudio,
  onRun, onUpload, isPending, isUploading, error, showOverwriteWarning,
}: {
  modelSize: string; setModelSize: (v: string) => void;
  whisperModels?: Record<string, string>;
  hasWhisperX: boolean;
  hasAudio: boolean;
  language: string; setLanguage: (v: string) => void;
  batchSize: number; setBatchSize: (v: number) => void;
  diarize: boolean; setDiarize: (v: boolean) => void;
  clean: boolean; setClean: (v: boolean) => void;
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
          <FormGrid>
            <HelpLabel label="Model" help="Speech recognition model. Bigger models make fewer mistakes but are slower and need more GPU memory." />
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

            <HelpLabel label="Language" help="ISO code of the spoken language (e.g. fr, en, de). Leave empty to auto-detect. Setting it improves accuracy and word-level alignment." />
            <input
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              placeholder="e.g. fr, en, de"
              className="input py-1 text-sm"
            />
          </FormGrid>
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
            <FormGrid>
              <HelpLabel label="Num speakers" help="How many speakers are in the episode (e.g. 2 for an interview). Leave empty to auto-detect. Setting it helps tell speakers apart more reliably." />
              <input
                value={numSpeakers}
                onChange={(e) => setNumSpeakers(e.target.value)}
                placeholder="auto-detect"
                className="input py-1 text-sm w-20"
              />
            </FormGrid>
          )}
        </div>
      </div>

      <AdvancedToggle className="border-t border-border/50 pt-3 space-y-3">
        <FormGrid className="pl-3 border-l-2 border-border">
          <HelpLabel label="Clean transcript" help="Automatically remove hallucinated segments (low speech density) and unknown speakers from the transcript." />
          <input
            type="checkbox"
            checked={clean}
            onChange={(e) => setClean(e.target.checked)}
            className="accent-primary"
          />

          <HelpLabel label="Batch size" help="How many audio chunks to process in parallel. Higher is faster but uses more GPU memory. Lower this if you run out of memory." />
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
            min={1}
            className="input py-1 text-sm w-20"
          />

          {diarize && (
            <>
              <HelpLabel label="HF token" help="HuggingFace access token, needed to download the speaker detection model. Get one free at huggingface.co/settings/tokens. Falls back to the HF_TOKEN environment variable." />
              <input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="from env if empty"
                className="input py-1 text-sm"
              />
            </>
          )}
        </FormGrid>
      </AdvancedToggle>

      <div className="flex items-center gap-3 border-t border-border/50 pt-3">
        <Button onClick={onRun} disabled={isPending || isUploading || !hasWhisperX || !hasAudio} size="sm" title={!hasAudio ? "Download the audio file first" : !hasWhisperX ? "Install the pipeline extra to enable automatic transcription" : undefined}>
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

function transcriptSourceLabel(provenance?: Record<string, unknown>): string | undefined {
  const prov = provenance?.transcript as Record<string, unknown> | undefined;
  if (!prov) return undefined;
  const params = prov.params as Record<string, unknown> | undefined;
  if (params?.filename) return String(params.filename);
  if (params?.source === "youtube-subtitles") return `YouTube subtitles (${params.lang || "auto"})`;
  if (prov.model) return `whisper ${prov.model}`;
  return undefined;
}
