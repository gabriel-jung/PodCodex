import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useEpisodeStore, useAudioPath, usePipelineConfigStore, useTaskStore } from "@/stores";
import { useShowActions } from "@/hooks/useShowActions";
import { EmptyState } from "@/components/ui/empty-state";
import { TRANSCRIBE_PRESETS, CPU_MODELS, GPU_MODELS, CPU_LABELS, GPU_LABELS } from "@/stores/pipelineConfigStore";
import {
  deleteTranscribeVersion,
  getSegments,
  getTranscribeVersions,
  importTranscript,
  loadTranscribeVersion,
  saveSegments,
  saveSpeakerMap,
  startTranscribe,
  uploadTranscript,
} from "@/api/client";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { Button } from "@/components/ui/button";
import { FileAudio, FileText, Upload } from "lucide-react";
import { errorMessage, languageToISO, selectClass, SUB_LANGUAGES } from "@/lib/utils";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { useCapabilities } from "@/hooks/useCapabilities";
import FormGrid from "@/components/common/FormGrid";
import HelpLabel from "@/components/common/HelpLabel";
import MissingDependency from "@/components/common/MissingDependency";
import Segmented from "@/components/common/Segmented";
import TranscriptViewer from "@/components/editor/TranscriptViewer";
import PipelinePanel from "@/components/common/PipelinePanel";

// The top row of the Language chip rack — these are always visible; anything
// else falls under "Other" with an ISO-code input.
const TOP_LANGUAGES = SUB_LANGUAGES.slice(0, 5);

export default function TranscribePanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const folder = useEpisodeStore((s) => s.folder);
  const audioPath = useAudioPath();

  const { has: hasCap } = useCapabilities();
  const hasWhisperX = hasCap("whisperx");
  const task = usePipelineTask(audioPath, "transcribe");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { downloadMutation: episodeDownloadMutation } = useShowActions(folder ?? "", showMeta ?? undefined, { withSubs: false });
  const downloadTaskId = useTaskStore((s) => s.downloadTaskId);
  const downloadDisabled = episodeDownloadMutation.isPending || !!downloadTaskId;

  const { whisperModels: whisperModelsMap, detectedKeys } = useLLMProviders();
  const hfTokenDetected = !!detectedKeys.hf_token;

  // Form state
  const tc = usePipelineConfigStore((s) => s.transcribe);
  const setTc = usePipelineConfigStore((s) => s.setTranscribe);
  const transcribePreset = usePipelineConfigStore((s) => s.transcribePreset);
  const applyTranscribePreset = usePipelineConfigStore((s) => s.applyTranscribePreset);
  // Infer CPU mode from preset OR current model — drives model-list filtering
  // and hides the speaker identification column (pyannote needs GPU).
  const isCpu = transcribePreset === "cpu"
    || (transcribePreset === "" && CPU_MODELS.has(tc.modelSize));
  // Language state — chips for the top 5 common languages, "Other" chip
  // reveals an ISO-code input. `language` holds either "" (auto), a known
  // ISO code from the chip set, or "other" (meaning use customLang instead).
  const showLangISO = languageToISO(showMeta?.language || "");
  const showLangInTop = TOP_LANGUAGES.some((l) => l.code === showLangISO);
  const [language, setLanguage] = useState<string>(
    !showLangISO ? "" : showLangInTop ? showLangISO : "other",
  );
  const [customLang, setCustomLang] = useState(showLangInTop || !showLangISO ? "" : showLangISO);
  // Sync when showMeta loads after initial render (only if user hasn't touched it)
  const [prevShowLang, setPrevShowLang] = useState(showLangISO);
  if (showLangISO !== prevShowLang) {
    setPrevShowLang(showLangISO);
    if (!language && !customLang) {
      if (showLangInTop) setLanguage(showLangISO);
      else if (showLangISO) { setLanguage("other"); setCustomLang(showLangISO); }
    }
  }
  const effectiveLang = language === "other" ? customLang : language;

  // Existing subtitle files for reimport controls
  const subtitleFiles = (episode?.files ?? []).filter(
    (f) => f.endsWith(".vtt") || f.endsWith(".srt"),
  );
  const hasSubs = !!episode?.has_subtitles || subtitleFiles.length > 0;
  const hasRealAudio = !!episode?.audio_path;

  // Source toggle — answers "where does the transcript come from".
  // Audio = transcribe with WhisperX; Subtitles = reimport a .vtt/.srt already
  // in the episode folder (typically YouTube auto-subs, only shown if any
  // exist); Upload = upload any transcript file from disk.
  const [transcribeSource, setTranscribeSource] = useState<"audio" | "subtitles" | "upload">(
    hasRealAudio ? "audio" : hasSubs ? "subtitles" : "upload",
  );
  // Auto-switch to "audio" when it becomes available, unless the user
  // has already picked something else.
  const userPickedSourceRef = useRef(false);
  const pickSource = (v: "audio" | "subtitles" | "upload") => {
    userPickedSourceRef.current = true;
    setTranscribeSource(v);
  };
  useEffect(() => {
    if (userPickedSourceRef.current) return;
    setTranscribeSource(hasRealAudio ? "audio" : hasSubs ? "subtitles" : "upload");
  }, [hasRealAudio, hasSubs]);

  // Per-episode cleanup mode — defaults to Manual because the user is about
  // to open the editor and can delete junk by hand. Auto runs the density
  // filter that batch mode uses. Local state: the store's `clean` flag only
  // drives batch runs.
  const [cleanMode, setCleanMode] = useState<"manual" | "auto">("manual");

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
        language: effectiveLang || undefined,
        batch_size: tc.batchSize ?? undefined,
        force: episode!.transcribed,
        // CPU mode forces diarize off regardless of stored preference —
        // pyannote needs a GPU in practice, and the UI column is hidden.
        diarize: isCpu ? false : tc.diarize,
        clean: cleanMode === "auto",
        hf_token: tc.hfToken || undefined,
        num_speakers: tc.numSpeakers ? Number(tc.numSpeakers) : undefined,
        show: showMeta?.name || "",
        episode: episode!.title,
      }),
    onSuccess: (data) => task.startTask(data.task_id),
  });

  if (!episode) return null;
  const expanded = task.expanded || !episode.transcribed;

  const emptyStateProps = downloadDisabled
    ? {
        title: "Downloading audio…",
        description: "The audio is downloading — this view will refresh when it's ready.",
      }
    : {
        title: "No audio file yet",
        description:
          "Download the audio file to transcribe it, or upload an existing transcript (JSON, SRT, VTT).",
        action: {
          label: "Download audio",
          onClick: () => episodeDownloadMutation.mutate({ guids: [episode.id] }),
        },
        secondaryAction: {
          label: "Upload transcript",
          onClick: () => fileInputRef.current?.click(),
        },
      };

  return (
    <PipelinePanel
      title="Transcribe"
      description="Transcribe audio or import subtitles."
      blocker={
        !audioPath ? (
          <>
            <EmptyState icon={FileAudio} {...emptyStateProps} />
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.srt,.vtt"
              onChange={handleFileUpload}
              className="hidden"
            />
          </>
        ) : undefined
      }
      done={episode.transcribed}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel={transcribeSource === "audio" ? "Re-run transcription" : transcribeSource === "subtitles" ? "Reimport subtitles" : "Upload transcript"}
      settingsLabel={transcribeSource === "audio" ? "Transcription settings" : transcribeSource === "subtitles" ? "Import subtitles" : "Upload transcript"}
      taskId={task.activeTaskId}
      onTaskComplete={task.handleComplete}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="No transcript yet."
      controls={
        <div className="px-4 pt-3 pb-4 space-y-4">
          <FormGrid>
            <HelpLabel label="Source" />
            <Segmented
              value={transcribeSource}
              onChange={pickSource}
              options={[
                ["audio", "Audio", hasRealAudio ? "Transcribe the audio file" : "No audio file available", hasRealAudio],
                ...(hasSubs ? [["subtitles", "Subtitles", "Reimport a .vtt/.srt already in the episode folder"] as const] : []),
                ["upload", "Upload", "Upload any transcript file from disk"],
              ]}
            />

            {transcribeSource === "audio" && hasWhisperX && (
              <TranscribeAudioRows
                preset={transcribePreset}
                onSelectPreset={applyTranscribePreset}
                isCpu={isCpu}
                modelSize={tc.modelSize} setModelSize={(v) => setTc({ modelSize: v })}
                language={language} setLanguage={setLanguage}
                customLang={customLang} setCustomLang={setCustomLang}
                diarize={tc.diarize} setDiarize={(v) => setTc({ diarize: v })}
                hfToken={tc.hfToken} setHfToken={(v) => setTc({ hfToken: v })}
                hfTokenDetected={hfTokenDetected}
                cleanMode={cleanMode} setCleanMode={setCleanMode}
                whisperModels={whisperModelsMap}
              />
            )}
          </FormGrid>

          {transcribeSource === "audio" && !hasWhisperX && (
            <MissingDependency
              extra="pipeline"
              label="WhisperX"
              description="Required for automatic transcription. You can still upload a transcript file manually."
            />
          )}

          {transcribeSource === "subtitles" && (
            <div className="space-y-2">
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
              {importFileMutation.isError && (
                <p className="text-destructive text-xs">{errorMessage(importFileMutation.error)}</p>
              )}
            </div>
          )}

          {transcribeSource === "upload" && (
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground">
                Upload a transcript or subtitle file (JSON, SRT, VTT).
              </p>
              <Button onClick={() => fileInputRef.current?.click()} disabled={uploadMutation.isPending} variant="outline" size="sm">
                <Upload className="w-3.5 h-3.5 mr-1.5" />
                {uploadMutation.isPending ? "Uploading…" : "Upload file"}
              </Button>
              {uploadMutation.isError && (
                <p className="text-destructive text-xs">{errorMessage(uploadMutation.error)}</p>
              )}
            </div>
          )}

          {transcribeSource === "audio" && hasWhisperX && (
            <div className="flex items-baseline gap-3 flex-wrap pt-1">
              <Button
                onClick={() => startMutation.mutate()}
                disabled={startMutation.isPending || !hasRealAudio}
                size="sm"
                title={!hasRealAudio ? "Download the audio file first" : undefined}
              >
                {startMutation.isPending ? "Starting…" : episode.transcribed ? "Re-transcribe" : "Transcribe"}
              </Button>
              {episode.transcribed && (
                <span className="text-xs text-muted-foreground">Saves a new version — previous ones stay in History.</span>
              )}
              {startMutation.isError && (
                <p className="text-destructive text-xs w-full">{errorMessage(startMutation.error)}</p>
              )}
            </div>
          )}

          <input ref={fileInputRef} type="file" accept=".json,.srt,.vtt" onChange={handleFileUpload} className="hidden" />
        </div>
      }
    >
      {episode.transcribed && !task.activeTaskId && (
        <TranscriptViewer
          editorKey="transcribe"
          audioPath={audioPath ?? undefined}
          loadSegments={() => getSegments(audioPath!)}
          saveSegments={(segs) => saveSegments(audioPath!, segs)}
          saveSpeakerMap={(m) => saveSpeakerMap(audioPath!, m)}
          showDelete
          showFlags
          showSpeaker
          speakers={showMeta?.speakers}
          sourceLabel={transcriptSourceLabel(episode.provenance)}
          exportSource="transcript"
          exportFilename={episode.stem || undefined}
          loadVersions={() => getTranscribeVersions(audioPath!)}
          loadVersion={(id) => loadTranscribeVersion(audioPath!, id)}
          deleteVersion={(id) => deleteTranscribeVersion(audioPath!, id)}
        />
      )}
    </PipelinePanel>
  );
}

/** Grid-row fragments for the audio-source transcription form.
 *  Must be rendered inside a FormGrid so Source/Transcription/Model/Language/
 *  Cleanup labels share the same auto-sized label column. */
function TranscribeAudioRows({
  preset, onSelectPreset, isCpu,
  modelSize, setModelSize,
  language, setLanguage,
  customLang, setCustomLang,
  diarize, setDiarize,
  hfToken, setHfToken, hfTokenDetected,
  cleanMode, setCleanMode,
  whisperModels,
}: {
  preset: string;
  onSelectPreset: (key: keyof typeof TRANSCRIBE_PRESETS) => void;
  isCpu: boolean;
  modelSize: string; setModelSize: (v: string) => void;
  whisperModels?: Record<string, string>;
  language: string; setLanguage: (v: string) => void;
  customLang: string; setCustomLang: (v: string) => void;
  diarize: boolean; setDiarize: (v: boolean) => void;
  cleanMode: "manual" | "auto"; setCleanMode: (v: "manual" | "auto") => void;
  hfToken: string; setHfToken: (v: string) => void;
  hfTokenDetected: boolean;
}) {
  return (
    <>
      <HelpLabel label="Transcription" />
      <Segmented
        value={preset as keyof typeof TRANSCRIBE_PRESETS}
        onChange={onSelectPreset}
        options={(Object.entries(TRANSCRIBE_PRESETS) as [keyof typeof TRANSCRIBE_PRESETS, (typeof TRANSCRIBE_PRESETS)[keyof typeof TRANSCRIBE_PRESETS]][]).map(
          ([key, p]) => [key, p.label, p.desc] as const,
        )}
      />

      <HelpLabel label="Model" help="Speech recognition model. Bigger models make fewer mistakes but are slower and need more GPU memory." />
      <select
        value={modelSize}
        onChange={(e) => setModelSize(e.target.value)}
        className={selectClass}
      >
        {(() => {
          const entries = whisperModels && Object.keys(whisperModels).length > 0
            ? Object.entries(whisperModels)
            : [[modelSize, modelSize] as [string, string]];
          const filtered = entries.filter(([key]) => (isCpu ? CPU_MODELS : GPU_MODELS).has(key));
          const labels = isCpu ? CPU_LABELS : GPU_LABELS;
          return filtered.map(([key, desc]) => (
            <option key={key} value={key} title={desc}>{key} — {labels[key] || desc}</option>
          ));
        })()}
      </select>

      <HelpLabel label="Language" help="The spoken language of the audio. Auto-detect works for most cases; setting it explicitly improves accuracy and word-level alignment." />
      <div className="flex flex-wrap gap-1.5">
        {([{ code: "", label: "Auto" }, ...TOP_LANGUAGES] as const).map((l) => (
          <button
            key={l.code || "auto"}
            onClick={() => { setLanguage(l.code); setCustomLang(""); }}
            className={`px-2.5 py-1 text-xs rounded-md border transition ${language === l.code ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
          >
            {l.label}
          </button>
        ))}
        <button
          onClick={() => setLanguage("other")}
          className={`px-2.5 py-1 text-xs rounded-md border transition ${language === "other" ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
        >
          Other
        </button>
        {language === "other" && (
          <input
            value={customLang}
            onChange={(e) => setCustomLang(e.target.value.toLowerCase().slice(0, 5))}
            placeholder="ISO code (e.g. ja, zh, ar)"
            className="input text-xs w-36"
            autoFocus
          />
        )}
      </div>

      {!isCpu && (
        <>
          <HelpLabel label="Speakers" />
          <label className="flex items-center gap-1.5 cursor-pointer text-xs text-muted-foreground w-fit">
            <input
              type="checkbox"
              checked={diarize}
              onChange={(e) => setDiarize(e.target.checked)}
              className="accent-primary"
            />
            Identify speakers
          </label>
        </>
      )}

      {!isCpu && diarize && !hfTokenDetected && (
        <>
          <HelpLabel label="HF token" help="HuggingFace access token, needed to download the speaker detection model. Get one free at huggingface.co/settings/tokens." />
          <input
            type="password"
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
            placeholder="hf_..."
            className="input"
          />
        </>
      )}

      <HelpLabel label="Cleanup" />
      <Segmented
        value={cleanMode}
        onChange={setCleanMode}
        options={[
          ["manual", "Manual", "Keep every segment — you'll review and delete junk in the editor"],
          ["auto", "Auto", "Automatically remove hallucinated segments (low speech density)"],
        ]}
      />
    </>
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
