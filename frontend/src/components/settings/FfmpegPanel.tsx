import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getConfig, getHealth, updateConfig, validateFfmpegPath } from "@/api/client";
import type { AppConfig } from "@/api/types";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { restartApp } from "@/lib/restartApp";
import { usePlatform } from "@/platform";
import {
  AlertCircle, CheckCircle2, RefreshCw, Wrench, Loader2, Power,
  FolderOpen, Check, X,
} from "lucide-react";

export default function FfmpegPanel() {
  const qc = useQueryClient();
  const platform = usePlatform();

  const { data: health, isLoading: healthLoading, isFetching: healthFetching } = useQuery({
    queryKey: queryKeys.health(),
    queryFn: getHealth,
    staleTime: 30_000,
  });
  const { data: config } = useQuery({
    queryKey: queryKeys.config(),
    queryFn: getConfig,
  });

  const ffmpegOk = !!health?.capabilities?.ffmpeg;
  const onRecheck = () => qc.invalidateQueries({ queryKey: queryKeys.health() });

  return (
    <section className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold flex items-center gap-2">
          <Wrench className="w-4 h-4" /> ffmpeg
        </h2>
        <Button variant="ghost" size="sm" onClick={onRecheck} className="h-7" disabled={healthFetching}>
          <RefreshCw className={`w-3.5 h-3.5 ${healthFetching ? "animate-spin" : ""}`} />
        </Button>
      </div>

      {healthLoading ? (
        <p className="text-sm text-muted-foreground flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading…
        </p>
      ) : (
        <>
          <StatusCard ok={ffmpegOk} />
          {!ffmpegOk && <InstallInstructions />}
          {config && <OverrideRow config={config} platform={platform} />}
          {!ffmpegOk && (
            <div className="flex items-center gap-2">
              <Button variant="outline" onClick={onRecheck} disabled={healthFetching}>
                <RefreshCw className={`w-4 h-4 mr-2 ${healthFetching ? "animate-spin" : ""}`} />
                I&apos;ve installed it, re-check
              </Button>
              <Button onClick={() => void restartApp()}>
                <Power className="w-4 h-4 mr-2" /> Restart app
              </Button>
            </div>
          )}
        </>
      )}

      <ExplainerCopy />
    </section>
  );
}

function StatusCard({ ok }: { ok: boolean }) {
  const Icon = ok ? CheckCircle2 : AlertCircle;
  const heading = ok ? "ffmpeg available" : "ffmpeg not found";
  const sub = ok
    ? "Pipeline steps that decode audio (transcribe, voice samples) will work."
    : "Transcription, diarization and voice-sample upload will fail until you install ffmpeg.";

  return (
    <div className={`border rounded-lg p-4 flex items-start gap-3 ${
      ok ? "border-border" : "border-warning/40 bg-warning/5"
    }`}>
      <Icon className={`w-6 h-6 shrink-0 ${ok ? "text-success" : "text-warning"}`} />
      <div className="flex-1 min-w-0">
        <div className="font-medium text-sm">{heading}</div>
        <div className="text-xs text-muted-foreground mt-0.5">{sub}</div>
      </div>
    </div>
  );
}

function InstallInstructions() {
  return (
    <div className="rounded-lg border border-border p-4 space-y-3 text-sm">
      <div className="font-medium text-sm">Install ffmpeg</div>
      <p className="text-xs text-muted-foreground">
        PodCodex shells out to the system ffmpeg for audio decode and clip
        extraction. We don&apos;t bundle it because the codec-rich builds are
        GPL-licensed. Pick the line for your OS:
      </p>
      <div className="space-y-2">
        <CommandRow label="Windows" cmd="winget install Gyan.FFmpeg" />
        <CommandRow label="macOS" cmd="brew install ffmpeg" />
        <CommandRow label="Linux (Debian / Ubuntu)" cmd="sudo apt install ffmpeg" />
      </div>
    </div>
  );
}

function OverrideRow({ config, platform }: { config: AppConfig; platform: ReturnType<typeof usePlatform> }) {
  const qc = useQueryClient();
  const [draft, setDraft] = useState(config.ffmpeg_exe_override ?? "");
  const [validation, setValidation] = useState<
    { state: "idle" } | { state: "ok"; version: string } | { state: "err"; error: string }
  >({ state: "idle" });

  const validateMut = useMutation({
    mutationFn: (path: string) => validateFfmpegPath(path),
    onSuccess: (resp) => {
      setValidation(
        resp.ok
          ? { state: "ok", version: resp.version }
          : { state: "err", error: resp.error || "Validation failed" },
      );
    },
    onError: (err: Error) => setValidation({ state: "err", error: err.message }),
  });

  const saveMut = useMutation({
    mutationFn: (next: string) => updateConfig({ ...config, ffmpeg_exe_override: next }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.config() });
      qc.invalidateQueries({ queryKey: queryKeys.health() });
    },
  });

  const persisted = config.ffmpeg_exe_override ?? "";
  const trimmed = draft.trim();
  const dirty = trimmed !== persisted;
  const onPick = async () => {
    const picked = await platform.fs.openFileDialog();
    if (picked) {
      setDraft(picked);
      setValidation({ state: "idle" });
    }
  };

  return (
    <div className="rounded-lg border border-border p-4 space-y-3 text-sm">
      <div className="font-medium text-sm">Custom ffmpeg path (optional)</div>
      <p className="text-xs text-muted-foreground">
        Point PodCodex at a non-PATH ffmpeg install. Saved persistently and
        injected into the sidecar on each launch, so it survives restarts.
        Leave blank to use whatever&apos;s on PATH.
      </p>
      <div className="flex items-stretch gap-2">
        <input
          value={draft}
          onChange={(e) => { setDraft(e.target.value); setValidation({ state: "idle" }); }}
          placeholder={platform.isTauri ? "Click Browse, or paste an absolute path" : "/absolute/path/to/ffmpeg"}
          className="input flex-1 font-mono text-xs"
          spellCheck={false}
        />
        {platform.isTauri && (
          <Button variant="outline" size="sm" onClick={onPick}>
            <FolderOpen className="w-3.5 h-3.5 mr-1.5" /> Browse
          </Button>
        )}
      </div>

      <ValidationLine validation={validation} pending={validateMut.isPending} />

      <div className="flex items-center gap-2 justify-end">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => trimmed && validateMut.mutate(trimmed)}
          disabled={!trimmed || validateMut.isPending}
        >
          {validateMut.isPending ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : <Check className="w-3.5 h-3.5 mr-1.5" />}
          Validate
        </Button>
        <Button
          size="sm"
          onClick={() => saveMut.mutate(trimmed)}
          disabled={!dirty || saveMut.isPending}
        >
          {saveMut.isPending ? <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" /> : null}
          Save
        </Button>
        {persisted && !dirty && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => saveMut.mutate("")}
            disabled={saveMut.isPending}
            title="Clear the override and fall back to PATH"
          >
            <X className="w-3.5 h-3.5 mr-1.5" /> Clear
          </Button>
        )}
      </div>
      {saveMut.isSuccess && !dirty && (
        <p className="text-xs text-success flex items-center gap-1">
          <Check className="w-3.5 h-3.5" /> Saved. Restart the app to apply to running pipeline workers.
        </p>
      )}
    </div>
  );
}

function ValidationLine({
  validation,
  pending,
}: {
  validation: { state: "idle" } | { state: "ok"; version: string } | { state: "err"; error: string };
  pending: boolean;
}) {
  if (pending) {
    return (
      <p className="text-xs text-muted-foreground flex items-center gap-1">
        <Loader2 className="w-3.5 h-3.5 animate-spin" /> Probing ffmpeg…
      </p>
    );
  }
  if (validation.state === "ok") {
    return (
      <p className="text-xs text-success flex items-center gap-1">
        <Check className="w-3.5 h-3.5" /> {validation.version || "ffmpeg responded"}
      </p>
    );
  }
  if (validation.state === "err") {
    return (
      <p className="text-xs text-destructive flex items-start gap-1">
        <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
        <span>{validation.error}</span>
      </p>
    );
  }
  return null;
}

function CommandRow({ label, cmd }: { label: string; cmd: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-muted-foreground w-44 shrink-0">{label}</span>
      <code className="font-mono text-xs bg-muted px-2 py-1 rounded select-all flex-1">
        {cmd}
      </code>
    </div>
  );
}

function ExplainerCopy() {
  return (
    <p className="text-xs text-muted-foreground leading-relaxed">
      Pipeline subprocess workers inherit PATH at spawn, so a recent install is
      picked up the next time you transcribe. Restart the app only if a step
      still fails after re-checking.
    </p>
  );
}
