import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Cpu, Zap, Download, Power, PowerOff, Trash2, Loader2,
  CheckCircle2, AlertCircle,
} from "lucide-react";
import {
  getGPUStatus, downloadGPUBackend, activateGPUBackend,
  deactivateGPUBackend, uninstallGPUBackend,
  getDeviceInfo, setDeviceOverride,
} from "@/api/client";
import type { DeviceInfo, DeviceOverride } from "@/api/gpu";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { ErrorAlert } from "@/components/ui/error-alert";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { RefreshIconButton } from "@/components/common/RefreshIconButton";
import { restartApp } from "@/lib/restartApp";
import { useGpuDownloadTask } from "@/hooks/useGpuDownload";
import { useGpuDownloadStore } from "@/stores/gpuDownloadStore";

// Sidecar selection happens once at app launch (lib.rs::spawn_backend_if_needed),
// so flipping the activated marker — or replacing the binary under an already-
// activated marker — has no effect on the running sidecar. Auto-restart after
// activate/deactivate, and after a download that completes while activated
// (the update flow), so the user sees the change without a manual relaunch.

export default function GPUBackendPanel() {
  const qc = useQueryClient();
  // The download lives in a store and is finished by the app-wide watcher,
  // so leaving Settings mid-download loses nothing.
  const { taskId: activeTaskId, task } = useGpuDownloadTask();
  const lastError = useGpuDownloadStore((s) => s.lastError);
  const startDownload = useGpuDownloadStore((s) => s.start);
  const setLastError = useGpuDownloadStore((s) => s.setError);

  const { data: status, isLoading, isFetching, error: statusError, refetch } = useQuery({
    queryKey: queryKeys.gpuStatus(),
    queryFn: getGPUStatus,
  });

  const { data: deviceInfo, isFetching: deviceFetching, refetch: refetchDevice } = useQuery({
    queryKey: queryKeys.deviceInfo(),
    queryFn: getDeviceInfo,
  });
  const statusRefreshing = isFetching || deviceFetching;

  const overrideMut = useMutation({
    mutationFn: (next: DeviceOverride) => setDeviceOverride(next),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.deviceInfo() });
      void restartApp();
    },
  });


  const downloadMut = useMutation({
    mutationFn: () => downloadGPUBackend(),
    onMutate: () => setLastError(null),
    onSuccess: (resp) => startDownload(resp.task_id, !!status?.activated),
    onError: (err) => {
      const msg = err instanceof Error ? err.message : String(err);
      setLastError(msg);
      console.error("GPU download POST failed:", err);
    },
  });
  const activateMut = useMutation({
    mutationFn: activateGPUBackend,
    onSuccess: () => { void restartApp(); },
  });
  const deactivateMut = useMutation({
    mutationFn: deactivateGPUBackend,
    onSuccess: () => { void restartApp(); },
  });
  const uninstallMut = useMutation({
    mutationFn: uninstallGPUBackend,
    meta: { invalidates: [queryKeys.gpuStatus()] },
  });

  if (statusError && !status) {
    return (
      <section className="space-y-4">
        <Heading />
        <ErrorAlert error={statusError} onRetry={() => void refetch()} />
      </section>
    );
  }

  if (isLoading || !status) {
    return (
      <section className="space-y-4">
        <Heading />
        <p className="text-sm text-muted-foreground flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading…
        </p>
      </section>
    );
  }

  // Pending, running or not polled yet: anything but a finished task.
  const downloading = !!activeTaskId;
  // lastError survives task completion + mutation reset; the live error
  // sources take precedence while they're set.
  const downloadError =
    (downloadMut.error ? (downloadMut.error as Error).message : null) ??
    (task?.status === "failed" ? task.error ?? "Download failed" : null) ??
    lastError;

  return (
    <section className="space-y-6">
      <div className="flex items-center justify-between">
        <Heading />
        <RefreshIconButton
          refreshing={statusRefreshing}
          onClick={() => { void refetch(); void refetchDevice(); }}
        />
      </div>

      <StatusCard status={status} />

      <DeviceOverrideRow
        info={deviceInfo}
        pending={overrideMut.isPending}
        error={overrideMut.error ? (overrideMut.error as Error).message : null}
        onChange={(next) => overrideMut.mutate(next)}
      />

      {status.mode === "bundle" && status.needs_update && (
        <UpdateAvailableBanner
          installedVersion={status.installed_server_version}
          appVersion={status.app_version}
          downloading={downloading}
          onUpdate={() => downloadMut.mutate()}
          mutating={downloadMut.isPending}
        />
      )}

      {status.mode === "dev" && <DevModeBanner />}

      {status.mode === "bundle" && (
        <ActionBlock
          status={status}
          downloading={downloading}
          downloadError={downloadError}
          downloadProgress={task?.progress ?? 0}
          downloadMessage={task?.message ?? ""}
          onDownload={() => downloadMut.mutate()}
          onActivate={() => activateMut.mutate()}
          onDeactivate={() => deactivateMut.mutate()}
          onUninstall={() =>
            confirmDialog.open({
              title: "Uninstall the GPU backend?",
              description: "This deletes the downloaded CUDA backend (about 3 GB). Using the GPU again means downloading it again.",
              confirmLabel: "Uninstall",
              variant: "destructive",
              onConfirm: () => uninstallMut.mutate(),
            })
          }
          mutating={
            activateMut.isPending || deactivateMut.isPending ||
            uninstallMut.isPending || downloadMut.isPending
          }
        />
      )}

      <ExplainerCopy />
    </section>
  );
}

function Heading() {
  return (
    <h2 className="text-base font-semibold flex items-center gap-2">
      <Zap className="w-4 h-4" /> GPU acceleration
    </h2>
  );
}

function StatusCard({ status }: { status: import("@/api/gpu").GPUStatus }) {
  const isGPU = status.activated && status.installed_version !== null;
  const Icon = isGPU ? Zap : Cpu;
  const heading = isGPU
    ? "CUDA Backend"
    : status.gpu_detected
    ? "CPU only"
    : "CPU only";
  const sub = isGPU
    // installed_version is the cuda-libs content tag, a hash: say which
    // PodCodex build is installed instead.
    ? status.installed_server_version
      ? `Active · v${status.installed_server_version}`
      : "Active"
    : status.gpu_detected
    ? `${status.gpu_name} (${status.vram_mb} MB) detected, not yet activated`
    : "No NVIDIA GPU detected";

  return (
    <div className="border border-border rounded-lg p-4 flex items-start gap-3">
      <Icon className={`w-6 h-6 shrink-0 ${isGPU ? "text-primary" : "text-muted-foreground"}`} />
      <div className="flex-1 min-w-0">
        <div className="font-medium text-sm">{heading}</div>
        <div className="text-xs text-muted-foreground mt-0.5">{sub}</div>
      </div>
    </div>
  );
}

function DevModeBanner() {
  return (
    <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm space-y-1">
      <div className="flex items-center gap-2 font-medium">
        <AlertCircle className="w-4 h-4 text-muted-foreground" />
        Dev mode
      </div>
      <p className="text-xs text-muted-foreground">
        GPU backend management is only available in the packaged desktop app.
        Your dev environment uses whatever torch is installed in the venv.
        On this machine that&apos;s the GPU build, so the pipeline already
        runs on your GPU.
      </p>
    </div>
  );
}

function UpdateAvailableBanner({
  installedVersion,
  appVersion,
  downloading,
  onUpdate,
  mutating,
}: {
  installedVersion: string | null;
  appVersion: string;
  downloading: boolean;
  onUpdate: () => void;
  mutating: boolean;
}) {
  return (
    <div className="rounded-lg border border-warning/40 bg-warning/5 p-4 text-sm space-y-2">
      <div className="flex items-center gap-2 font-medium text-warning">
        <AlertCircle className="w-4 h-4" />
        GPU backend out of date
      </div>
      <p className="text-xs text-muted-foreground">
        Your installed GPU backend is at version{" "}
        <code className="font-mono text-2xs">{installedVersion ?? "unknown"}</code>{" "}
        but the app is at{" "}
        <code className="font-mono text-2xs">{appVersion}</code>. Hardware
        acceleration is currently disabled, and the app fell back to the CPU
        sidecar. Update to restore GPU acceleration: only the parts that
        changed are downloaded, usually just the PodCodex server core.
      </p>
      <Button onClick={onUpdate} disabled={downloading || mutating} size="sm">
        {downloading || mutating ? (
          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
        ) : (
          <Download className="w-4 h-4 mr-2" />
        )}
        {downloading ? "Updating…" : "Update GPU backend"}
      </Button>
    </div>
  );
}

interface ActionBlockProps {
  status: import("@/api/gpu").GPUStatus;
  downloading: boolean;
  downloadError: string | null;
  downloadProgress: number;
  downloadMessage: string;
  onDownload: () => void;
  onActivate: () => void;
  onDeactivate: () => void;
  onUninstall: () => void;
  mutating: boolean;
}

function ActionBlock({
  status, downloading, downloadError, downloadProgress, downloadMessage,
  onDownload, onActivate, onDeactivate, onUninstall, mutating,
}: ActionBlockProps) {
  if (!status.gpu_detected) return null;
  if (downloading) {
    const pct = Math.round(downloadProgress * 100);
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <Loader2 className="w-4 h-4 animate-spin text-primary" />
          <span>{downloadMessage || "Downloading…"}</span>
          <span className="ml-auto tabular-nums text-muted-foreground">{pct}%</span>
        </div>
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
        </div>
      </div>
    );
  }

  const installed = status.installed_version !== null;
  const activated = status.activated;

  return (
    <div className="space-y-3">
      {!installed && (
        <div className="rounded-lg border border-border p-4 space-y-2">
          <div className="text-sm font-medium">Download CUDA backend</div>
          <p className="text-xs text-muted-foreground">
            About 3 GB the first time. Installs into{" "}
            <code className="font-mono text-2xs">{status.install_dir}</code>.
            App updates then download only what changed, usually just the
            PodCodex server core; torch and the CUDA libraries are fetched
            again only when they change.
          </p>
          <Button onClick={onDownload} disabled={mutating} className="mt-2">
            {mutating ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Download className="w-4 h-4 mr-2" />
            )}
            {mutating ? "Starting…" : "Download"}
          </Button>
        </div>
      )}

      {installed && !activated && (
        <div className="flex items-center gap-2">
          <Button onClick={onActivate} disabled={mutating}>
            <Power className="w-4 h-4 mr-2" /> Activate
          </Button>
          <Button variant="ghost" onClick={onUninstall} disabled={mutating}>
            <Trash2 className="w-4 h-4 mr-2" /> Uninstall
          </Button>
        </div>
      )}

      {installed && activated && (
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 text-sm text-success">
            <CheckCircle2 className="w-4 h-4" /> Active
          </div>
          <Button variant="ghost" onClick={onDeactivate} disabled={mutating} className="ml-auto">
            <PowerOff className="w-4 h-4 mr-2" /> Deactivate
          </Button>
          <Button variant="ghost" onClick={onUninstall} disabled={mutating}>
            <Trash2 className="w-4 h-4 mr-2" /> Uninstall
          </Button>
        </div>
      )}

      {downloadError && (
        <p className="text-xs text-destructive flex items-center gap-1">
          <AlertCircle className="w-3.5 h-3.5" /> {downloadError}
        </p>
      )}
    </div>
  );
}

function DeviceOverrideRow({
  info,
  pending,
  error,
  onChange,
}: {
  info: DeviceInfo | undefined;
  pending: boolean;
  error: string | null;
  onChange: (next: DeviceOverride) => void;
}) {
  // Persisted setting drives the control; env override (if different) is
  // surfaced as a hint so the user understands why the running process
  // diverges from what's saved on disk.
  const persisted = info?.persisted_override ?? "auto";
  const envForced = info?.override ?? null;
  const envOverridesPersisted = envForced !== null && envForced !== persisted;
  const isCpu = persisted === "cpu";
  const next: DeviceOverride = isCpu ? "auto" : "cpu";

  return (
    <div className="rounded-lg border border-border p-4 space-y-2">
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <div className="text-sm font-medium">Force CPU mode</div>
          <p className="text-xs text-muted-foreground mt-0.5">
            Skip GPU even when one is available. Useful for low-VRAM setups
            or when a driver mismatch is causing crashes.
          </p>
        </div>
        <div className="shrink-0 flex items-center gap-2">
          <Button
            variant={isCpu ? "default" : "outline"}
            size="sm"
            disabled={pending}
            onClick={() => onChange(next)}
          >
            {pending && <Loader2 className="w-3.5 h-3.5 mr-2 animate-spin" />}
            {isCpu ? "On" : "Off"}
          </Button>
        </div>
      </div>
      {envOverridesPersisted && (
        <p className="text-xs text-muted-foreground flex items-start gap-1">
          <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          <span>
            Running process is forced to <code className="font-mono text-2xs">{envForced}</code>{" "}
            via the <code className="font-mono text-2xs">PODCODEX_DEVICE</code> env var,
            which overrides this setting until unset.
          </span>
        </p>
      )}
      {error && (
        <p className="text-xs text-destructive flex items-center gap-1">
          <AlertCircle className="w-3.5 h-3.5" /> {error}
        </p>
      )}
    </div>
  );
}

function ExplainerCopy() {
  return (
    <p className="text-xs text-muted-foreground leading-relaxed">
      PodCodex automatically detects and uses the best available GPU on your
      system. On Apple Silicon Macs, transcription falls back to CPU until
      WhisperX upstream adds MPS support. On Windows or Linux with an NVIDIA
      GPU, you can download an optional CUDA backend for hardware-accelerated
      transcription, diarization, and embedding. When no GPU is detected,
      every pipeline step still runs, just slower.
    </p>
  );
}
