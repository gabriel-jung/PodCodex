import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { getModels, deleteModel, getExtras, installExtra, removeExtra } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import type { ExtraInfo } from "@/api/types";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Trash2, HardDrive, Cpu, RefreshCw, Puzzle, Download, X, Loader2 } from "lucide-react";
import { useState } from "react";

export default function SettingsPage() {
  const navigate = useNavigate();
  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
        <div className="flex items-center gap-4">
          <Button onClick={() => navigate({ to: "/" })} variant="ghost" size="sm">
            <ArrowLeft /> Home
          </Button>
          <h1 className="text-2xl font-bold">Settings</h1>
        </div>
        <PluginsPanel />
        <ModelCachePanel />
      </div>
    </div>
  );
}

// ── Plugins ──────────────────────────────────

function PluginsPanel() {
  const qc = useQueryClient();
  const { data, isLoading, refetch } = useQuery({
    queryKey: queryKeys.capabilities(),
    queryFn: getExtras,
  });

  const [pendingAction, setPendingAction] = useState<string | null>(null);

  const installMut = useMutation({
    mutationFn: (extra: string) => installExtra(extra),
    onMutate: (extra) => setPendingAction(extra),
    onSettled: () => {
      setPendingAction(null);
      qc.invalidateQueries({ queryKey: queryKeys.capabilities() });
      qc.invalidateQueries({ queryKey: queryKeys.health() });
    },
  });

  const removeMut = useMutation({
    mutationFn: (extra: string) => removeExtra(extra),
    onMutate: (extra) => setPendingAction(extra),
    onSettled: () => {
      setPendingAction(null);
      qc.invalidateQueries({ queryKey: queryKeys.capabilities() });
      qc.invalidateQueries({ queryKey: queryKeys.health() });
    },
  });

  const extras = data?.extras ?? {};
  const entries = Object.entries(extras) as [string, ExtraInfo][];
  const installedCount = entries.filter(([, v]) => v.installed).length;

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Puzzle className="w-5 h-5" /> Plugins
        </h2>
        <Button variant="ghost" size="sm" onClick={() => refetch()} className="h-7">
          <RefreshCw className="w-3.5 h-3.5" />
        </Button>
      </div>

      <p className="text-sm text-muted-foreground">
        PodCodex features are split into optional plugins so you only install what you need.
        Install or remove them here — changes take effect after restarting the backend.
      </p>

      <div className="text-xs text-muted-foreground">
        {installedCount} of {entries.length} plugin{entries.length !== 1 ? "s" : ""} installed
      </div>

      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading...</p>
      ) : (
        <div className="border border-border rounded-lg divide-y divide-border">
          {entries.map(([name, info]) => {
            const busy = pendingAction === name;
            return (
              <div key={name} className="flex items-center gap-4 px-4 py-3">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">{name}</span>
                    {info.installed ? (
                      <span className="text-2xs font-medium px-1.5 py-0.5 rounded-full bg-success/10 text-success">
                        installed
                      </span>
                    ) : (
                      <span className="text-2xs font-medium px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">
                        not installed
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5 truncate">{info.description}</p>
                </div>
                <div className="shrink-0">
                  {busy ? (
                    <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                  ) : info.installed ? (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 text-xs text-muted-foreground hover:text-destructive"
                      onClick={() => removeMut.mutate(name)}
                      disabled={!!pendingAction}
                    >
                      <X className="w-3.5 h-3.5 mr-1" /> Remove
                    </Button>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs"
                      onClick={() => installMut.mutate(name)}
                      disabled={!!pendingAction}
                    >
                      <Download className="w-3.5 h-3.5 mr-1" /> Install
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

// ── Model Cache ──────────────────────────────

function ModelCachePanel() {
  const qc = useQueryClient();
  const { data, isLoading, refetch } = useQuery({
    queryKey: queryKeys.models(),
    queryFn: getModels,
  });

  const [deleting, setDeleting] = useState<string | null>(null);
  const deleteMut = useMutation({
    mutationFn: (id: string) => deleteModel(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.models() });
      setDeleting(null);
    },
  });

  const models = data?.models ?? [];
  const cacheDir = data?.cache_dir ?? "";
  const vram = data?.vram ?? null;
  const totalMB = models.reduce((sum, m) => sum + m.size_mb, 0);

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <HardDrive className="w-5 h-5" /> Model Cache
        </h2>
        <Button variant="ghost" size="sm" onClick={() => refetch()} className="h-7">
          <RefreshCw className="w-3.5 h-3.5" />
        </Button>
      </div>

      <p className="text-sm text-muted-foreground">
        PodCodex downloads ML models for transcription, diarization, embedding,
        and TTS. They are stored in a single cache directory so you can see
        what&apos;s on disk and reclaim space when needed.
      </p>

      {cacheDir && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="font-mono break-all">{cacheDir}</span>
          <span className="shrink-0 text-muted-foreground/60">
            (override with <code className="font-mono">PODCODEX_CACHE_DIR</code>)
          </span>
        </div>
      )}

      {/* VRAM status */}
      {vram && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-sm">
            <Cpu className="w-4 h-4 text-muted-foreground" />
            <span className="font-medium">{vram.device}</span>
            <span className="text-muted-foreground ml-auto">
              {vram.used_mb} / {vram.total_mb} MB used
            </span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all"
              style={{ width: `${Math.min(100, (vram.used_mb / vram.total_mb) * 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{vram.free_mb} MB free</span>
            <span>{vram.reserved_mb} MB reserved</span>
          </div>
        </div>
      )}

      {/* Model table */}
      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading...</p>
      ) : models.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No cached models yet. Models are downloaded automatically the first
          time you run a pipeline step (transcribe, polish, index, etc.).
        </p>
      ) : (
        <>
          <div className="text-xs text-muted-foreground">
            {models.length} model{models.length !== 1 ? "s" : ""} &middot; {totalMB.toFixed(1)} MB total
          </div>
          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-muted/50">
                  <th className="text-left px-4 py-2 font-medium">Model</th>
                  <th className="text-right px-4 py-2 font-medium">Size</th>
                  <th className="w-12" />
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.id} className="border-b border-border/50 last:border-0">
                    <td className="px-4 py-2">
                      <span className="font-mono text-xs">{m.name}</span>
                    </td>
                    <td className="px-4 py-2 text-right text-muted-foreground tabular-nums">
                      {m.size_mb >= 1024
                        ? `${(m.size_mb / 1024).toFixed(1)} GB`
                        : `${m.size_mb} MB`}
                    </td>
                    <td className="px-2 py-2">
                      {deleting === m.id ? (
                        <div className="flex items-center gap-1">
                          <Button
                            variant="destructive"
                            size="sm"
                            className="h-6 text-xs"
                            onClick={() => deleteMut.mutate(m.id)}
                            disabled={deleteMut.isPending}
                          >
                            Confirm
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-6 text-xs"
                            onClick={() => setDeleting(null)}
                          >
                            Cancel
                          </Button>
                        </div>
                      ) : (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                          onClick={() => setDeleting(m.id)}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
