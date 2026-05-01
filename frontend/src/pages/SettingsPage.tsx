import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getModels, deleteModel, getExtras, installExtra, removeExtra, getSecretsStatus, updateSecrets, getHealth } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import type { ExtraInfo } from "@/api/types";
import type { SecretStatus } from "@/api/config";
import { Button } from "@/components/ui/button";
import {
  Trash2, HardDrive, Cpu, RefreshCw, Puzzle, Download, X, Loader2,
  Sun, Moon, Monitor, Keyboard, Palette, Mic, Sparkles, Database, Languages, Plug,
  KeyRound, Eye, EyeOff, Check, Zap,
} from "lucide-react";
import AppSidebar from "@/components/layout/AppSidebar";
import PageHeader from "@/components/layout/PageHeader";
import IntegrationsPanel from "@/components/settings/IntegrationsPanel";
import BundleExportPanel from "@/components/settings/BundleExportPanel";
import GPUBackendPanel from "@/components/settings/GPUBackendPanel";
import { useEffect, useState } from "react";
import { useTheme } from "@/hooks/useTheme";
import { SHORTCUTS, Kbd } from "@/components/ShortcutsHelp";
import { NullableNumberInput } from "@/components/ui/number-input";
import PresetCards from "@/components/common/PresetCards";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import {
  TRANSCRIBE_PRESETS, LLM_PRESETS, INDEX_PRESETS,
  CPU_LABELS, GPU_LABELS, CPU_MODELS, GPU_MODELS,
  usePipelineConfigStore,
} from "@/stores/pipelineConfigStore";
import { useFlagPatternsStore } from "@/stores/flagPatternsStore";
import { selectClass } from "@/lib/utils";

type SettingsTab = "general" | "pipeline" | "credentials" | "integrations" | "plugins" | "gpu" | "cache";

// Plugins panel runs `uv sync --extra X` to install Python extras — only
// meaningful when a venv exists (dev mode). The bundled sidecar has its
// extras compiled into the PyInstaller bundle and there's nothing to install
// or remove at runtime. We hide the tab in bundle mode.
const ALL_SECTIONS = [
  { key: "general", label: "General", icon: Palette },
  { key: "pipeline", label: "Pipeline", icon: Sparkles },
  { key: "credentials", label: "Credentials", icon: KeyRound },
  { key: "integrations", label: "Integrations", icon: Plug },
  { key: "plugins", label: "Plugins", icon: Puzzle, devOnly: true },
  { key: "gpu", label: "GPU acceleration", icon: Zap },
  { key: "cache", label: "Model cache", icon: HardDrive },
] as const;

const VALID_TABS: readonly SettingsTab[] = ["general", "pipeline", "credentials", "integrations", "plugins", "gpu", "cache"];

function readInitialTab(): SettingsTab {
  if (typeof window === "undefined") return "general";
  const t = new URLSearchParams(window.location.search).get("tab");
  return (VALID_TABS as readonly string[]).includes(t ?? "") ? (t as SettingsTab) : "general";
}

export default function SettingsPage() {
  const [tab, setTab] = useState<SettingsTab>(readInitialTab);
  const { data: health } = useQuery({
    queryKey: queryKeys.health(),
    queryFn: getHealth,
    staleTime: Infinity,
  });
  const isBundleMode = health?.mode === "bundle";
  const sections = [
    {
      items: ALL_SECTIONS.filter((s) => !(isBundleMode && "devOnly" in s && s.devOnly)),
    },
  ];

  useEffect(() => {
    if (typeof window === "undefined") return;
    const onPop = () => setTab(readInitialTab());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const id = window.location.hash.slice(1);
    if (!id) return;
    const raf = requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "center" });
    });
    return () => cancelAnimationFrame(raf);
  }, [tab]);

  const selectTab = (t: SettingsTab) => {
    setTab(t);
    const usp = new URLSearchParams(window.location.search);
    usp.set("tab", t);
    window.history.replaceState(null, "", `?${usp.toString()}${window.location.hash}`);
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <PageHeader title="Settings" />
      <div className="flex-1 flex overflow-hidden">
        <AppSidebar
          pageSections={sections}
          activeItem={tab}
          onItemClick={(k) => selectTab(k as SettingsTab)}
        />
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-10 py-10 space-y-10">
            {tab === "general" && (
              <>
                <AppearancePanel />
                <ShortcutsPanel />
              </>
            )}
            {tab === "pipeline" && <PipelineDefaultsPanel />}
            {tab === "credentials" && <CredentialsPanel />}
            {tab === "integrations" && <IntegrationsPanel />}
            {tab === "plugins" && <PluginsPanel />}
            {tab === "gpu" && <GPUBackendPanel />}
            {tab === "cache" && (
              <>
                <ModelCachePanel />
                <BundleExportPanel />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Appearance ───────────────────────────────

function AppearancePanel() {
  const { theme, setTheme } = useTheme();
  const options: { value: "light" | "dark" | "system"; label: string; icon: typeof Sun }[] = [
    { value: "light", label: "Light", icon: Sun },
    { value: "dark", label: "Dark", icon: Moon },
    { value: "system", label: "System", icon: Monitor },
  ];

  return (
    <section className="space-y-4">
      <h2 className="font-display text-2xl font-semibold flex items-center gap-2">
        <Palette className="w-5 h-5" /> Appearance
      </h2>
      <div className="flex gap-2">
        {options.map(({ value, label, icon: Icon }) => (
          <button
            key={value}
            onClick={() => setTheme(value)}
            className={`flex-1 flex flex-col items-center gap-2 px-4 py-4 rounded-lg border transition ${
              theme === value
                ? "border-primary bg-primary/5"
                : "border-border hover:bg-accent/50"
            }`}
          >
            <Icon className={`w-5 h-5 ${theme === value ? "text-primary" : "text-muted-foreground"}`} />
            <span className="text-sm">{label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

// ── Shortcuts ────────────────────────────────

function ShortcutsPanel() {
  return (
    <section className="space-y-4">
      <h2 className="font-display text-2xl font-semibold flex items-center gap-2">
        <Keyboard className="w-5 h-5" /> Keyboard shortcuts
      </h2>
      <div className="border border-border rounded-lg divide-y divide-border">
        {SHORTCUTS.map((group) => (
          <div key={group.heading} className="px-4 py-3">
            <p className="text-xs font-medium text-muted-foreground mb-2">{group.heading}</p>
            <ul className="space-y-1.5">
              {group.items.map((sc) => (
                <li key={sc.label} className="flex items-center justify-between text-sm">
                  <span>{sc.label}</span>
                  <span className="flex gap-1">
                    {sc.keys.map((k) => <Kbd key={k}>{k}</Kbd>)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}

// ── Pipeline defaults ────────────────────────

function PipelineDefaultsPanel() {
  const { apiProviders, whisperModels, detectedKeys } = useLLMProviders();

  const transcribe = usePipelineConfigStore((s) => s.transcribe);
  const setTranscribe = usePipelineConfigStore((s) => s.setTranscribe);
  const transcribePreset = usePipelineConfigStore((s) => s.transcribePreset);
  const applyTranscribePreset = usePipelineConfigStore((s) => s.applyTranscribePreset);

  const llm = usePipelineConfigStore((s) => s.llm);
  const setLLM = usePipelineConfigStore((s) => s.setLLM);
  const llmPreset = usePipelineConfigStore((s) => s.llmPreset);
  const applyLLMPreset = usePipelineConfigStore((s) => s.applyLLMPreset);

  const targetLang = usePipelineConfigStore((s) => s.targetLang);
  const setTargetLang = usePipelineConfigStore((s) => s.setTargetLang);

  const indexModel = usePipelineConfigStore((s) => s.indexModel);
  const setIndexModel = usePipelineConfigStore((s) => s.setIndexModel);
  const indexPreset = usePipelineConfigStore((s) => s.indexPreset);
  const applyIndexPreset = usePipelineConfigStore((s) => s.applyIndexPreset);

  const cpuEntries = Object.entries(CPU_LABELS);
  const gpuEntries = Object.entries(GPU_LABELS);
  const modelHasLabel = CPU_MODELS.has(transcribe.modelSize) || GPU_MODELS.has(transcribe.modelSize);
  const customModels = modelHasLabel
    ? []
    : Object.keys(whisperModels).filter((m) => !CPU_MODELS.has(m) && !GPU_MODELS.has(m));

  return (
    <div className="space-y-10">
      <div>
        <p className="text-sm text-muted-foreground">
          These values prefill every episode panel and the batch modal. Per-show
          overrides live in each show&apos;s Settings tab; episode panels can
          still tweak values for a single run.
        </p>
      </div>

      <section className="space-y-3">
        <h2 className="font-display text-2xl font-semibold flex items-center gap-2">
          <Mic className="w-5 h-5" /> Transcribe
        </h2>
        <PresetCards
          presets={TRANSCRIBE_PRESETS}
          active={transcribePreset}
          onSelect={applyTranscribePreset}
        />
        <label className="block">
          <span className="text-xs text-muted-foreground">Whisper model</span>
          <select
            value={transcribe.modelSize}
            onChange={(e) => setTranscribe({ modelSize: e.target.value })}
            className={selectClass + " mt-1"}
          >
            <optgroup label="CPU-friendly">
              {cpuEntries.map(([key, label]) => (
                <option key={key} value={key}>{key} — {label}</option>
              ))}
            </optgroup>
            <optgroup label="GPU">
              {gpuEntries.map(([key, label]) => (
                <option key={key} value={key}>{key} — {label}</option>
              ))}
            </optgroup>
            {customModels.length > 0 && (
              <optgroup label="Other">
                {customModels.map((m) => <option key={m} value={m}>{m}</option>)}
              </optgroup>
            )}
          </select>
        </label>
        <div className="space-y-2 text-sm">
          <label className="flex items-start gap-2">
            <input
              type="checkbox"
              checked={transcribe.diarize}
              onChange={(e) => setTranscribe({ diarize: e.target.checked })}
              className="mt-1"
            />
            <span>
              Diarize speakers
              <span className="block text-xs text-muted-foreground">
                Detect and label different speakers (requires a HuggingFace token).
              </span>
            </span>
          </label>
          {transcribe.diarize && !detectedKeys.hf_token && (
            <p className="pl-6 text-xs text-muted-foreground">
              HuggingFace token needed —{" "}
              <a
                href="?tab=credentials#HF_TOKEN"
                onClick={(e) => {
                  e.preventDefault();
                  window.history.pushState(null, "", "?tab=credentials#HF_TOKEN");
                  window.dispatchEvent(new PopStateEvent("popstate"));
                }}
                className="underline hover:text-foreground"
              >
                set it up in Credentials
              </a>
              .
            </p>
          )}
          <label className="flex items-start gap-2">
            <input
              type="checkbox"
              checked={transcribe.clean}
              onChange={(e) => setTranscribe({ clean: e.target.checked })}
              className="mt-1"
            />
            <span>
              Clean low-quality segments
              <span className="block text-xs text-muted-foreground">
                Drop garbled or off-mic segments that the model flags as low-confidence.
              </span>
            </span>
          </label>
        </div>
        <label className="block">
          <span className="text-xs text-muted-foreground">Batch size</span>
          <NullableNumberInput
            value={transcribe.batchSize}
            onChange={(batchSize) => setTranscribe({ batchSize })}
            placeholder="Auto"
            min={1}
            className="input mt-1 w-24"
          />
          <span className="block text-xs text-muted-foreground mt-1">
            Empty = auto-detect from VRAM (8 for ≤10&nbsp;GB, 16 above). Lower
            if WhisperX runs out of memory; raise on a big GPU for more speed.
          </span>
        </label>
      </section>

      <section className="space-y-3">
        <h2 className="font-display text-2xl font-semibold flex items-center gap-2">
          <Sparkles className="w-5 h-5" /> Correct &amp; Translate (LLM)
        </h2>
        <PresetCards
          presets={LLM_PRESETS}
          active={llmPreset}
          onSelect={applyLLMPreset}
        />
        {llm.mode === "api" && (
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-xs text-muted-foreground">Provider</span>
              <select
                value={llm.provider}
                onChange={(e) => setLLM({ provider: e.target.value })}
                className={selectClass + " mt-1"}
              >
                {apiProviders.map(([key, spec]) => (
                  <option key={key} value={key}>
                    {spec.label}{detectedKeys[key] ? " ✓" : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="text-xs text-muted-foreground">Model</span>
              <input
                value={llm.model}
                onChange={(e) => setLLM({ model: e.target.value })}
                placeholder="e.g. gpt-4o-mini"
                className="input mt-1"
              />
            </label>
          </div>
        )}
        {llm.mode === "ollama" && (
          <label className="block">
            <span className="text-xs text-muted-foreground">Ollama model</span>
            <input
              value={llm.model}
              onChange={(e) => setLLM({ model: e.target.value })}
              placeholder="e.g. llama3.1:8b"
              className="input mt-1"
            />
          </label>
        )}
        <label className="block">
          <span className="text-xs text-muted-foreground flex items-center gap-1">
            <Languages className="w-3 h-3" /> Default target language
          </span>
          <input
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            placeholder="English"
            className="input mt-1"
          />
        </label>
      </section>

      <section className="space-y-3">
        <h2 className="font-display text-2xl font-semibold flex items-center gap-2">
          <Database className="w-5 h-5" /> Index (embeddings)
        </h2>
        <PresetCards
          presets={INDEX_PRESETS}
          active={indexPreset}
          onSelect={applyIndexPreset}
        />
        <label className="block">
          <span className="text-xs text-muted-foreground">Embedding model</span>
          <input
            value={indexModel}
            onChange={(e) => setIndexModel(e.target.value)}
            className="input mt-1 font-mono text-xs"
          />
        </label>
      </section>

      <FlagPatternsSection />
    </div>
  );
}

function FlagPatternsSection() {
  const patterns = useFlagPatternsStore((s) => s.patterns);
  const setPatterns = useFlagPatternsStore((s) => s.setPatterns);
  const [draft, setDraft] = useState(patterns.join("\n"));

  return (
    <section className="space-y-3">
      <h2 className="font-display text-2xl font-semibold flex items-center gap-2">
        <Sparkles className="w-5 h-5" /> Editor flagging
      </h2>
      <p className="text-xs text-muted-foreground">
        One pattern per line. Segments whose text contains any pattern
        (case-insensitive substring) are flagged for review. Punctuation-only
        segments are flagged automatically.
      </p>
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          const list = draft.split("\n").map((p) => p.trim()).filter(Boolean);
          setPatterns(list);
          setDraft(list.join("\n"));
        }}
        placeholder={"Sous-titres réalisés par\n[Music]\nthanks for watching"}
        rows={6}
        className="input w-full font-mono text-xs resize-y"
      />
    </section>
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
        <h2 className="font-display text-2xl font-semibold flex items-center gap-2">
          <Puzzle className="w-5 h-5" /> Plugins
        </h2>
        <Button variant="ghost" size="sm" onClick={() => refetch()} className="h-7">
          <RefreshCw className="w-3.5 h-3.5" />
        </Button>
      </div>

      <p className="text-sm text-muted-foreground">
        PodCodex features are split into optional plugins so you only install what you need.
        Install or remove them here. Changes take effect after restarting the backend.
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
        <h2 className="font-display text-2xl font-semibold flex items-center gap-2">
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

      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading...</p>
      ) : models.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          No cached models yet. Models are downloaded automatically the first
          time you run a pipeline step (transcribe, correct, index, etc.).
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

// ── Credentials ─────────────────────────────

const SECRET_LABELS: Record<string, { label: string; hint: React.ReactNode; usedFor: string }> = {
  HF_TOKEN: {
    label: "HuggingFace token",
    hint: (
      <>
        Get one free at{" "}
        <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="underline hover:text-foreground">huggingface.co/settings/tokens</a>
        {". "}
        Then accept the terms for{" "}
        <a href="https://huggingface.co/pyannote/speaker-diarization-community-1" target="_blank" rel="noreferrer" className="underline hover:text-foreground">pyannote/speaker-diarization-community-1</a>.
      </>
    ),
    usedFor: "Speaker diarization (transcribe step)",
  },
  OPENAI_API_KEY: {
    label: "OpenAI API key",
    hint: <>Keys at <a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer" className="underline hover:text-foreground">platform.openai.com/api-keys</a>.</>,
    usedFor: "LLM correct / translate / synthesize with OpenAI provider",
  },
  ANTHROPIC_API_KEY: {
    label: "Anthropic API key",
    hint: <>Keys at <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noreferrer" className="underline hover:text-foreground">console.anthropic.com/settings/keys</a>.</>,
    usedFor: "LLM correct / translate / synthesize with Anthropic provider",
  },
  MISTRAL_API_KEY: {
    label: "Mistral API key",
    hint: <>Keys at <a href="https://console.mistral.ai/api-keys" target="_blank" rel="noreferrer" className="underline hover:text-foreground">console.mistral.ai/api-keys</a>.</>,
    usedFor: "LLM correct / translate with Mistral provider",
  },
  DISCORD_TOKEN: {
    label: "Discord bot token",
    hint: <>Reset at the bot's page on <a href="https://discord.com/developers/applications" target="_blank" rel="noreferrer" className="underline hover:text-foreground">discord.com/developers/applications</a>.</>,
    usedFor: "Running the Discord bot (`podcodex-bot`)",
  },
};

function CredentialsPanel() {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: queryKeys.secrets(),
    queryFn: getSecretsStatus,
  });

  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [reveal, setReveal] = useState<Record<string, boolean>>({});
  const [savedAt, setSavedAt] = useState<number | null>(null);

  const mutation = useMutation({
    mutationFn: updateSecrets,
    onSuccess: (next) => {
      queryClient.setQueryData(queryKeys.secrets(), next);
      queryClient.invalidateQueries({ queryKey: queryKeys.pipelineConfig() });
      setDrafts({});
      setSavedAt(Date.now());
    },
  });

  const dirty = Object.values(drafts).some((v) => v !== undefined);

  return (
    <section className="space-y-6">
      <div>
        <h2 className="font-display text-2xl font-semibold flex items-center gap-2">
          <KeyRound className="w-5 h-5" /> Credentials
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          API keys and tokens used by the pipeline, search, and Discord bot. Saved
          to a user-scoped file ({data?.path ?? "~/.config/podcodex/secrets.env"})
          with read/write restricted to your account. Leave blank to rely on an
          environment variable instead.
        </p>
      </div>

      {isLoading ? (
        <div className="text-sm text-muted-foreground flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin" /> Loading…
        </div>
      ) : (
        <div className="space-y-4">
          {data?.items.map((item) => {
            const meta = SECRET_LABELS[item.key] ?? { label: item.key, hint: null, usedFor: "" };
            const draft = drafts[item.key];
            const showReveal = !!reveal[item.key];
            return (
              <div key={item.key} id={item.key} className="border border-border rounded-lg p-4 space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-medium">{meta.label}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">{meta.usedFor}</div>
                  </div>
                  <SecretBadge item={item} />
                </div>
                <div className="flex items-stretch gap-2">
                  <input
                    type={showReveal ? "text" : "password"}
                    value={draft ?? ""}
                    onChange={(e) => setDrafts((d) => ({ ...d, [item.key]: e.target.value }))}
                    placeholder={item.set ? item.masked : "Not set"}
                    className="input flex-1 font-mono text-xs"
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <button
                    type="button"
                    onClick={() => setReveal((r) => ({ ...r, [item.key]: !showReveal }))}
                    className="px-2 rounded-md border border-border hover:bg-accent text-muted-foreground"
                    aria-label={showReveal ? "Hide secret" : "Reveal secret while typing"}
                  >
                    {showReveal ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                  {item.set && item.source === "file" && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setDrafts((d) => ({ ...d, [item.key]: "" }))}
                      title="Clear this key on next save"
                    >
                      Clear
                    </Button>
                  )}
                </div>
                {meta.hint && <p className="text-xs text-muted-foreground">{meta.hint}</p>}
              </div>
            );
          })}
        </div>
      )}

      <div className="flex items-center justify-end gap-3">
        {savedAt && !dirty && (
          <span className="text-xs text-success flex items-center gap-1">
            <Check className="w-3.5 h-3.5" /> Saved
          </span>
        )}
        {mutation.isError && (
          <span className="text-xs text-destructive">{(mutation.error as Error).message}</span>
        )}
        <Button
          onClick={() => mutation.mutate(drafts)}
          disabled={!dirty || mutation.isPending}
          size="sm"
        >
          {mutation.isPending ? <><Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />Saving</> : "Save changes"}
        </Button>
      </div>
    </section>
  );
}

function SecretBadge({ item }: { item: SecretStatus }) {
  if (!item.set) {
    return (
      <span className="text-xs text-muted-foreground px-2 py-0.5 rounded-md border border-border">
        Not set
      </span>
    );
  }
  const label = item.source === "file" ? "Stored" : "From environment";
  return (
    <span className="text-xs text-success flex items-center gap-1">
      <Check className="w-3 h-3" /> {label} · {item.masked}
    </span>
  );
}
