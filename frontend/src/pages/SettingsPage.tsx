import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { settingsRoute } from "@/router";
import { getModels, deleteModel, getExtras, installExtra, removeExtra, getSecretsStatus, updateSecrets, getHealth } from "@/api/client";
import {
  createApiKey,
  deleteApiKey,
  scanEnvForKeys,
  updateApiKey,
} from "@/api/keys";
import type { APIKeyPublic } from "@/api/keys";
import {
  createProviderProfile,
  deleteProviderProfile,
  updateProviderProfile,
} from "@/api/providerProfiles";
import type { ProviderProfile } from "@/api/providerProfiles";
import { queryKeys } from "@/api/queryKeys";
import type { ExtraInfo } from "@/api/types";
import type { SecretStatus } from "@/api/config";
import { Button } from "@/components/ui/button";
import {
  Trash2, HardDrive, Cpu, RefreshCw, Puzzle, Download, X, Loader2,
  Sun, Moon, Monitor, Keyboard, Palette, Sparkles, Plug,
  KeyRound, Eye, EyeOff, Check, Zap, Plus, Lock, Search, Settings, Wrench,
} from "lucide-react";
import AppSidebar from "@/components/layout/AppSidebar";
import EditorialHeader from "@/components/layout/EditorialHeader";
import IntegrationsPanel from "@/components/settings/IntegrationsPanel";
import BundleExportPanel from "@/components/settings/BundleExportPanel";
import GPUBackendPanel from "@/components/settings/GPUBackendPanel";
import FfmpegPanel from "@/components/settings/FfmpegPanel";
import { useEffect, useMemo, useState } from "react";
import { useTheme } from "@/hooks/useTheme";
import { Kbd } from "@/components/ShortcutsHelp";
import { SHORTCUTS } from "@/lib/shortcuts";
import { NullableNumberInput } from "@/components/ui/number-input";
import { SettingRow, SettingSection } from "@/components/ui/setting-row";
import { useIndexConfig } from "@/hooks/useIndexConfig";
import { useLLMProviders } from "@/hooks/useLLMProviders";
import { useApiKeys } from "@/hooks/useApiKeys";
import { useProviderProfiles } from "@/hooks/useProviderProfiles";
import { usePipelineConfigStore } from "@/stores/pipelineConfigStore";
import { useFlagPatternsStore } from "@/stores/flagPatternsStore";
import { inputWidth, selectClass } from "@/lib/utils";

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
  { key: "ffmpeg", label: "ffmpeg", icon: Wrench },
  { key: "cache", label: "Model cache", icon: HardDrive },
] as const;

type SettingsTab = (typeof ALL_SECTIONS)[number]["key"];

const VALID_TABS: readonly SettingsTab[] = ALL_SECTIONS.map((s) => s.key);

export default function SettingsPage() {
  // settingsRoute.useSearch() reactively returns the validated `?tab=…` so
  // both initial render and in-app navigation (sidebar warning click) land
  // on the right panel without hand-rolled popstate listeners.
  const search = settingsRoute.useSearch();
  const urlTab = search.tab && (VALID_TABS as readonly string[]).includes(search.tab)
    ? (search.tab as SettingsTab)
    : "general";
  const [tab, setTab] = useState<SettingsTab>(urlTab);
  useEffect(() => { setTab(urlTab); }, [urlTab]);
  const { data: health } = useQuery({
    queryKey: queryKeys.health(),
    queryFn: getHealth,
    staleTime: Infinity,
  });
  // Default unknown mode to "bundle" (conservative): hide dev-only tabs
  // until health proves we're in dev. Otherwise the plugins tab flashes
  // visible during the loading window in shipped builds.
  const isBundleMode = health?.mode !== "dev";
  const visibleSections = ALL_SECTIONS.filter((s) => !(isBundleMode && "devOnly" in s && s.devOnly));
  const sections = [{ items: visibleSections }];
  const visibleKeys = visibleSections.map((s) => s.key) as SettingsTab[];

  // If the active tab is hidden in current mode (e.g. bookmarked
  // ?tab=plugins opened in bundle build), fall back to general.
  const tabHidden = !visibleKeys.includes(tab);
  useEffect(() => {
    if (tabHidden) setTab("general");
  }, [tabHidden]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const id = window.location.hash.slice(1);
    if (!id) return;
    // Element may not exist yet on first paint (panels render async data).
    // Poll for ~2s, then give up.
    let cancelled = false;
    const start = Date.now();
    const tryScroll = () => {
      if (cancelled) return;
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
      if (Date.now() - start < 2000) requestAnimationFrame(tryScroll);
    };
    requestAnimationFrame(tryScroll);
    return () => { cancelled = true; };
  }, [tab]);

  const selectTab = (t: SettingsTab) => {
    setTab(t);
    const usp = new URLSearchParams(window.location.search);
    usp.set("tab", t);
    window.history.replaceState(null, "", `?${usp.toString()}${window.location.hash}`);
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <EditorialHeader title="Settings" fallbackIcon={Settings} />
      <div className="flex-1 flex flex-col overflow-hidden">
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
            {tab === "plugins" && !isBundleMode && <PluginsPanel />}
            {tab === "gpu" && <GPUBackendPanel />}
            {tab === "ffmpeg" && <FfmpegPanel />}
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
      <h2 className="text-base font-semibold flex items-center gap-2">
        <Palette className="w-4 h-4" /> Appearance
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
      <h2 className="text-base font-semibold flex items-center gap-2">
        <Keyboard className="w-4 h-4" /> Keyboard shortcuts
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
  const { whisperModels, detectedKeys, pipelineConfig } = useLLMProviders();
  const { profiles } = useProviderProfiles();
  const { keys: pooledKeys } = useApiKeys();
  const apiProfiles = useMemo(() => profiles.filter((p) => p.type !== "ollama"), [profiles]);
  const { data: indexConfig } = useIndexConfig();

  // App-wide defaults. Each show inherits these unless it overrides them in
  // its own Settings; episode and batch runs start from the show's resolved
  // values.
  const transcribe = usePipelineConfigStore((s) => s.appDefaults.transcribe);
  const setTranscribe = usePipelineConfigStore((s) => s.setAppTranscribe);
  const llm = usePipelineConfigStore((s) => s.appDefaults.llm);
  const setLLM = usePipelineConfigStore((s) => s.setAppLLM);
  const targetLang = usePipelineConfigStore((s) => s.appDefaults.targetLang);
  const setTargetLang = usePipelineConfigStore((s) => s.setAppTargetLang);
  const indexModel = usePipelineConfigStore((s) => s.appDefaults.indexModel);
  const setIndexModel = usePipelineConfigStore((s) => s.setAppIndexModel);
  const indexChunker = usePipelineConfigStore((s) => s.appDefaults.indexChunker);
  const setIndexChunker = usePipelineConfigStore((s) => s.setAppIndexChunker);

  return (
    <div className="space-y-8">
      <p className="text-sm text-muted-foreground">
        Defaults for every show on this computer. Each show can override these
        in its own Settings; one-off tweaks made in an episode panel apply to
        that run only.
      </p>

      <SettingSection
        title="Transcription"
        description="How episodes are turned into text."
      >
        <SettingRow
          label="Transcription model"
          help="Bigger models are more accurate but slower. The CPU options work without a graphics card; the GPU options need one."
        >
          <select
            value={transcribe.modelSize}
            onChange={(e) => setTranscribe({ modelSize: e.target.value })}
            className={selectClass}
            disabled={!pipelineConfig}
          >
            {pipelineConfig
              ? Object.entries(whisperModels).map(([key, label]) => (
                  <option key={key} value={key}>{label}</option>
                ))
              : <option>Loading…</option>}
          </select>
        </SettingRow>
        <SettingRow
          label="Identify speakers"
          help="Detect who is talking and label each line. Requires a Hugging Face token."
          below={transcribe.diarize && !detectedKeys.hf_token ? (
            <p className="text-xs text-muted-foreground">
              Hugging Face token needed.{" "}
              <a
                href="?tab=credentials#HF_TOKEN"
                onClick={(e) => {
                  e.preventDefault();
                  window.history.pushState(null, "", "?tab=credentials#HF_TOKEN");
                  window.dispatchEvent(new PopStateEvent("popstate"));
                }}
                className="underline hover:text-foreground"
              >
                Set it up in Credentials
              </a>
              .
            </p>
          ) : undefined}
        >
          <select
            value={transcribe.diarize ? "yes" : "no"}
            onChange={(e) => setTranscribe({ diarize: e.target.value === "yes" })}
            className={selectClass}
          >
            <option value="no">Off</option>
            <option value="yes">On</option>
          </select>
        </SettingRow>
        <SettingRow
          label="Drop low-quality segments"
          help="Remove garbled or off-mic lines that the model itself flags as unreliable. Leave off if you'd rather review and delete them by hand in the editor."
        >
          <select
            value={transcribe.clean ? "yes" : "no"}
            onChange={(e) => setTranscribe({ clean: e.target.value === "yes" })}
            className={selectClass}
          >
            <option value="no">Off</option>
            <option value="yes">On</option>
          </select>
        </SettingRow>
        <SettingRow
          label="GPU batch size"
          help="How many audio chunks WhisperX processes at once on the GPU. Leave blank to auto-pick from VRAM (8 for ≤10 GB, 16 above). Lower it if you hit out-of-memory errors; raise it on big GPUs for more speed."
        >
          <NullableNumberInput
            value={transcribe.batchSize}
            onChange={(batchSize) => setTranscribe({ batchSize })}
            placeholder="Auto"
            min={1}
            className={`input ${inputWidth.numeric}`}
          />
        </SettingRow>
      </SettingSection>

      <SettingSection
        title="AI correction & translation"
        description="The AI that cleans up raw transcripts and translates them."
      >
        <SettingRow
          label="Where the AI runs"
          help="Ollama runs on your own computer. Cloud API uses a paid online provider. Manual lets you copy the prompts and run them yourself."
        >
          <select
            value={llm.mode}
            onChange={(e) => setLLM({ mode: e.target.value as typeof llm.mode })}
            className={selectClass}
          >
            <option value="api">Cloud API</option>
            <option value="ollama">Ollama (local)</option>
            <option value="manual">Manual (copy-paste prompts)</option>
          </select>
        </SettingRow>
        {llm.mode === "api" && (
          <>
            <SettingRow
              label="AI provider"
              help="Which provider profile to use. Manage profiles in Settings → Credentials."
            >
              <select
                value={llm.providerProfile}
                onChange={(e) => setLLM({ providerProfile: e.target.value })}
                className={selectClass}
              >
                <option value="">Pick…</option>
                {apiProfiles.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}{p.builtin ? "" : " (custom)"}
                  </option>
                ))}
              </select>
            </SettingRow>
            <SettingRow
              label="AI API key"
              help="Which saved API key to use. Add keys in Settings → Credentials."
            >
              <select
                value={llm.keyName}
                onChange={(e) => setLLM({ keyName: e.target.value })}
                className={selectClass}
              >
                <option value="">
                  {pooledKeys.length === 0 ? "No keys yet" : "Pick…"}
                </option>
                {pooledKeys.map((k) => (
                  <option key={k.name} value={k.name}>
                    {k.name}
                    {k.suggested_provider ? ` (${k.suggested_provider})` : ""}
                  </option>
                ))}
              </select>
            </SettingRow>
          </>
        )}
        {(llm.mode === "api" || llm.mode === "ollama") && (
          <SettingRow
            label="AI model"
            help={llm.mode === "ollama"
              ? "Model tag served by your local Ollama instance."
              : "Specific model name. Leave blank to use the provider's default."}
          >
            <input
              value={llm.model}
              onChange={(e) => setLLM({ model: e.target.value })}
              placeholder={llm.mode === "ollama" ? "e.g. llama3.1:8b" : "e.g. gpt-4o-mini"}
              className={`input ${inputWidth.medium}`}
            />
          </SettingRow>
        )}
        <SettingRow
          label="Translate into"
          help="Language episodes are translated into."
        >
          <input
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            placeholder="English"
            className={`input ${inputWidth.short}`}
          />
        </SettingRow>
        <SettingRow
          label="Minutes per batch"
          help="Max audio duration per LLM request. Smaller batches stay within model context windows; larger batches are fewer requests but heavier prompts. Episodes panel can override per-run as a batch count."
        >
          <input
            type="number"
            min={1}
            step={1}
            value={llm.batchMinutes}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n) && n > 0) setLLM({ batchMinutes: n });
            }}
            className={`input ${inputWidth.numeric}`}
          />
        </SettingRow>
      </SettingSection>

      <SettingSection
        title="Search index"
        description="The embeddings that make episodes searchable, used by AI search and the MCP server."
      >
        <SettingRow
          label="Embedding model"
          help="The model that turns transcript text into vectors so search can find passages by meaning. AI search queries whichever model is set here."
        >
          <select
            value={indexModel}
            onChange={(e) => setIndexModel(e.target.value)}
            className={selectClass}
            disabled={!indexConfig}
          >
            {indexConfig
              ? Object.entries(indexConfig.models).map(([key, m]) => (
                  <option key={key} value={key}>{m.label}</option>
                ))
              : <option>Loading…</option>}
          </select>
        </SettingRow>
        <SettingRow
          label="Chunking"
          help="How transcripts are split before they are embedded. Semantic groups sentences with similar meaning; speaker groups consecutive lines from the same speaker."
        >
          <select
            value={indexChunker}
            onChange={(e) => setIndexChunker(e.target.value)}
            className={selectClass}
            disabled={!indexConfig}
          >
            {indexConfig
              ? Object.keys(indexConfig.chunking_strategies).map((key) => (
                  <option key={key} value={key}>{key}</option>
                ))
              : <option>Loading…</option>}
          </select>
        </SettingRow>
      </SettingSection>

      <FlagPatternsSection />
    </div>
  );
}

function FlagPatternsSection() {
  const patterns = useFlagPatternsStore((s) => s.patterns);
  const setPatterns = useFlagPatternsStore((s) => s.setPatterns);
  const [draft, setDraft] = useState(patterns.join("\n"));

  return (
    <SettingSection
      title="Auto-flag in editor"
      description="Words or phrases that should be flagged for review when they appear in a transcript, on top of the segments the model already marks as low-confidence."
    >
      <SettingRow
        label="Patterns"
        help="One per line. Case-insensitive substring match. Punctuation-only segments are flagged automatically."
        below={
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
        }
      >
        <span />
      </SettingRow>
    </SettingSection>
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
      // Extras gate which whisper / embedding models pipelineConfig serves.
      qc.invalidateQueries({ queryKey: queryKeys.pipelineConfig() });
      qc.invalidateQueries({ queryKey: queryKeys.models() });
    },
  });

  const removeMut = useMutation({
    mutationFn: (extra: string) => removeExtra(extra),
    onMutate: (extra) => setPendingAction(extra),
    onSettled: () => {
      setPendingAction(null);
      qc.invalidateQueries({ queryKey: queryKeys.capabilities() });
      qc.invalidateQueries({ queryKey: queryKeys.health() });
      qc.invalidateQueries({ queryKey: queryKeys.pipelineConfig() });
      qc.invalidateQueries({ queryKey: queryKeys.models() });
    },
  });

  const extras = data?.extras ?? {};
  const entries = Object.entries(extras) as [string, ExtraInfo][];
  const installedCount = entries.filter(([, v]) => v.installed).length;

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold flex items-center gap-2">
          <Puzzle className="w-4 h-4" /> Plugins
        </h2>
        <Button variant="ghost" size="sm" onClick={() => refetch()} className="h-7">
          <RefreshCw className="w-3.5 h-3.5" />
        </Button>
      </div>

      <p className="text-sm text-muted-foreground">
        PodCodex features are split into optional plugins so you only install
        what you need. Install or remove them here.
      </p>

      <div className="text-xs text-muted-foreground">
        {installedCount} of {entries.length} plugin{entries.length !== 1 ? "s" : ""} installed
      </div>

      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
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
    },
    onSettled: () => setDeleting(null),
  });

  const models = data?.models ?? [];
  const cacheDir = data?.cache_dir ?? "";
  const vram = data?.vram ?? null;
  const totalMB = models.reduce((sum, m) => sum + m.size_mb, 0);

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold flex items-center gap-2">
          <HardDrive className="w-4 h-4" /> Model Cache
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
        <p className="text-sm text-muted-foreground">Loading…</p>
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
    label: "Hugging Face token",
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
    <section className="space-y-10">
      <div>
        <h2 className="text-base font-semibold flex items-center gap-2">
          <KeyRound className="w-4 h-4" /> Credentials
        </h2>
        <p className="text-sm text-muted-foreground mt-1">
          LLM API keys live in a named pool below. The Hugging Face token is
          a singleton (pyannote needs it as an env var). Stored in user-scoped
          files with read/write restricted to your account.
        </p>
      </div>

      <ApiKeysSection />
      <ProviderProfilesSection />

      <div className="space-y-4">
        <div>
          <h3 className="text-sm font-semibold">Hugging Face token</h3>
          <p className="text-sm text-muted-foreground mt-1">
            Required for speaker diarization (pyannote). Saved at{" "}
            {data?.path ?? "~/.config/podcodex/secrets.env"}. Leave blank to rely
            on the same-named environment variable.
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
      </div>
    </section>
  );
}

// ── LLM API key pool ────────────────────────

function ApiKeysSection() {
  const qc = useQueryClient();
  const { keys } = useApiKeys();
  const { profiles } = useProviderProfiles();
  const apiProfileNames = useMemo(
    () => profiles.filter((p) => p.type !== "ollama").map((p) => p.name),
    [profiles],
  );

  const [adding, setAdding] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [draftValue, setDraftValue] = useState("");
  const [draftProvider, setDraftProvider] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    qc.invalidateQueries({ queryKey: queryKeys.apiKeys() });
  };

  const createMut = useMutation({
    mutationFn: createApiKey,
    onSuccess: () => {
      refresh();
      setAdding(false);
      setDraftName("");
      setDraftValue("");
      setDraftProvider("");
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const deleteMut = useMutation({
    mutationFn: deleteApiKey,
    onSuccess: refresh,
  });

  const scanMut = useMutation({
    mutationFn: scanEnvForKeys,
    onSuccess: refresh,
  });

  const submit = () => {
    setError(null);
    createMut.mutate({
      name: draftName.trim(),
      value: draftValue,
      suggested_provider: draftProvider || null,
    });
  };

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">LLM API keys</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Named pool used by the Correct and Translate steps. Each key carries
          an optional provider hint so the LLM picker can prefill the profile.
        </p>
      </div>

      {keys.length === 0 ? (
        <p className="text-sm text-muted-foreground">No keys yet.</p>
      ) : (
        <div className="border border-border rounded-lg divide-y divide-border">
          {keys.map((k) => (
            <ApiKeyRow
              key={k.name}
              entry={k}
              providerOptions={apiProfileNames}
              onDelete={() => deleteMut.mutate(k.name)}
            />
          ))}
        </div>
      )}

      {adding ? (
        <div className="border border-border rounded-lg p-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-xs text-muted-foreground">Name</span>
              <input
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                placeholder="e.g. Work OpenAI"
                className="input mt-1"
                autoFocus
              />
            </label>
            <label className="block">
              <span className="text-xs text-muted-foreground">Used with (optional)</span>
              <select
                value={draftProvider}
                onChange={(e) => setDraftProvider(e.target.value)}
                className={selectClass + " mt-1"}
              >
                <option value="">No hint</option>
                {apiProfileNames.map((p) => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </label>
          </div>
          <label className="block">
            <span className="text-xs text-muted-foreground">Value</span>
            <input
              type="password"
              value={draftValue}
              onChange={(e) => setDraftValue(e.target.value)}
              placeholder="sk-…"
              className="input mt-1 font-mono text-xs"
              autoComplete="off"
              spellCheck={false}
            />
          </label>
          {error && <p className="text-xs text-destructive">{error}</p>}
          <div className="flex items-center gap-2 justify-end">
            <Button variant="ghost" size="sm" onClick={() => { setAdding(false); setError(null); }}>Cancel</Button>
            <Button
              size="sm"
              onClick={submit}
              disabled={!draftName.trim() || !draftValue || createMut.isPending}
            >
              {createMut.isPending ? <><Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />Saving</> : "Add key"}
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={() => setAdding(true)}>
            <Plus className="w-3.5 h-3.5 mr-1" /> Add key
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => scanMut.mutate()}
            disabled={scanMut.isPending}
            title="Find unmanaged *_API_KEY env vars and add them to the pool"
          >
            {scanMut.isPending ? (
              <><Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />Scanning</>
            ) : (
              <><Search className="w-3.5 h-3.5 mr-1" />Re-scan .env for new keys</>
            )}
          </Button>
          {scanMut.isSuccess && scanMut.data && (
            <span className="text-xs text-muted-foreground">
              {scanMut.data.added.length === 0
                ? "No new keys"
                : `Added: ${scanMut.data.added.join(", ")}`}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function ApiKeyRow({
  entry,
  providerOptions,
  onDelete,
}: {
  entry: APIKeyPublic;
  providerOptions: string[];
  onDelete: () => void;
}) {
  const qc = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draftValue, setDraftValue] = useState("");
  const [draftProvider, setDraftProvider] = useState(entry.suggested_provider ?? "");

  const updateMut = useMutation({
    mutationFn: (patch: { value?: string; suggested_provider?: string | null }) =>
      updateApiKey(entry.name, patch),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.apiKeys() });
      setEditing(false);
      setDraftValue("");
    },
  });

  return (
    <div className="px-4 py-3 space-y-2">
      <div className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-sm">{entry.name}</span>
            <span className="text-xs text-muted-foreground font-mono">{entry.masked}</span>
            {entry.source === "env" && (
              <span className="text-2xs px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">from .env</span>
            )}
          </div>
          {entry.suggested_provider && (
            <div className="text-xs text-muted-foreground mt-0.5">
              Suggested provider: <span className="font-medium">{entry.suggested_provider}</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {!editing && (
            <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setEditing(true)}>
              Edit
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
            onClick={onDelete}
            aria-label="Delete key"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>
      {editing && (
        <div className="space-y-2 pl-1">
          <label className="block">
            <span className="text-xs text-muted-foreground">New value (leave blank to keep)</span>
            <input
              type="password"
              value={draftValue}
              onChange={(e) => setDraftValue(e.target.value)}
              placeholder="sk-…"
              className="input mt-1 font-mono text-xs"
              autoComplete="off"
              spellCheck={false}
            />
          </label>
          <label className="block">
            <span className="text-xs text-muted-foreground">Suggested provider</span>
            <select
              value={draftProvider}
              onChange={(e) => setDraftProvider(e.target.value)}
              className={selectClass + " mt-1"}
            >
              <option value="">No hint</option>
              {providerOptions.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </label>
          <div className="flex items-center gap-2 justify-end">
            <Button variant="ghost" size="sm" onClick={() => { setEditing(false); setDraftValue(""); }}>
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={() => {
                const patch: { value?: string; suggested_provider?: string | null } = {};
                if (draftValue) patch.value = draftValue;
                if (draftProvider !== (entry.suggested_provider ?? "")) {
                  patch.suggested_provider = draftProvider || "";
                }
                updateMut.mutate(patch);
              }}
              disabled={updateMut.isPending}
            >
              {updateMut.isPending ? "Saving" : "Save"}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Provider profiles ───────────────────────

function ProviderProfilesSection() {
  const qc = useQueryClient();
  const { profiles } = useProviderProfiles();

  const [adding, setAdding] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [draftUrl, setDraftUrl] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    qc.invalidateQueries({ queryKey: queryKeys.providerProfiles() });
  };

  const createMut = useMutation({
    mutationFn: createProviderProfile,
    onSuccess: () => {
      refresh();
      setAdding(false);
      setDraftName("");
      setDraftUrl("");
      setError(null);
    },
    onError: (err: Error) => setError(err.message),
  });

  const deleteMut = useMutation({
    mutationFn: deleteProviderProfile,
    onSuccess: refresh,
  });

  const updateMut = useMutation({
    mutationFn: ({ name, base_url }: { name: string; base_url: string }) =>
      updateProviderProfile(name, { base_url }),
    onSuccess: refresh,
  });

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Provider profiles</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Built-in profiles are read-only. Add custom OpenAI-compatible profiles
          (Groq, Together, OpenRouter, a self-hosted endpoint) by giving them a
          name and base URL.
        </p>
      </div>
      <div className="border border-border rounded-lg divide-y divide-border">
        {profiles.map((p) => (
          <ProviderProfileRow
            key={p.name}
            profile={p}
            onDelete={() => deleteMut.mutate(p.name)}
            onSaveUrl={(url) => updateMut.mutate({ name: p.name, base_url: url })}
            saving={updateMut.isPending}
          />
        ))}
      </div>

      {adding ? (
        <div className="border border-border rounded-lg p-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-xs text-muted-foreground">Name</span>
              <input
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                placeholder="e.g. Groq"
                className="input mt-1"
                autoFocus
              />
            </label>
            <label className="block">
              <span className="text-xs text-muted-foreground">Base URL</span>
              <input
                value={draftUrl}
                onChange={(e) => setDraftUrl(e.target.value)}
                placeholder="https://api.groq.com/openai/v1"
                className="input mt-1 font-mono text-xs"
                autoComplete="off"
                spellCheck={false}
              />
            </label>
          </div>
          {error && <p className="text-xs text-destructive">{error}</p>}
          <div className="flex items-center gap-2 justify-end">
            <Button variant="ghost" size="sm" onClick={() => { setAdding(false); setError(null); }}>Cancel</Button>
            <Button
              size="sm"
              onClick={() => createMut.mutate({ name: draftName.trim(), base_url: draftUrl.trim() })}
              disabled={!draftName.trim() || !draftUrl.trim() || createMut.isPending}
            >
              {createMut.isPending ? <><Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />Saving</> : "Add profile"}
            </Button>
          </div>
        </div>
      ) : (
        <Button size="sm" onClick={() => setAdding(true)}>
          <Plus className="w-3.5 h-3.5 mr-1" /> Add profile
        </Button>
      )}
    </div>
  );
}

function ProviderProfileRow({
  profile,
  onDelete,
  onSaveUrl,
  saving,
}: {
  profile: ProviderProfile;
  onDelete: () => void;
  onSaveUrl: (url: string) => void;
  saving: boolean;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(profile.base_url ?? "");

  return (
    <div className="px-4 py-3 space-y-2">
      <div className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-sm">{profile.name}</span>
            <span className="text-2xs px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">
              {profile.type}
            </span>
            {profile.builtin && (
              <span className="text-2xs flex items-center gap-1 text-muted-foreground">
                <Lock className="w-3 h-3" /> built-in
              </span>
            )}
          </div>
          {profile.base_url && (
            <div className="text-xs text-muted-foreground font-mono mt-0.5 truncate">
              {profile.base_url}
            </div>
          )}
        </div>
        {!profile.builtin && (
          <div className="flex items-center gap-1 shrink-0">
            {!editing && (
              <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={() => setEditing(true)}>
                Edit
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
              onClick={onDelete}
              aria-label="Delete profile"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
          </div>
        )}
      </div>
      {editing && !profile.builtin && (
        <div className="space-y-2 pl-1">
          <label className="block">
            <span className="text-xs text-muted-foreground">Base URL</span>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              className="input mt-1 font-mono text-xs"
              autoComplete="off"
              spellCheck={false}
            />
          </label>
          <div className="flex items-center gap-2 justify-end">
            <Button variant="ghost" size="sm" onClick={() => { setEditing(false); setDraft(profile.base_url ?? ""); }}>Cancel</Button>
            <Button
              size="sm"
              onClick={() => { onSaveUrl(draft.trim()); setEditing(false); }}
              disabled={saving || !draft.trim()}
            >
              {saving ? "Saving" : "Save"}
            </Button>
          </div>
        </div>
      )}
    </div>
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
