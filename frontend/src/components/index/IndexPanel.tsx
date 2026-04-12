import { useState, useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEpisodeStore, useAudioPath, usePipelineConfigStore } from "@/stores";
import { INDEX_PRESETS } from "@/stores/pipelineConfigStore";
import {
  getIndexConfig,
  getIndexStatus,
  getIndexSources,
  getStepVersions,
  startIndex,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { errorMessage, getShowName, selectClass, versionOption, versionInfo } from "@/lib/utils";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { Button } from "@/components/ui/button";
import { useCapabilities } from "@/hooks/useCapabilities";
import AdvancedToggle from "@/components/common/AdvancedToggle";
import FormGrid from "@/components/common/FormGrid";
import HelpLabel from "@/components/common/HelpLabel";
import MissingDependency from "@/components/common/MissingDependency";
import SectionHeader from "@/components/common/SectionHeader";
import PipelinePanel from "@/components/common/PipelinePanel";
import PresetCards from "@/components/common/PresetCards";

export default function IndexPanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const audioPath = useAudioPath();
  const showName = getShowName(showMeta, audioPath);
  const task = usePipelineTask(audioPath, "index");

  const { data: config } = useQuery({
    queryKey: queryKeys.indexConfig(),
    queryFn: getIndexConfig,
  });

  const { data: status } = useQuery({
    queryKey: queryKeys.indexStatus(audioPath, showName),
    queryFn: () => getIndexStatus(audioPath!, showName),
    enabled: !!audioPath,
  });

  const { data: sources } = useQuery({
    queryKey: queryKeys.indexSources(audioPath),
    queryFn: () => getIndexSources(audioPath!),
    enabled: !!audioPath,
  });

  // Pre-select the most advanced available source (first with exists=true)
  const availableSources = sources?.filter((s) => s.exists) ?? [];
  const defaultSource = availableSources[0]?.key ?? "transcript";

  // Form state
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const source = selectedSource ?? defaultSource;
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const storeIndexModel = usePipelineConfigStore((s) => s.indexModel);
  const indexPreset = usePipelineConfigStore((s) => s.indexPreset);
  const applyIndexPreset = usePipelineConfigStore((s) => s.applyIndexPreset);
  const [selectedModels, setSelectedModels] = useState<string[]>([storeIndexModel]);
  useEffect(() => setSelectedModels([storeIndexModel]), [storeIndexModel]);
  const [selectedChunkings, setSelectedChunkings] = useState<string[]>(["semantic"]);
  const [chunkSize, setChunkSize] = useState(256);
  const [threshold, setThreshold] = useState(0.5);
  const [overwrite, setOverwrite] = useState(!!episode?.indexed);

  // Load versions for the selected step
  const { data: stepVersions } = useQuery({
    queryKey: queryKeys.indexStepVersions(audioPath, source),
    queryFn: () => getStepVersions(audioPath!, source),
    enabled: !!audioPath && availableSources.length > 0,
  });

  // Resolve the currently-selected version entry (null = "Latest")
  const selectedVersion = selectedVersionId
    ? stepVersions?.find((v) => v.id === selectedVersionId) ?? null
    : stepVersions?.[0] ?? null;

  const handleSourceChange = (key: string) => {
    setSelectedSource(key);
    setSelectedVersionId(null); // reset version when step changes
  };

  const startMutation = useMutation({
    mutationFn: () =>
      startIndex({
        audio_path: audioPath!,
        show: showName,
        source,
        version_id: selectedVersionId,
        model_keys: selectedModels,
        chunkings: selectedChunkings,
        chunk_size: chunkSize,
        threshold,
        overwrite,
      }),
    onSuccess: (data) => task.startTask(data.task_id),
  });

  const { has: hasCap } = useCapabilities();
  const hasRAG = hasCap("embeddings") && hasCap("torch");

  if (!episode) return null;
  const expanded = task.expanded || !episode.indexed;

  const prereq = !episode.transcribed
    ? "You need a transcript first. Go to the Transcribe tab to create one."
    : undefined;

  const toggleModel = (key: string) => {
    setSelectedModels((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  };

  const toggleChunking = (key: string) => {
    setSelectedChunkings((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  };

  return (
    <PipelinePanel
      title="Index"
      description="Build a search index so you can find specific moments by meaning, not just keywords. Required before using the Search tab."
      prerequisite={prereq}
      blocker={!prereq && !hasRAG ? (
        <MissingDependency
          extra="rag"
          label="Search & indexing libraries"
          description="Semantic search requires torch, sentence-transformers, and other dependencies from the rag extra."
        />
      ) : undefined}
      done={episode.indexed}
      expanded={expanded}
      onToggle={() => task.setExpanded(!expanded)}
      rerunLabel="Re-index"
      settingsLabel="Index settings"
      taskId={task.activeTaskId}
      onTaskComplete={task.handleComplete}
      onRetry={task.handleRetry}
      onDismiss={task.handleDismiss}
      emptyMessage="Episode not yet indexed."
      controls={
        <div className="px-4 pb-3 space-y-4">
          <PresetCards presets={INDEX_PRESETS} active={indexPreset} onSelect={applyIndexPreset} />

          {/* Source selection: step + version */}
          {availableSources.length > 0 && (
            <div className="space-y-2">
              <SectionHeader>Source</SectionHeader>
              <div className="flex flex-wrap items-center gap-2">
                <select
                  value={source}
                  onChange={(e) => handleSourceChange(e.target.value)}
                  className={selectClass}
                >
                  {availableSources.map((s) => (
                    <option key={s.key} value={s.key}>{s.label}</option>
                  ))}
                </select>

                {stepVersions && stepVersions.length > 0 && (
                  <select
                    value={selectedVersionId ?? ""}
                    onChange={(e) => setSelectedVersionId(e.target.value || null)}
                    className={`${selectClass} flex-1 min-w-0`}
                  >
                    <option value="">Latest</option>
                    {stepVersions.map((v) => (
                      <option key={v.id} value={v.id}>
                        {versionOption(v)}
                      </option>
                    ))}
                  </select>
                )}
                {selectedVersion && (
                  <VersionDetails version={selectedVersion} />
                )}
              </div>
            </div>
          )}

          {/* Model selection */}
          <div className="space-y-2">
            <SectionHeader>Embedding Models</SectionHeader>
            <div className="flex flex-wrap gap-2">
              {config &&
                Object.entries(config.models).map(([key, spec]) => (
                  <button
                    key={key}
                    onClick={() => toggleModel(key)}
                    className={`text-xs px-3 py-1.5 rounded-full border transition ${
                      selectedModels.includes(key)
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-secondary text-secondary-foreground border-border hover:border-foreground/30"
                    }`}
                    title={spec.description}
                  >
                    {spec.label}
                  </button>
                ))}
            </div>
          </div>

          {/* Chunking selection */}
          <div className="space-y-2">
            <SectionHeader>Chunking Strategy</SectionHeader>
            <div className="flex flex-wrap gap-2">
              {config &&
                Object.entries(config.chunking_strategies).map(([key, desc]) => (
                  <button
                    key={key}
                    onClick={() => toggleChunking(key)}
                    className={`text-xs px-3 py-1.5 rounded-full border transition ${
                      selectedChunkings.includes(key)
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-secondary text-secondary-foreground border-border hover:border-foreground/30"
                    }`}
                    title={desc}
                  >
                    {key}
                  </button>
                ))}
            </div>
          </div>

          {/* Advanced settings */}
          <AdvancedToggle className="border-t border-border/50 pt-3 space-y-3">
            <FormGrid className="pl-3 border-l-2 border-border">
              <HelpLabel label="Chunk size" help="Number of tokens per chunk. Smaller chunks give more precise search results, larger chunks preserve more context." />
              <input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="input py-1 text-sm w-20"
                min={64}
                max={1024}
              />
              <HelpLabel label="Similarity threshold" help="For semantic chunking: how similar adjacent sentences must be to stay in the same chunk. 0 = always split, 1 = never split." />
              <input
                type="number"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="input py-1 text-sm w-20"
                min={0}
                max={1}
                step={0.05}
              />
              <HelpLabel label="Overwrite" help="Re-index from scratch even if this episode was already indexed." />
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={overwrite}
                  onChange={(e) => setOverwrite(e.target.checked)}
                  className="accent-primary"
                />
                <span className="text-xs text-muted-foreground">Replace existing index</span>
              </label>
            </FormGrid>
          </AdvancedToggle>

          {/* Run button */}
          <div className="border-t border-border/50 pt-3">
            <Button
              onClick={() => startMutation.mutate()}
              disabled={
                startMutation.isPending ||
                selectedModels.length === 0 ||
                selectedChunkings.length === 0
              }
              size="sm"
            >
              {startMutation.isPending ? "Starting..." : episode.indexed ? "Re-index" : "Index"}
            </Button>
            {startMutation.isError && (
              <p className="text-destructive text-xs mt-1">
                {errorMessage(startMutation.error)}
              </p>
            )}
          </div>
        </div>
      }
    >
      {/* Status grid */}
      {status && status.combinations.some((c) => c.indexed) && !expanded && !task.activeTaskId && (
        <div className="p-4 space-y-2">
          <SectionHeader>Indexed combinations</SectionHeader>
          <div className="grid gap-2">
            {status.combinations
              .filter((c) => c.indexed)
              .map((c) => (
                <div
                  key={`${c.model}-${c.chunking}`}
                  className="flex items-center gap-3 text-sm px-3 py-2 rounded bg-secondary border border-border"
                >
                  <span className="w-2 h-2 rounded-full bg-success shrink-0" />
                  <span className="font-medium">{c.model}</span>
                  <span className="text-muted-foreground">/ {c.chunking}</span>
                  <span className="ml-auto text-muted-foreground">
                    {c.chunk_count} chunks
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </PipelinePanel>
  );
}

function VersionDetails({ version }: { version: import("@/api/types").VersionEntry }) {
  const [expanded, setExpanded] = useState(false);
  const items = versionInfo(version);
  if (items.length === 0) return null;
  return (
    <>
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-xs text-muted-foreground/60 hover:text-muted-foreground transition shrink-0"
      >
        File details
      </button>
      {expanded && (
        <div className="w-full bg-secondary/50 rounded border border-border/50 px-3 py-2 text-xs space-y-0.5">
          {items.map(({ key, value }) => (
            <div key={key} className="flex gap-2">
              <span className="text-muted-foreground shrink-0 w-20">{key}</span>
              <span className="truncate">{value}</span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
