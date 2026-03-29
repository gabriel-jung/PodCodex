import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEpisodeStore } from "@/stores";
import {
  getIndexConfig,
  getIndexStatus,
  startIndex,
} from "@/api/client";
import { errorMessage, getShowName } from "@/lib/utils";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { Button } from "@/components/ui/button";
import { useCapabilities } from "@/hooks/useCapabilities";
import AdvancedToggle from "@/components/common/AdvancedToggle";
import HelpLabel from "@/components/common/HelpLabel";
import MissingDependency from "@/components/common/MissingDependency";
import SectionHeader from "@/components/common/SectionHeader";
import PipelinePanel from "@/components/common/PipelinePanel";

export default function IndexPanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  if (!episode) return null;
  const showName = getShowName(showMeta, episode.audio_path);
  const task = usePipelineTask(episode.audio_path, "index");
  const expanded = task.expanded || !episode.indexed;

  const { data: config } = useQuery({
    queryKey: ["index", "config"],
    queryFn: getIndexConfig,
  });

  const { data: status } = useQuery({
    queryKey: ["index", "status", episode.audio_path, showName],
    queryFn: () => getIndexStatus(episode.audio_path!, showName),
    enabled: !!episode.audio_path,
  });

  // Form state
  const [selectedModels, setSelectedModels] = useState<string[]>(["bge-m3"]);
  const [selectedChunkings, setSelectedChunkings] = useState<string[]>(["semantic"]);
  const [chunkSize, setChunkSize] = useState(256);
  const [threshold, setThreshold] = useState(0.5);
  const [overwrite, setOverwrite] = useState(false);

  const startMutation = useMutation({
    mutationFn: () =>
      startIndex({
        audio_path: episode.audio_path!,
        show: showName,
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

  const prereq = !episode.audio_path
    ? "Download the audio file first before indexing."
    : !episode.transcribed
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
            <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-x-4 gap-y-2 sm:gap-y-3 items-start sm:items-center text-sm pl-3 border-l-2 border-border">
              <HelpLabel label="Chunk size" help="Number of tokens per chunk. Smaller chunks give more precise search results, larger chunks preserve more context." />
              <input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="input py-1 text-sm w-20"
                min={64}
                max={1024}
              />
              <HelpLabel label="Similarity threshold" help="For semantic chunking: how similar adjacent sentences must be to stay in the same chunk (0 = always split, 1 = never split)." />
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
            </div>
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
                  <span className="w-2 h-2 rounded-full bg-green-500 shrink-0" />
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
