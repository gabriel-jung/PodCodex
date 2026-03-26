import { useState, useCallback } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Episode, ShowMeta } from "@/api/types";
import {
  getIndexConfig,
  getIndexStatus,
  startIndex,
} from "@/api/client";
import { Button } from "@/components/ui/button";
import { Settings2 } from "lucide-react";
import HelpLabel from "@/components/common/HelpLabel";
import PipelinePanel from "@/components/common/PipelinePanel";

interface IndexPanelProps {
  episode: Episode;
  showMeta?: ShowMeta | null;
}

export default function IndexPanel({ episode, showMeta }: IndexPanelProps) {
  const queryClient = useQueryClient();
  const [taskId, setTaskId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(!episode.indexed);

  const showName = showMeta?.name || episode.audio_path?.split("/").slice(-2, -1)[0] || "show";

  const { data: config } = useQuery({
    queryKey: ["index", "config"],
    queryFn: getIndexConfig,
  });

  const { data: status, refetch: refetchStatus } = useQuery({
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
  const [showAdvanced, setShowAdvanced] = useState(false);

  const invalidate = useCallback(() => {
    refetchStatus();
    queryClient.invalidateQueries({ queryKey: ["index"] });
    queryClient.invalidateQueries({ queryKey: ["episodes"] });
  }, [queryClient, refetchStatus]);

  const handleComplete = () => {
    invalidate();
    setTaskId(null);
    setExpanded(false);
  };

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
    onSuccess: (data) => setTaskId(data.task_id),
  });

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
      done={episode.indexed}
      expanded={expanded}
      onToggle={() => setExpanded(!expanded)}
      rerunLabel="Re-index"
      settingsLabel="Index settings"
      taskId={taskId}
      onTaskComplete={handleComplete}
      emptyMessage="Episode not yet indexed."
      controls={
        <div className="px-4 pb-3 space-y-4">
          {/* Model selection */}
          <div className="space-y-2">
            <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Embedding Models
            </h5>
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
            <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
              Chunking Strategy
            </h5>
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

          {/* Advanced settings — collapsible */}
          <div className="border-t border-border/50 pt-3 space-y-3">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition"
            >
              <Settings2 className="w-3 h-3" />
              <span className="font-semibold uppercase tracking-wide">
                {showAdvanced ? "Hide advanced" : "Advanced settings"}
              </span>
            </button>

            {showAdvanced && (
              <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 items-center text-sm pl-3 border-l-2 border-border">
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
            )}
          </div>

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
                {(startMutation.error as Error).message}
              </p>
            )}
          </div>
        </div>
      }
    >
      {/* Status grid — shown when indexed and controls collapsed */}
      {status && status.combinations.some((c) => c.indexed) && !expanded && !taskId && (
        <div className="p-4 space-y-2">
          <h5 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Indexed combinations
          </h5>
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
