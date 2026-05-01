import { useEffect, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEpisodeStore, useAudioPath, usePipelineConfigStore } from "@/stores";
import {
  getIndexConfig,
  getIndexStatus,
  startIndex,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { getShowName, selectClass, versionOption } from "@/lib/utils";
import { usePipelineTask } from "@/hooks/usePipelineTask";
import { useInputVersions } from "@/hooks/useLLMPipeline";
import { useCapabilities } from "@/hooks/useCapabilities";
import AdvancedToggle from "@/components/common/AdvancedToggle";
import FormGrid from "@/components/common/FormGrid";
import HelpLabel from "@/components/common/HelpLabel";
import MissingDependency from "@/components/common/MissingDependency";
import PipelinePanel from "@/components/common/PipelinePanel";
import PipelineRunFooter from "@/components/common/PipelineRunFooter";
import Segmented from "@/components/common/Segmented";
import IndexInspectorModal from "@/components/index/IndexInspectorModal";
import { Button } from "@/components/ui/button";

export default function IndexPanel() {
  const episode = useEpisodeStore((s) => s.episode);
  const showMeta = useEpisodeStore((s) => s.showMeta);
  const audioPath = useAudioPath();
  const outputDir = episode?.output_dir;
  const showName = getShowName(showMeta, audioPath);
  const task = usePipelineTask(audioPath, "index");

  const { data: config } = useQuery({
    queryKey: queryKeys.indexConfig(),
    queryFn: getIndexConfig,
  });

  const { data: status } = useQuery({
    queryKey: queryKeys.indexStatus(audioPath ?? outputDir, showName),
    queryFn: () => getIndexStatus(audioPath, showName, outputDir),
    enabled: !!audioPath || !!outputDir,
  });

  const expanded = task.expanded || !episode?.indexed;

  const inputVersions = useInputVersions(audioPath, "index", !!episode?.transcribed && expanded, outputDir);

  const [sourceVersionId, setSourceVersionId] = useState<string | null>(null);
  const indexModel = usePipelineConfigStore((s) => s.indexModel);
  const setIndexModel = usePipelineConfigStore((s) => s.setIndexModel);
  const [chunking, setChunking] = useState("semantic");
  const [chunkSize, setChunkSize] = useState(256);
  const [threshold, setThreshold] = useState(0.5);
  const [overwrite, setOverwrite] = useState(!!episode?.indexed);
  const [inspectTarget, setInspectTarget] = useState<{ model: string; chunking: string } | null>(null);
  // Reset on episode switch only — preserve user toggle when status refetches.
  useEffect(() => {
    setOverwrite(!!episode?.indexed);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episode?.id]);

  const startMutation = useMutation({
    mutationFn: () =>
      startIndex({
        audio_path: audioPath!,
        show: showName,
        version_id: sourceVersionId ?? undefined,
        model_keys: [indexModel],
        chunkings: [chunking],
        chunk_size: chunkSize,
        threshold,
        overwrite,
      }),
    onSuccess: (data) => task.startTask(data.task_id),
  });

  const { has: hasCap, isLoaded: capsLoaded } = useCapabilities();
  const hasRAG = hasCap("embeddings") && hasCap("torch");

  if (!episode) return null;

  const prereq = !episode.transcribed
    ? "You need a transcript first. Go to the Transcribe tab to create one."
    : undefined;

  const models = config?.models as Record<string, { label: string; description: string }> | undefined;
  const selectedModelInfo = models?.[indexModel];
  const modelOptions: (readonly [string, string, string?])[] = models
    ? Object.entries(models).map(([key, spec]) => [key, spec.label, spec.description] as const)
    : [[indexModel, indexModel] as const];

  const chunkingOptions: (readonly [string, string, string?])[] = config
    ? Object.entries(config.chunking_strategies).map(([key, desc]) => [key, key, desc as string] as const)
    : [["semantic", "semantic"] as const];

  return (
    <PipelinePanel
      title="Index"
      description="Build a search index so you can find specific moments by meaning, not just keywords."
      prerequisite={prereq}
      blocker={!prereq && capsLoaded && !hasRAG ? (
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
        <div className="px-4 pt-3 pb-4 space-y-4">
          <FormGrid>
            {inputVersions && inputVersions.length > 0 && (
              <>
                <HelpLabel label="Source" help="Which transcript version to index. Corrected transcripts give better search results." />
                <select
                  value={sourceVersionId ?? ""}
                  onChange={(e) => setSourceVersionId(e.target.value || null)}
                  className={`${selectClass} text-xs max-w-full min-w-0`}
                >
                  <option value="">Latest — {versionOption(inputVersions[0])}</option>
                  {inputVersions.map((v) => (
                    <option key={v.id} value={v.id}>{versionOption(v)}</option>
                  ))}
                </select>
              </>
            )}

            <HelpLabel label="Model" help="Embedding model used for semantic search. Larger models give better results but are slower." />
            <div className="space-y-1">
              <Segmented
                value={indexModel}
                onChange={setIndexModel}
                options={modelOptions}
              />
              {selectedModelInfo && (
                <p className="text-xs text-muted-foreground">{selectedModelInfo.description}</p>
              )}
            </div>

            <HelpLabel label="Chunking" help="How transcripts are split into searchable pieces. Semantic groups by meaning, speaker groups by speaker turn." />
            <Segmented
              value={chunking}
              onChange={setChunking}
              options={chunkingOptions}
            />
          </FormGrid>

          <AdvancedToggle className="border-t border-border/50 pt-3 space-y-3">
            <FormGrid className="pl-3 border-l-2 border-border">
              <HelpLabel label="Chunk size" help="Number of tokens per chunk. Smaller chunks give more precise results, larger chunks preserve more context." />
              <input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="input w-20"
                min={64}
                max={1024}
              />
              {chunking === "semantic" && (
                <>
                  <HelpLabel label="Similarity" help="How similar adjacent sentences must be to stay in the same chunk. 0 = always split, 1 = never split." />
                  <input
                    type="number"
                    value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))}
                    className="input w-20"
                    min={0}
                    max={1}
                    step={0.05}
                  />
                </>
              )}
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

          <PipelineRunFooter
            onRun={() => startMutation.mutate()}
            isPending={startMutation.isPending}
            mutationError={startMutation.isError ? startMutation.error : null}
            hasExisting={episode.indexed}
            initialLabel="Index"
            rerunLabel="Re-index"
          />
        </div>
      }
    >
      {status && status.combinations.some((c) => c.indexed) && !expanded && !task.activeTaskId && (
        <div className="p-4 space-y-2">
          <h5 className="text-xs font-medium text-muted-foreground">Indexed combinations</h5>
          <div className="grid gap-2">
            {status.combinations
              .filter((c) => c.indexed)
              .map((c) => (
                <div
                  key={`${c.model}-${c.chunking}`}
                  className="flex items-center gap-3 text-sm px-3 py-2 rounded bg-secondary border border-border"
                >
                  <span className="w-2 h-2 rounded-full bg-success shrink-0" />
                  <span className="font-medium">{models?.[c.model]?.label ?? c.model}</span>
                  <span className="text-muted-foreground">/ {c.chunking}</span>
                  <span className="ml-auto text-muted-foreground tabular-nums">
                    {c.chunk_count} chunks
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7"
                    onClick={() => setInspectTarget({ model: c.model, chunking: c.chunking })}
                  >
                    Inspect
                  </Button>
                </div>
              ))}
          </div>
        </div>
      )}
      {audioPath && inspectTarget && (
        <IndexInspectorModal
          open={!!inspectTarget}
          onClose={() => setInspectTarget(null)}
          audioPath={audioPath}
          show={showName}
          model={inspectTarget.model}
          modelLabel={models?.[inspectTarget.model]?.label ?? inspectTarget.model}
          chunking={inspectTarget.chunking}
        />
      )}
    </PipelinePanel>
  );
}
