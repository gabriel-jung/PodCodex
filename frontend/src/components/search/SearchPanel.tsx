import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import type { SearchResult } from "@/api/types";
import { useEpisodeStore, useAudioPath, useSearchStore } from "@/stores";
import { getSearchConfig, getIndexStats, searchQuery, exactSearch, randomQuote } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { errorMessage, getShowName, selectClass } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Settings2, Shuffle } from "lucide-react";
import { useCapabilities } from "@/hooks/useCapabilities";
import FormGrid from "@/components/common/FormGrid";
import HelpLabel from "@/components/common/HelpLabel";
import MissingDependency from "@/components/common/MissingDependency";
import SearchResultCard from "./SearchResultCard";

type SearchMode = "semantic" | "exact";

/** Episode-scoped search (from EpisodePage) — reads from store. */
interface EpisodeSearchProps {
  scope: "episode";
}

/** Show-wide search (from ShowPage). */
interface ShowSearchProps {
  scope: "show";
  folder: string;
  showName: string;
}

type SearchPanelProps = EpisodeSearchProps | ShowSearchProps;

export default function SearchPanel(props: SearchPanelProps) {
  const isShowScope = props.scope === "show";
  const storeEpisode = useEpisodeStore((s) => s.episode);
  const storeShowMeta = useEpisodeStore((s) => s.showMeta);
  const storeAudioPath = useAudioPath();
  const showName = isShowScope ? props.showName : getShowName(storeShowMeta, storeEpisode?.audio_path);
  const folder = isShowScope ? props.folder : undefined;
  const episode = isShowScope ? undefined : storeEpisode;
  const audioPath = isShowScope ? undefined : (storeAudioPath ?? undefined);

  const { lastQuery, addToHistory, setLastQuery } = useSearchStore();
  const [query, setQuery] = useState(lastQuery);
  const [mode, setMode] = useState<SearchMode>("semantic");
  const [model, setModel] = useState("bge-m3");
  const [chunking, setChunking] = useState("semantic");
  const [topK, setTopK] = useState(isShowScope ? 10 : 5);
  const [alpha, setAlpha] = useState(0.5);
  const [scopeAll, setScopeAll] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { data: config } = useQuery({
    queryKey: queryKeys.searchConfig(),
    queryFn: getSearchConfig,
  });

  const { data: stats } = useQuery({
    queryKey: queryKeys.searchStats(folder ?? "", showName),
    queryFn: () => getIndexStats(folder!, showName),
    enabled: isShowScope,
  });

  const searchMutation = useMutation({
    mutationFn: () => {
      const base = {
        query,
        show: showName,
        model,
        chunking,
        top_k: topK,
        ...(folder ? { folder } : { audio_path: audioPath! }),
        ...(!isShowScope && !scopeAll ? { episode: episode?.stem || undefined } : {}),
      };
      return mode === "exact"
        ? exactSearch(base)
        : searchQuery({ ...base, alpha });
    },
    onSuccess: (data) => setResults(data),
  });

  const randomMutation = useMutation({
    mutationFn: () =>
      randomQuote({
        show: showName,
        model,
        chunking,
        ...(folder ? { folder } : { audio_path: audioPath! }),
        ...(!isShowScope && !scopeAll ? { episode: episode?.stem || undefined } : {}),
      }),
    onSuccess: (data) => setResults(data ? [data] : []),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setLastQuery(query);
      addToHistory(query);
      searchMutation.mutate();
    }
  };

  const isPending = searchMutation.isPending || randomMutation.isPending;
  const isError = searchMutation.isError || randomMutation.isError;
  const error = searchMutation.error || randomMutation.error;
  const hasResults = searchMutation.isSuccess || randomMutation.isSuccess;

  const { has: hasCap } = useCapabilities();
  const hasRAG = hasCap("embeddings") && hasCap("torch");

  // Prerequisite checks
  const prereq = isShowScope
    ? (stats?.total_chunks ?? 0) === 0
      ? "No indexed episodes yet. Index episodes first from the episode page."
      : undefined
    : !episode?.indexed
        ? "You need to build a search index first. Go to the Index tab."
        : undefined;

  const px = isShowScope ? "px-6" : "px-4";

  return (
    <div className="flex flex-col h-full">
      {/* Header (episode scope only) */}
      {!isShowScope && (
        <div className="px-4 pt-3 pb-2 border-b border-border">
          <h3 className="text-sm font-semibold">Search</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Search across indexed segments by meaning or keywords.
          </p>
        </div>
      )}

      {/* Stats bar (show scope only) */}
      {isShowScope && stats && stats.total_chunks > 0 && (
        <div className="px-6 py-2 border-b border-border flex items-center gap-4 text-xs text-muted-foreground">
          <span>{stats.total_episodes} episode{stats.total_episodes !== 1 ? "s" : ""} indexed</span>
          <span>{stats.total_chunks} chunks</span>
          {stats.collections.length > 1 && (
            <span>{stats.collections.length} collections</span>
          )}
        </div>
      )}

      {!hasRAG ? (
        <div className={`${isShowScope ? "p-12" : "p-6"}`}>
          <MissingDependency
            extra="rag"
            label="Search & indexing libraries"
            description="Semantic search requires torch, sentence-transformers, and other dependencies from the rag extra."
          />
        </div>
      ) : prereq ? (
        <div className={`${isShowScope ? "p-12 text-center" : "p-6"} text-muted-foreground text-sm`}>{prereq}</div>
      ) : (<>
        {/* Query bar */}
        <div className={`${px} py-3 border-b border-border space-y-2`}>
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={mode === "exact" ? "Exact text to find..." : isShowScope ? "Search by meaning across all episodes..." : "Search by meaning..."}
              className="flex-1 input py-1.5 text-sm"
            />
            <Button type="submit" size="sm" disabled={!query.trim() || isPending}>
              {isPending ? "Searching..." : "Search"}
            </Button>
            <Button
              type="button"
              onClick={() => randomMutation.mutate()}
              disabled={isPending}
              variant="outline"
              size="sm"
              title="Random quote"
            >
              <Shuffle className="w-3.5 h-3.5" />
            </Button>
          </form>

          {/* Mode + scope + advanced toggle */}
          <div className="flex items-center gap-3 text-xs">
            <div className="flex gap-2">
              {(["semantic", "exact"] as const).map((m) => (
                <label key={m} className="flex items-center gap-1 cursor-pointer">
                  <input
                    type="radio"
                    checked={mode === m}
                    onChange={() => setMode(m)}
                    className="accent-primary"
                  />
                  <span className={mode === m ? "text-foreground" : "text-muted-foreground"}>
                    {m === "semantic" ? "Semantic" : "Exact match"}
                  </span>
                </label>
              ))}
            </div>

            {/* Scope toggle (episode scope only) */}
            {!isShowScope && (
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={scopeAll}
                  onChange={(e) => setScopeAll(e.target.checked)}
                  className="accent-primary"
                />
                <span className="text-muted-foreground">All episodes</span>
              </label>
            )}

            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition ml-auto"
            >
              <Settings2 className="w-3 h-3" />
              <span className="font-medium">
                {showAdvanced ? "Hide advanced" : "Advanced settings"}
              </span>
            </button>
          </div>

          {/* Advanced settings */}
          {showAdvanced && (
            <FormGrid className="pl-3 border-l-2 border-border">
              <HelpLabel label="Embedding model" help="Embedding model used during indexing. Must match what you selected in the Index tab." />
              {config ? (
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className={selectClass}
                >
                  {Object.entries(config.models).map(([key, spec]) => (
                    <option key={key} value={key}>{spec.label}</option>
                  ))}
                </select>
              ) : (
                <span className="text-muted-foreground text-xs">Loading...</span>
              )}

              <HelpLabel label="Chunking" help="Chunking strategy used during indexing. Must match what you selected in the Index tab." />
              {config ? (
                <select
                  value={chunking}
                  onChange={(e) => setChunking(e.target.value)}
                  className={selectClass}
                >
                  {Object.entries(config.chunking_strategies).map(([key, desc]) => (
                    <option key={key} value={key} title={desc}>{key}</option>
                  ))}
                </select>
              ) : (
                <span className="text-muted-foreground text-xs">Loading...</span>
              )}

              <HelpLabel label="Results" help="Maximum number of results to return." />
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="input w-20"
                min={1}
                max={50}
              />

              {mode === "semantic" && (
                <>
                  <HelpLabel label="Keyword vs meaning" help="Balance between keyword and semantic search. 0 = pure keywords, 1 = pure meaning, 0.5 = balanced mix." />
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      value={alpha}
                      onChange={(e) => setAlpha(Number(e.target.value))}
                      min={0}
                      max={1}
                      step={0.1}
                      className="flex-1"
                    />
                    <span className="text-xs text-muted-foreground w-6 text-right">{alpha.toFixed(1)}</span>
                  </div>
                </>
              )}
            </FormGrid>
          )}
        </div>

        {/* Error */}
        {isError && (
          <div className={`${px} py-2`}>
            <p className="text-destructive text-xs">{errorMessage(error)}</p>
          </div>
        )}

        {/* Results */}
        <div className="flex-1 overflow-y-auto">
          {results.length > 0 && (
            <div className={`${isShowScope ? "p-6" : "p-4"} space-y-3`}>
              <p className="text-xs text-muted-foreground">
                {results.length} result{results.length !== 1 ? "s" : ""}
              </p>
              {results.map((r, i) => (
                <SearchResultCard key={i} result={r} audioPath={audioPath} />
              ))}
            </div>
          )}

          {hasResults && results.length === 0 && (
            <div className="p-6 text-muted-foreground text-sm">No results found.</div>
          )}
        </div>
      </>)}
    </div>
  );
}
