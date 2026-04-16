import { useState, useMemo, useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import type { SearchResult } from "@/api/types";
import { useEpisodeStore, useSearchStore } from "@/stores";
import { getSearchConfig, getIndexStats, searchQuery, exactSearch, randomQuote } from "@/api/client";
import { artworkUrl } from "@/api/filesystem";
import { queryKeys } from "@/api/queryKeys";
import { errorMessage, getShowName } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Settings2, Shuffle } from "lucide-react";
import { useCapabilities } from "@/hooks/useCapabilities";
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
  showName: string;
  folder: string;
  artwork?: string;
}

type SearchPanelProps = EpisodeSearchProps | ShowSearchProps;

export default function SearchPanel(props: SearchPanelProps) {
  const isShowScope = props.scope === "show";
  const storeEpisode = useEpisodeStore((s) => s.episode);
  const storeShowMeta = useEpisodeStore((s) => s.showMeta);
  const storeFolder = useEpisodeStore((s) => s.folder);
  const showName = isShowScope ? props.showName : getShowName(storeShowMeta, storeEpisode?.audio_path ?? null);
  const episode = isShowScope ? undefined : storeEpisode;
  const showFolder = isShowScope ? props.folder : (storeFolder ?? undefined);
  const showArtwork = isShowScope
    ? props.artwork
    : (storeShowMeta?.artwork_url ? artworkUrl(storeFolder ?? "") : undefined);

  const { lastQuery, addToHistory, setLastQuery } = useSearchStore();
  const [query, setQuery] = useState(lastQuery);
  const [mode, setMode] = useState<SearchMode>("semantic");
  const [model, setModel] = useState("bge-m3");
  const [chunking, setChunking] = useState("semantic");
  const [topK, setTopK] = useState(isShowScope ? 10 : 5);
  const [alpha, setAlpha] = useState(0.5);
  const [source, setSource] = useState<string>("");
  const [scopeAll, setScopeAll] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [searchMs, setSearchMs] = useState<number | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { data: config } = useQuery({
    queryKey: queryKeys.searchConfig(),
    queryFn: getSearchConfig,
  });

  const { data: stats } = useQuery({
    queryKey: queryKeys.searchStats(showName),
    queryFn: () => getIndexStats(showName),
    enabled: !!showName && isShowScope,
  });

  const availableModels = useMemo(
    () => Array.from(new Set((stats?.collections ?? []).map((c) => c.model))),
    [stats],
  );
  const availableChunkings = useMemo(
    () => Array.from(new Set((stats?.collections ?? []).map((c) => c.chunking))),
    [stats],
  );
  const availableSources = useMemo(
    () => Array.from(new Set((stats?.collections ?? []).flatMap((c) => c.sources ?? []))).sort(),
    [stats],
  );

  useEffect(() => {
    if (availableModels.length > 0 && !availableModels.includes(model)) {
      setModel(availableModels[0]);
    }
  }, [availableModels, model]);
  useEffect(() => {
    if (availableChunkings.length > 0 && !availableChunkings.includes(chunking)) {
      setChunking(availableChunkings[0]);
    }
  }, [availableChunkings, chunking]);

  const searchMutation = useMutation({
    mutationFn: () => {
      const t0 = performance.now();
      const base = {
        query,
        show: showName,
        model,
        chunking,
        source: source || null,
        ...(!isShowScope && !scopeAll ? { episode: episode?.stem || undefined } : {}),
      };
      const req = mode === "exact"
        ? exactSearch(base)
        : searchQuery({ ...base, top_k: topK, alpha });
      return req.then((data) => { setSearchMs(performance.now() - t0); return data; });
    },
    onSuccess: (data) => setResults(data),
  });

  const randomMutation = useMutation({
    mutationFn: () => {
      const t0 = performance.now();
      return randomQuote({
        show: showName,
        model,
        chunking,
        source: source || null,
        ...(!isShowScope && !scopeAll ? { episode: episode?.stem || undefined } : {}),
      }).then((data) => { setSearchMs(performance.now() - t0); return data; });
    },
    onSuccess: (data) => setResults(data ? [data] : []),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setLastQuery(query);
      addToHistory(query);
      setSubmittedQuery(query);
      searchMutation.mutate();
    }
  };

  const tiers = useMemo(() => {
    const exact = results.filter((r) => !r.accent_match && !r.fuzzy_match);
    const accent = results.filter((r) => r.accent_match);
    const fuzzy = results.filter((r) => r.fuzzy_match);
    const displayed = mode === "exact" ? [...exact, ...accent, ...fuzzy] : results;
    return { exact, accent, fuzzy, displayed };
  }, [results, mode]);

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
          <div className="flex items-center gap-4 text-xs">
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={mode === "exact"}
                onChange={(e) => setMode(e.target.checked ? "exact" : "semantic")}
                className="accent-primary"
              />
              <span className="text-muted-foreground">Exact match only</span>
            </label>

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

          {/* Advanced settings — single inline strip */}
          {showAdvanced && config && (
            <div className="flex items-center gap-3 text-xs flex-wrap pl-3 border-l-2 border-border/60">
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={availableModels.length === 0}
                title="Embedding model (only indexed ones listed)"
                className="bg-transparent border-0 p-0 text-foreground hover:underline focus:outline-none cursor-pointer"
              >
                {availableModels.length === 0 && <option>no model indexed</option>}
                {availableModels.map((key) => (
                  <option key={key} value={key}>{config.models[key]?.label ?? key}</option>
                ))}
              </select>

              <span className="text-muted-foreground/40">·</span>

              <select
                value={chunking}
                onChange={(e) => setChunking(e.target.value)}
                disabled={availableChunkings.length === 0}
                title="Chunking strategy (only indexed ones listed)"
                className="bg-transparent border-0 p-0 text-foreground hover:underline focus:outline-none cursor-pointer"
              >
                {availableChunkings.length === 0 && <option>no chunker indexed</option>}
                {availableChunkings.map((key) => (
                  <option key={key} value={key} title={config.chunking_strategies[key]}>{key}</option>
                ))}
              </select>

              {availableSources.length > 1 && (
                <>
                  <span className="text-muted-foreground/40">·</span>
                  <select
                    value={source}
                    onChange={(e) => setSource(e.target.value)}
                    title="Filter by indexed source version"
                    className="bg-transparent border-0 p-0 text-foreground hover:underline focus:outline-none cursor-pointer"
                  >
                    <option value="">all sources</option>
                    {availableSources.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </>
              )}

              <span className="text-muted-foreground/40">·</span>

              <label className="flex items-center gap-1 text-muted-foreground">
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  min={1}
                  max={50}
                  className="bg-transparent border-0 p-0 w-8 text-foreground text-right focus:outline-none focus:underline"
                />
                <span>results</span>
              </label>

              {mode === "semantic" && (
                <>
                  <span className="text-muted-foreground/40">·</span>
                  <div className="flex items-center gap-2 flex-1 min-w-40">
                    <span className="italic text-muted-foreground/60 text-2xs">keyword</span>
                    <input
                      type="range"
                      value={alpha}
                      onChange={(e) => setAlpha(Number(e.target.value))}
                      min={0}
                      max={1}
                      step={0.1}
                      className="flex-1 accent-primary"
                      title={`Blend: ${alpha.toFixed(1)} (0 = keyword, 1 = meaning)`}
                    />
                    <span className="italic text-muted-foreground/60 text-2xs">meaning</span>
                  </div>
                </>
              )}
            </div>
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
          {hasResults && (() => {
            const { exact, accent, fuzzy, displayed } = tiers;
            const timingLabel = searchMs != null
              ? searchMs < 1000 ? `${searchMs.toFixed(0)} ms` : `${(searchMs / 1000).toFixed(2)} s`
              : null;

            return (
              <div className={`${isShowScope ? "p-6" : "p-4"} space-y-3`}>
                <div className="text-xs text-muted-foreground">
                  {results.length === 0 ? (
                    <>No results found{timingLabel ? ` in ${timingLabel}.` : "."}</>
                  ) : mode === "exact" ? (
                    <>
                      {`Found ${displayed.length} result${displayed.length !== 1 ? "s" : ""}`}
                      {timingLabel ? ` in ${timingLabel}` : ""}
                      {` (`}
                      {exact.length} exact
                      {accent.length > 0 && `, ${accent.length} variant${accent.length !== 1 ? "s" : ""}`}
                      {fuzzy.length > 0 && `, ${fuzzy.length} near-typo${fuzzy.length !== 1 ? "s" : ""}`}
                      {`)`}
                    </>
                  ) : (
                    <>{`Found ${results.length} result${results.length !== 1 ? "s" : ""}`}{timingLabel ? ` in ${timingLabel}.` : "."}</>
                  )}
                </div>
                {displayed.map((r, i) => (
                  <SearchResultCard
                    key={i}
                    result={r}
                    show={{ name: showName, folder: showFolder, artwork: showArtwork }}
                    query={submittedQuery}
                  />
                ))}
              </div>
            );
          })()}
        </div>
      </>)}
    </div>
  );
}
