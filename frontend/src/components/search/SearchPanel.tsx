import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import type { Episode, ShowMeta, SearchResult } from "@/api/types";
import { getSearchConfig, searchQuery } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Settings2 } from "lucide-react";
import HelpLabel from "@/components/common/HelpLabel";
import SearchResultCard from "./SearchResultCard";

interface SearchPanelProps {
  episode: Episode;
  showMeta?: ShowMeta | null;
}

export default function SearchPanel({ episode, showMeta }: SearchPanelProps) {
  const showName = showMeta?.name || episode.audio_path?.split("/").slice(-2, -1)[0] || "show";

  const [query, setQuery] = useState("");
  const [model, setModel] = useState("bge-m3");
  const [chunking, setChunking] = useState("semantic");
  const [topK, setTopK] = useState(5);
  const [alpha, setAlpha] = useState(0.5);
  const [scopeAll, setScopeAll] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { data: config } = useQuery({
    queryKey: ["search", "config"],
    queryFn: getSearchConfig,
  });

  const searchMutation = useMutation({
    mutationFn: () =>
      searchQuery({
        query,
        audio_path: episode.audio_path!,
        show: showName,
        model,
        chunking,
        top_k: topK,
        alpha,
        episode: scopeAll ? undefined : (episode.stem || undefined),
      }),
    onSuccess: (data) => setResults(data),
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) searchMutation.mutate();
  };

  const prereq = !episode.audio_path
    ? "Download the audio file first before searching."
    : !episode.indexed
      ? "You need to build a search index first. Go to the Index tab."
      : undefined;

  return (
    <div className="flex flex-col h-full">
      {/* Step header */}
      <div className="px-4 pt-3 pb-2 border-b border-border">
        <h3 className="text-sm font-semibold">Search</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Search across indexed segments by meaning or keywords.
        </p>
      </div>

      {prereq ? (
        <div className="p-6 text-muted-foreground">{prereq}</div>
      ) : (<>

      {/* Query input */}
      <div className="px-4 py-3 border-b border-border">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search indexed content..."
            className="flex-1 bg-secondary text-secondary-foreground rounded px-3 py-1.5 border border-border text-sm focus:outline-none focus:ring-1 focus:ring-primary"
          />
          <Button type="submit" size="sm" disabled={!query.trim() || searchMutation.isPending}>
            {searchMutation.isPending ? "Searching..." : "Search"}
          </Button>
        </form>

        {/* Scope + advanced toggle */}
        <div className="flex items-center gap-4 mt-2 text-sm">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={scopeAll}
              onChange={(e) => setScopeAll(e.target.checked)}
              className="accent-primary"
            />
            <span className="text-xs text-muted-foreground">All episodes</span>
          </label>

          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition ml-auto"
          >
            <Settings2 className="w-3 h-3" />
            <span className="font-semibold uppercase tracking-wide">
              {showAdvanced ? "Hide advanced" : "Advanced settings"}
            </span>
          </button>
        </div>

        {/* Advanced filters */}
        {showAdvanced && (
          <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 items-center text-sm mt-3 pl-3 border-l-2 border-border">
            <HelpLabel label="Embedding model" help="Which model was used to index. Must match what you selected in the Index tab." />
            {config ? (
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
              >
                {Object.entries(config.models).map(([key, spec]) => (
                  <option key={key} value={key}>{spec.label}</option>
                ))}
              </select>
            ) : (
              <span className="text-muted-foreground text-xs">Loading...</span>
            )}

            <HelpLabel label="Chunking" help="Which chunking strategy was used during indexing. Must match what you selected in the Index tab." />
            {config ? (
              <select
                value={chunking}
                onChange={(e) => setChunking(e.target.value)}
                className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border text-sm"
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
              className="input py-1 text-sm w-20"
              min={1}
              max={50}
            />

            <HelpLabel label="Keyword vs meaning" help="0 = pure keyword matching, 1 = pure meaning-based. 0.5 is a balanced mix of both." />
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
          </div>
        )}
      </div>

      {/* Error */}
      {searchMutation.isError && (
        <div className="px-4 py-2">
          <p className="text-destructive text-xs">{(searchMutation.error as Error).message}</p>
        </div>
      )}

      {/* Results — scrollable */}
      <div className="flex-1 overflow-y-auto">
        {results.length > 0 && (
          <div className="p-4 space-y-3">
            <p className="text-xs text-muted-foreground">
              {results.length} result{results.length !== 1 ? "s" : ""}
            </p>
            {results.map((r, i) => (
              <SearchResultCard key={i} result={r} audioPath={episode.audio_path!} />
            ))}
          </div>
        )}

        {searchMutation.isSuccess && results.length === 0 && (
          <div className="p-6 text-muted-foreground text-sm">No results found.</div>
        )}
      </div>
      </>)}
    </div>
  );
}
