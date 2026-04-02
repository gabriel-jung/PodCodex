import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import {
  getConfig,
  listShows,
  refreshRSS,
  refreshYouTube,
} from "@/api/client";
import { Button } from "@/components/ui/button";
import { timeAgo } from "@/lib/utils";
import { useConfigStore } from "@/stores/configStore";
import ShowCard from "@/components/show/ShowCard";
import ShowListRow from "@/components/show/ShowListRow";
import AddShowModal from "@/components/show/AddShowModal";
import { Plus, RefreshCw, List, LayoutGrid } from "lucide-react";

export default function HomePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: config } = useQuery({ queryKey: ["config"], queryFn: getConfig });
  const { data: shows } = useQuery({
    queryKey: ["shows"],
    queryFn: listShows,
  });

  const [addOpen, setAddOpen] = useState(false);
  const viewMode = useConfigStore((s) => s.showViewMode);
  const setViewMode = useConfigStore((s) => s.setShowViewMode);
  const cardSize = useConfigStore((s) => s.showCardSize);
  const setCardSize = useConfigStore((s) => s.setShowCardSize);

  const rssShows = shows?.filter((s) => s.has_rss) ?? [];
  const ytShows = shows?.filter((s) => s.has_youtube) ?? [];

  // Oldest RSS update across all shows (to display in the button)
  const oldestRssUpdate = rssShows.reduce<string | null>((oldest, s) => {
    if (!s.last_rss_update) return oldest;
    if (!oldest) return s.last_rss_update;
    return s.last_rss_update < oldest ? s.last_rss_update : oldest;
  }, null);

  const refreshAllMutation = useMutation({
    mutationFn: async () => {
      await Promise.allSettled([
        ...rssShows.map((s) => refreshRSS(s.path)),
        ...ytShows.map((s) => refreshYouTube(s.path)),
      ]);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["shows"] });
      queryClient.invalidateQueries({ queryKey: ["episodes"] });
    },
  });

  const goToShow = (folder: string) =>
    navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(folder) } });

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold">PodCodex</h1>
            <p className="text-sm text-muted-foreground mt-1">Podcast processing pipeline</p>
          </div>
          <div className="flex gap-2">
            {(rssShows.length > 0 || ytShows.length > 0) && (
              <Button
                onClick={() => refreshAllMutation.mutate()}
                disabled={refreshAllMutation.isPending}
                variant="outline"
                size="sm"
                title="Refresh RSS feeds for all shows"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${refreshAllMutation.isPending ? "animate-spin" : ""}`} />
                {refreshAllMutation.isPending
                  ? "Refreshing..."
                  : oldestRssUpdate
                    ? `Updated ${timeAgo(oldestRssUpdate)}`
                    : "Update feeds"}
              </Button>
            )}
            <Button onClick={() => setAddOpen(true)} size="sm"><Plus /> Add show</Button>
          </div>
        </div>

        {shows && shows.length > 0 && (
          <>
            {/* Toolbar: view toggle + card size */}
            <div className="flex items-center justify-end gap-2 mb-4">
              {viewMode === "card" && (
                <input
                  type="range"
                  min={1}
                  max={5}
                  value={cardSize}
                  onChange={(e) => setCardSize(Number(e.target.value))}
                  className="w-16 accent-primary"
                />
              )}
              <div className="flex border border-border rounded overflow-hidden">
                <button
                  onClick={() => setViewMode("list")}
                  className={`px-1.5 py-1 transition ${viewMode === "list" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
                  title="List view"
                >
                  <List className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setViewMode("card")}
                  className={`px-1.5 py-1 transition ${viewMode === "card" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
                  title="Card view"
                >
                  <LayoutGrid className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {viewMode === "card" ? (
              <div
                className="grid gap-4"
                style={{ gridTemplateColumns: `repeat(${cardSize}, minmax(0, 1fr))` }}
              >
                {shows.map((show) => (
                  <ShowCard key={show.path} show={show} onClick={() => goToShow(show.path)} vertical={cardSize >= 5} />
                ))}
              </div>
            ) : (
              <div className="border border-border rounded-lg overflow-hidden">
                {shows.map((show) => (
                  <ShowListRow key={show.path} show={show} onClick={() => goToShow(show.path)} />
                ))}
              </div>
            )}
          </>
        )}

        {shows && shows.length === 0 && (
          <div className="text-center py-20 text-muted-foreground">
            <p className="text-lg mb-2">No shows yet</p>
            <p className="text-sm">Search for a podcast or import an existing folder</p>
          </div>
        )}

        {addOpen && (
          <AddShowModal
            defaultSavePath={config?.default_save_path || "~"}
            onClose={() => setAddOpen(false)}
            onCreated={(folder) => {
              queryClient.invalidateQueries({ queryKey: ["shows"] });
              setAddOpen(false);
              goToShow(folder);
            }}
            onOpenFile={(path) => {
              setAddOpen(false);
              navigate({ to: "/file/$path", params: { path: encodeURIComponent(path) } });
            }}
          />
        )}
      </div>
    </div>
  );
}
