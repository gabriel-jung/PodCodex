import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useMemo, useState } from "react";
import {
  getConfig,
  listShows,
  refreshRSS,
  refreshYouTube,
} from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { StaleUpdatedLabel } from "@/components/common/StaleUpdatedLabel";
import { useLayoutStore } from "@/stores";
import type { ShowSummary } from "@/api/generated-types";
import ShowCard from "@/components/show/ShowCard";
import ShowListRow from "@/components/show/ShowListRow";
import AddShowModal from "@/components/show/AddShowModal";
import { Plus, RefreshCw, List, LayoutGrid, Podcast, Group } from "lucide-react";
import { EmptyState } from "@/components/ui/empty-state";
import AppSidebar from "@/components/layout/AppSidebar";
import EditorialHeader from "@/components/layout/EditorialHeader";
import DropOverlay from "@/components/common/DropOverlay";
import { useTauriFileDrop } from "@/hooks/useTauriFileDrop";
import OnboardingModal from "@/components/OnboardingModal";

const AUDIO_EXTS = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac"];

export default function HomePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: config } = useQuery({ queryKey: queryKeys.config(), queryFn: getConfig });
  const { data: rawShows } = useQuery({
    queryKey: queryKeys.shows(),
    queryFn: listShows,
  });

  const [addOpen, setAddOpen] = useState(false);
  const viewMode = useLayoutStore((s) => s.showViewMode);
  const setViewMode = useLayoutStore((s) => s.setShowViewMode);
  const cardSize = useLayoutStore((s) => s.showCardSize);
  const setCardSize = useLayoutStore((s) => s.setShowCardSize);
  const groupBy = useLayoutStore((s) => s.showGroupBy);
  const setGroupBy = useLayoutStore((s) => s.setShowGroupBy);

  const sorted = useMemo(() => {
    if (!rawShows) return undefined;
    return [...rawShows].sort((a, b) => a.name.localeCompare(b.name));
  }, [rawShows]);

  // Partition shows once, reused for sections and refresh
  const { sections, rssShows, ytShows } = useMemo(() => {
    if (!sorted) return { sections: undefined, rssShows: [] as ShowSummary[], ytShows: [] as ShowSummary[] };
    const rss: ShowSummary[] = [], yt: ShowSummary[] = [], local: ShowSummary[] = [];
    for (const s of sorted) {
      if (s.has_youtube) yt.push(s);
      else if (s.has_rss) rss.push(s);
      else local.push(s);
    }
    const sects = groupBy === "none"
      ? [{ label: "", shows: sorted }]
      : [
          { label: "Podcasts", shows: rss },
          { label: "YouTube", shows: yt },
          { label: "Local", shows: local },
        ].filter((g) => g.shows.length > 0);
    return { sections: sects, rssShows: rss, ytShows: yt };
  }, [sorted, groupBy]);

  // Oldest feed update across all shows with a feed (RSS or YouTube),
  // so the refresh button reflects staleness of either source.
  const oldestFeedUpdate = useMemo(() =>
    [...rssShows, ...ytShows].reduce<string | null>((oldest, s) => {
      if (!s.last_rss_update) return oldest;
      if (!oldest) return s.last_rss_update;
      return s.last_rss_update < oldest ? s.last_rss_update : oldest;
    }, null),
  [rssShows, ytShows]);

  const refreshAllMutation = useMutation({
    mutationFn: async () => {
      await Promise.allSettled([
        ...rssShows.map((s) => refreshRSS(s.path)),
        ...ytShows.map((s) => refreshYouTube(s.path)),
      ]);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
      queryClient.invalidateQueries({ queryKey: queryKeys.episodesAll() });
    },
  });

  const goToShow = useCallback((folder: string) =>
    navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(folder) } }),
  [navigate]);

  const { isHovering } = useTauriFileDrop({
    accept: AUDIO_EXTS,
    onDrop: (paths) => {
      if (paths.length === 0) return;
      navigate({ to: "/file/$path", params: { path: encodeURIComponent(paths[0]) } });
    },
  });

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {isHovering && <DropOverlay message="Drop audio to preview and transcribe" />}
      {sorted && sorted.length === 0 && <OnboardingModal onAddShow={() => setAddOpen(true)} />}
      <EditorialHeader
        title="PodCodex"
        subtitle="Transcribe, translate, search your podcasts."
        fallbackIcon={Podcast}
        stats={[
          ...(sorted && sorted.length > 0
            ? [{ value: sorted.length, label: `show${sorted.length !== 1 ? "s" : ""}` }]
            : []),
          ...(rssShows.length > 0 ? [{ value: rssShows.length, label: "RSS" }] : []),
          ...(ytShows.length > 0 ? [{ value: ytShows.length, label: "YouTube" }] : []),
        ]}
        actions={
          <div className="flex items-center gap-2">
            {(rssShows.length > 0 || ytShows.length > 0) && (
              <Button
                onClick={() => refreshAllMutation.mutate()}
                disabled={refreshAllMutation.isPending}
                variant="outline"
                size="sm"
                title="Refresh all feeds (RSS + YouTube)"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${refreshAllMutation.isPending ? "animate-spin" : ""}`} />
                {refreshAllMutation.isPending ? (
                  "Refreshing..."
                ) : oldestFeedUpdate ? (
                  <StaleUpdatedLabel timestamp={oldestFeedUpdate} />
                ) : (
                  "Update feeds"
                )}
              </Button>
            )}
            <Button onClick={() => setAddOpen(true)} size="sm"><Plus /> Add show</Button>
          </div>
        }
      />

      <div className="flex-1 flex overflow-hidden">
      <AppSidebar />
      <div className="flex-1 overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-8">

        {sections && sections.length > 0 && (
          <>
            {/* Toolbar: group toggle + view toggle + card size */}
            <div className="flex items-center justify-end gap-2 mb-4">
              <button
                onClick={() => setGroupBy(groupBy === "none" ? "source" : "none")}
                className={`px-1.5 py-1 rounded transition ${groupBy !== "none" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
                title="Group by source"
              >
                <Group className="w-3.5 h-3.5" />
              </button>
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

            {sections.map((section) => (
              <div key={section.label || "all"} className={sections.length > 1 ? "mb-6" : ""}>
                {section.label && (
                  <h3 className="text-xs font-medium text-muted-foreground mb-3">{section.label}</h3>
                )}
                {viewMode === "card" ? (
                  <div
                    className="grid gap-4"
                    style={{ gridTemplateColumns: `repeat(${cardSize}, minmax(0, 1fr))` }}
                  >
                    {section.shows.map((show) => (
                      <ShowCard key={show.path} show={show} onClick={goToShow} vertical={cardSize >= 5} />
                    ))}
                  </div>
                ) : (
                  <div className="border border-border rounded-lg overflow-hidden">
                    {section.shows.map((show) => (
                      <ShowListRow key={show.path} show={show} onClick={goToShow} />
                    ))}
                  </div>
                )}
              </div>
            ))}
          </>
        )}

        {sorted && sorted.length === 0 && (
          <EmptyState
            icon={Podcast}
            title="Welcome to PodCodex"
            description="Add a show to transcribe, correct, translate, and search podcast episodes."
            steps={[
              { label: "Add a show from RSS, YouTube, or a local folder" },
              { label: "Download or import episodes" },
              { label: "Transcribe, review, and index for search" },
            ]}
            action={{ label: "Add your first show", onClick: () => setAddOpen(true) }}
          />
        )}

        {addOpen && (
          <AddShowModal
            defaultSavePath={config?.default_save_path || "~"}
            onClose={() => setAddOpen(false)}
            onCreated={(folder) => {
              queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
              setAddOpen(false);
              goToShow(folder);
            }}
            onImported={(_folder) => {
              queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
              setAddOpen(false);
            }}
            onOpenFile={(path) => {
              setAddOpen(false);
              navigate({ to: "/file/$path", params: { path: encodeURIComponent(path) } });
            }}
          />
        )}
      </div>
      </div>
      </div>
    </div>
  );
}
