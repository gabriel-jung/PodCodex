import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useCallback, useMemo, useRef, useState } from "react";
import { getConfig, listShows } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { FeedRefreshButton } from "@/components/common/FeedRefreshButton";
import { useFeedRefreshAll } from "@/hooks/useFeedRefresh";
import { useLayoutStore } from "@/stores";
import type { FilesImportResponse, ShowSummary } from "@/api/types";
import { AUDIO_EXTENSIONS } from "@/api/types";
import ShowCard from "@/components/show/ShowCard";
import ShowListRow from "@/components/show/ShowListRow";
import CompactToggle from "@/components/show/CompactToggle";
import AddShowModal from "@/components/show/AddShowModal";
import ImportErrorsBanner from "@/components/show/ImportErrorsBanner";
import ImportFileDialog from "@/components/show/ImportFileDialog";
import ImportTargetDialog from "@/components/show/ImportTargetDialog";
import { useAudioImportQueue } from "@/hooks/useAudioImportQueue";
import { Plus, List, LayoutGrid, Podcast, Group } from "lucide-react";
import { EmptyState } from "@/components/ui/empty-state";
import { ErrorAlert } from "@/components/ui/error-alert";
import AppSidebar from "@/components/layout/AppSidebar";
import EditorialHeader from "@/components/layout/EditorialHeader";
import DropOverlay from "@/components/common/DropOverlay";
import { useTauriFileDrop } from "@/hooks/useTauriFileDrop";
import { useUniformCardHeight } from "@/hooks/useUniformCardHeight";
import OnboardingModal from "@/components/OnboardingModal";
import { showCardGridTemplate } from "@/lib/cardGrid";

export default function HomePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: config } = useQuery({ queryKey: queryKeys.config(), queryFn: getConfig });
  const {
    data: rawShows,
    isError: showsFailed,
    error: showsError,
    refetch: refetchShows,
  } = useQuery({
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
  const compact = useLayoutStore((s) => s.compact);
  const cardsContainerRef = useRef<HTMLDivElement | null>(null);

  const sorted = useMemo(() => {
    if (!rawShows) return undefined;
    return [...rawShows].sort((a, b) => a.name.localeCompare(b.name));
  }, [rawShows]);

  // Partition shows once, reused for sections and refresh
  const { sections, rssShows, ytShows, localShows } = useMemo(() => {
    if (!sorted) {
      return {
        sections: undefined,
        rssShows: [] as ShowSummary[],
        ytShows: [] as ShowSummary[],
        localShows: [] as ShowSummary[],
      };
    }
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
    return { sections: sects, rssShows: rss, ytShows: yt, localShows: local };
  }, [sorted, groupBy]);

  // Import destinations come from the server's own flag, not from the
  // has_rss/has_youtube grouping above: the two answer different questions,
  // and gating the picker on presentation state is how it ends up offering
  // a show the import endpoint rejects.
  const importTargets = useMemo(
    () => (sorted ?? []).filter((s) => s.accepts_imports),
    [sorted],
  );

  // Oldest feed update across all shows with a feed (RSS or YouTube),
  // so the refresh button reflects staleness of either source.
  const oldestFeedUpdate = useMemo(() =>
    [...rssShows, ...ytShows].reduce<string | null>((oldest, s) => {
      if (!s.last_rss_update) return oldest;
      if (!oldest) return s.last_rss_update;
      return s.last_rss_update < oldest ? s.last_rss_update : oldest;
    }, null),
  [rssShows, ytShows]);

  const { mutation: refreshAllMutation, refreshingLabel } =
    useFeedRefreshAll(rssShows, ytShows);

  const goToShow = useCallback((folder: string) =>
    navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(folder) } }),
  [navigate]);

  useUniformCardHeight(cardsContainerRef, [
    sorted?.length ?? 0,
    viewMode,
    cardSize,
    compact,
  ]);

  // Standalone-file import: dropped audio first goes through the target
  // picker (any local show, or a new one), then copies one file at a time.
  // A 409 (name taken) pauses the queue and opens the rename dialog; other
  // failures collect into a dismissible banner and the queue moves on. When
  // every file succeeded, one import opens the episode and several open the
  // show; any failure keeps the user here so the banner stays visible.
  const [pendingImport, setPendingImport] = useState<string[] | null>(null);
  const lastImportTarget = useLayoutStore((s) => s.lastImportTarget);
  const setLastImportTarget = useLayoutStore((s) => s.setLastImportTarget);

  const finishImports = useCallback((imported: FilesImportResponse[], errors: string[]) => {
    if (imported.length === 0) return;
    queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
    const folder = imported[imported.length - 1].folder;
    queryClient.invalidateQueries({ queryKey: queryKeys.episodesForFolder(folder) });
    if (errors.length > 0) return;
    if (imported.length === 1) {
      navigate({
        to: "/show/$folder/episode/$stem",
        params: { folder: encodeURIComponent(folder), stem: encodeURIComponent(imported[0].stem) },
      });
    } else {
      navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(folder) } });
    }
  }, [navigate, queryClient]);

  const {
    run: runImports,
    conflict: importConflict,
    resumeAfterConflict,
    skipConflict,
    errors: importErrors,
    dismissErrors,
  } = useAudioImportQueue(finishImports);

  // Queue a drop for the target picker. Merges rather than replaces: a second
  // drop while the picker is still open would otherwise discard the first
  // batch silently, and both batches want the same destination anyway.
  const queueImport = useCallback((paths: string[]) => {
    if (paths.length === 0) return;
    setPendingImport((prev) => (prev ? [...prev, ...paths] : paths));
  }, []);

  const { isHovering } = useTauriFileDrop({
    accept: AUDIO_EXTENSIONS,
    onDrop: queueImport,
  });

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {isHovering && <DropOverlay message="Drop audio to add it to a show" />}
      {sorted && sorted.length === 0 && <OnboardingModal onAddShow={() => setAddOpen(true)} />}
      <EditorialHeader
        title="PodCodex"
        subtitle="Transcribe, translate, search your podcasts."
        artworkUrl="/icon.png?v=5"
        artworkBare
        fallbackIcon={Podcast}
        stats={[
          ...(rssShows.length > 0
            ? [{ value: rssShows.length, label: rssShows.length === 1 ? "podcast" : "podcasts" }]
            : []),
          ...(ytShows.length > 0 ? [{ value: ytShows.length, label: "YouTube" }] : []),
          ...(localShows.length > 0 ? [{ value: localShows.length, label: "local" }] : []),
        ]}
        actions={
          <div className="flex items-center gap-2">
            {(rssShows.length > 0 || ytShows.length > 0) && (
              <FeedRefreshButton
                onRefresh={() => refreshAllMutation.mutate()}
                title={oldestFeedUpdate ? "Refresh all feeds (RSS + YouTube)" : "Update feeds"}
                lastUpdate={oldestFeedUpdate}
                idleLabel="Update feeds"
                refreshingLabel={refreshingLabel}
                labelClassName="hidden md:inline"
              />
            )}
            <Button onClick={() => setAddOpen(true)} size="sm"><Plus /> Add show</Button>
          </div>
        }
      />

      <div className="flex-1 flex flex-col overflow-hidden">
      <AppSidebar />
      <div className="flex-1 overflow-y-auto">
      <div className="px-6 py-8">

        <ImportErrorsBanner errors={importErrors} onDismiss={dismissErrors} className="mb-4" />

        {showsFailed && (
          <ErrorAlert
            error={showsError}
            onRetry={() => void refetchShows()}
            className="mb-4"
          />
        )}

        {sections && sections.length > 0 && (
          <>
            {/* Toolbar: group toggle + view toggle + card size */}
            <div className="flex items-center justify-end gap-2 mb-4">
              <button
                onClick={() => setGroupBy(groupBy === "none" ? "source" : "none")}
                className={`px-1.5 py-1 rounded transition ${groupBy !== "none" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
                title="Group by source"
                aria-label="Group by source"
              >
                <Group className="w-3.5 h-3.5" />
              </button>
              <CompactToggle />
              {viewMode === "card" && (
                <input
                  type="range"
                  min={1}
                  max={5}
                  value={cardSize}
                  onChange={(e) => setCardSize(Number(e.target.value))}
                  className="w-16 accent-primary"
                  aria-label="Card size"
                />
              )}
              <div className="flex border border-border rounded overflow-hidden">
                <button
                  onClick={() => setViewMode("list")}
                  className={`px-1.5 py-1 transition ${viewMode === "list" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
                  title="List view"
                  aria-label="List view"
                >
                  <List className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setViewMode("card")}
                  className={`px-1.5 py-1 transition ${viewMode === "card" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
                  title="Card view"
                  aria-label="Card view"
                >
                  <LayoutGrid className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div ref={cardsContainerRef}>
            {sections.map((section) => (
              <div key={section.label || "all"} className={sections.length > 1 ? "mb-6" : ""}>
                {section.label && (
                  <h3 className="text-xs font-medium text-muted-foreground mb-3">{section.label}</h3>
                )}
                {viewMode === "card" ? (
                  <div
                    className={compact ? "grid gap-2" : "grid gap-4"}
                    style={{ gridTemplateColumns: showCardGridTemplate(cardSize) }}
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
            </div>
          </>
        )}

        {sorted && sorted.length === 0 && !showsFailed && (
          <EmptyState
            icon={Podcast}
            title="Welcome to PodCodex"
            description="Add a show to transcribe, correct, translate, and search podcast episodes."
            steps={[
              { label: "Add a show from RSS, YouTube, or a local folder (or drop an audio file here)" },
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
            onImported={() => {
              queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
              setAddOpen(false);
            }}
            onOpenFile={(path) => {
              setAddOpen(false);
              queueImport([path]);
            }}
          />
        )}

        {pendingImport && (
          <ImportTargetDialog
            fileCount={pendingImport.length}
            localShows={importTargets}
            defaultFolder={lastImportTarget}
            onConfirm={(folder) => {
              setLastImportTarget(folder);
              setPendingImport(null);
              void runImports(pendingImport, folder);
            }}
            onClose={() => setPendingImport(null)}
          />
        )}

        {importConflict && (
          <ImportFileDialog
            conflict={importConflict}
            onImported={resumeAfterConflict}
            onClose={skipConflict}
          />
        )}
      </div>
      </div>
      </div>
    </div>
  );
}
