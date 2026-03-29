import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import {
  getConfig,
  listShows,
  registerShow,
  refreshRSS,
} from "@/api/client";
import { Button } from "@/components/ui/button";
import { timeAgo } from "@/lib/utils";
import FolderPicker from "@/components/common/FolderPicker";
import ShowCard from "@/components/show/ShowCard";
import AddShowModal from "@/components/show/AddShowModal";
import { Plus, FolderOpen, FileAudio, RefreshCw } from "lucide-react";

export default function HomePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: config } = useQuery({ queryKey: ["config"], queryFn: getConfig });
  const { data: shows } = useQuery({
    queryKey: ["shows"],
    queryFn: listShows,
  });

  const [addMode, setAddMode] = useState<"rss" | "import" | "file" | null>(null);

  const rssShows = shows?.filter((s) => s.has_rss) ?? [];

  // Oldest RSS update across all shows (to display in the button)
  const oldestRssUpdate = rssShows.reduce<string | null>((oldest, s) => {
    if (!s.last_rss_update) return oldest;
    if (!oldest) return s.last_rss_update;
    return s.last_rss_update < oldest ? s.last_rss_update : oldest;
  }, null);

  const refreshAllMutation = useMutation({
    mutationFn: async () => {
      await Promise.allSettled(rssShows.map((s) => refreshRSS(s.path)));
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["shows"] });
      queryClient.invalidateQueries({ queryKey: ["episodes"] });
    },
  });

  const goToShow = (folder: string) =>
    navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(folder) } });

  const handleImport = async (path: string) => {
    await registerShow(path);
    queryClient.invalidateQueries({ queryKey: ["shows"] });
    setAddMode(null);
    goToShow(path);
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-2xl font-bold">PodCodex</h1>
            <p className="text-sm text-muted-foreground mt-1">Podcast processing pipeline</p>
          </div>
          <div className="flex gap-2">
            {rssShows.length > 0 && (
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
            <Button onClick={() => setAddMode("rss")} size="sm"><Plus /> Add podcast</Button>
            <Button onClick={() => setAddMode("import")} variant="outline" size="sm">
              <FolderOpen /> Import folder
            </Button>
            <Button onClick={() => setAddMode("file")} variant="outline" size="sm">
              <FileAudio /> Open file
            </Button>
          </div>
        </div>

        {shows && shows.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {shows.map((show) => (
              <ShowCard key={show.path} show={show} onClick={() => goToShow(show.path)} />
            ))}
          </div>
        )}

        {shows && shows.length === 0 && (
          <div className="text-center py-20 text-muted-foreground">
            <p className="text-lg mb-2">No shows yet</p>
            <p className="text-sm">Search for a podcast or import an existing folder</p>
          </div>
        )}

        {addMode === "rss" && (
          <AddShowModal
            defaultSavePath={config?.default_save_path || "~"}
            onClose={() => setAddMode(null)}
            onCreated={(folder) => {
              queryClient.invalidateQueries({ queryKey: ["shows"] });
              setAddMode(null);
              goToShow(folder);
            }}
          />
        )}

        {addMode === "import" && (
          <FolderPicker
            open
            title="Import an existing folder"
            description="Select a folder that already contains podcast episodes or transcripts."
            onClose={() => setAddMode(null)}
            onSelect={handleImport}
            initialPath={config?.default_save_path || "~"}
          />
        )}

        {addMode === "file" && (
          <FolderPicker
            open
            mode="file"
            title="Open an audio file"
            description="Browse for a single audio file to transcribe and process."
            onClose={() => setAddMode(null)}
            onSelect={(path) => {
              setAddMode(null);
              navigate({ to: "/file/$path", params: { path: encodeURIComponent(path) } });
            }}
            initialPath={config?.default_save_path || "~"}
          />
        )}
      </div>
    </div>
  );
}
