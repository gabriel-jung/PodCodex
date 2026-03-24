import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import {
  getConfig,
  listShows,
  createFromRSS,
  registerShow,
  searchPodcasts,
} from "@/api/client";
import type { PodcastSearchResult, ShowSummary } from "@/api/types";
import { Button } from "@/components/ui/button";
import FolderPicker from "@/components/common/FolderPicker";
import { Plus, FolderOpen, Search, FileAudio } from "lucide-react";

export default function HomePage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: config } = useQuery({ queryKey: ["config"], queryFn: getConfig });
  const { data: shows } = useQuery({
    queryKey: ["shows"],
    queryFn: listShows,
  });

  const [addMode, setAddMode] = useState<"rss" | "import" | "file" | null>(null);

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

function ShowCard({ show, onClick }: { show: ShowSummary; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="text-left p-5 rounded-xl bg-card border border-border hover:border-muted-foreground/30 transition group flex items-center gap-4"
    >
      {show.artwork_url ? (
        <img src={show.artwork_url} alt="" className="w-12 h-12 rounded-lg shrink-0" />
      ) : (
        <div className="w-12 h-12 shrink-0" />
      )}
      <div className="min-w-0">
        <h3 className="font-medium truncate group-hover:text-primary transition">{show.name}</h3>
        <div className="mt-1 flex gap-3 text-xs text-muted-foreground">
          {show.episode_count > 0 && <span>{show.episode_count} episode{show.episode_count !== 1 && "s"}</span>}
          {show.has_rss && <span>RSS</span>}
        </div>
      </div>
    </button>
  );
}

function AddShowModal({ defaultSavePath, onClose, onCreated }: {
  defaultSavePath: string;
  onClose: () => void;
  onCreated: (folder: string) => void;
}) {
  const [step, setStep] = useState<"search" | "location">("search");
  const [searchQuery, setSearchQuery] = useState("");
  const [rssUrl, setRssUrl] = useState("");
  const [artworkUrl, setArtworkUrl] = useState("");
  const [fullPath, setFullPath] = useState("");
  const [pickerOpen, setPickerOpen] = useState(false);

  const searchMutation = useMutation({
    mutationFn: (q: string) => searchPodcasts(q),
  });

  const createMutation = useMutation({
    mutationFn: () => {
      // Split fullPath into parent dir + folder name for the API
      const lastSlash = fullPath.lastIndexOf("/");
      const savePath = lastSlash > 0 ? fullPath.slice(0, lastSlash) : fullPath;
      const folderName = lastSlash > 0 ? fullPath.slice(lastSlash + 1) : "";
      return createFromRSS(rssUrl, savePath, folderName, artworkUrl);
    },
    onSuccess: (data) => onCreated(data.folder),
  });

  const defaultPath = (name: string) =>
    `${defaultSavePath.replace(/\/+$/, "")}/${name}`;

  const selectResult = (result: PodcastSearchResult) => {
    setRssUrl(result.feed_url);
    setArtworkUrl(result.artwork_url);
    setFullPath(defaultPath(result.name));
    setStep("location");
  };

  const handleSearch = () => {
    if (searchQuery.trim()) searchMutation.mutate(searchQuery.trim());
  };

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
        <div className="bg-card border border-border rounded-xl p-6 max-w-lg w-full shadow-2xl max-h-[80vh] flex flex-col">
          <div className="flex items-center justify-between mb-1">
            <h3 className="font-medium">
              {step === "search" ? "Find and add a podcast" : "Choose save location"}
            </h3>
            <Button onClick={onClose} variant="ghost" size="sm">x</Button>
          </div>
          <p className="text-xs text-muted-foreground mb-4">
            {step === "search"
              ? "Search the Apple Podcasts catalog or paste an RSS feed URL to get started."
              : "Choose where to save the podcast episodes and metadata on your machine."}
          </p>

          {step === "search" && (
            <div className="flex flex-col gap-4 overflow-hidden">
              {/* Search */}
              <div className="flex gap-2">
                <input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                  placeholder="Search by podcast name..."
                  className="input flex-1"
                  autoFocus
                />
                <Button
                  onClick={handleSearch}
                  disabled={!searchQuery.trim() || searchMutation.isPending}
                  size="sm"
                >
                  <Search /> {searchMutation.isPending ? "..." : "Search"}
                </Button>
              </div>

              {/* Results */}
              {searchMutation.data && searchMutation.data.length > 0 && (
                <div className="overflow-y-auto flex-1 -mx-2">
                  {searchMutation.data.map((r, i) => (
                    <button
                      key={i}
                      onClick={() => selectResult(r)}
                      className="w-full text-left px-3 py-2.5 rounded-lg hover:bg-accent/50 transition flex items-center gap-3"
                    >
                      {r.artwork_url && (
                        <img src={r.artwork_url} alt="" className="w-10 h-10 rounded shrink-0" />
                      )}
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate">{r.name}</p>
                        <p className="text-xs text-muted-foreground truncate">{r.artist}</p>
                      </div>
                    </button>
                  ))}
                </div>
              )}

              {searchMutation.data && searchMutation.data.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-4">No results found</p>
              )}

              {/* Manual URL */}
              <div className="border-t border-border pt-4">
                <label className="text-xs text-muted-foreground block mb-1">Or paste RSS feed URL directly</label>
                <div className="flex gap-2">
                  <input
                    value={rssUrl}
                    onChange={(e) => setRssUrl(e.target.value)}
                    placeholder="https://feeds.example.com/podcast.xml"
                    className="input flex-1"
                  />
                  <Button
                    onClick={() => {
                      if (!fullPath) setFullPath(defaultPath("podcast"));
                      setStep("location");
                    }}
                    disabled={!rssUrl.trim()}
                    size="sm"
                  >
                    Next
                  </Button>
                </div>
              </div>
            </div>
          )}

          {step === "location" && (
            <div className="space-y-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Save to</label>
                <div className="flex gap-2">
                  <input
                    value={fullPath}
                    onChange={(e) => setFullPath(e.target.value)}
                    className="input flex-1"
                    autoFocus
                  />
                  <Button onClick={() => setPickerOpen(true)} variant="outline" size="sm">Browse</Button>
                </div>
              </div>
              {createMutation.isError && (
                <p className="text-destructive text-xs">{(createMutation.error as Error).message}</p>
              )}
              <div className="flex gap-2">
                <Button onClick={() => setStep("search")} variant="outline" className="flex-1">
                  Back
                </Button>
                <Button
                  onClick={() => createMutation.mutate()}
                  disabled={!fullPath.trim() || createMutation.isPending}
                  className="flex-1"
                >
                  {createMutation.isPending ? "Creating show..." : "Add show"}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>

      <FolderPicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onSelect={(p) => { setFullPath(p); setPickerOpen(false); }}
        initialPath={fullPath || defaultSavePath || "~"}
      />
    </>
  );
}
