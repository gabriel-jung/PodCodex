import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { createFromRSS, searchPodcasts } from "@/api/client";
import type { PodcastSearchResult } from "@/api/types";
import { Button } from "@/components/ui/button";
import { errorMessage } from "@/lib/utils";
import FolderPicker from "@/components/common/FolderPicker";
import { Search } from "lucide-react";

export interface AddShowModalProps {
  defaultSavePath: string;
  onClose: () => void;
  onCreated: (folder: string) => void;
}

export default function AddShowModal({ defaultSavePath, onClose, onCreated }: AddShowModalProps) {
  const [step, setStep] = useState<"search" | "location">("search");
  const [searchQuery, setSearchQuery] = useState("");
  const [rssUrl, setRssUrl] = useState("");
  const [artworkUrl, setArtworkUrl] = useState("");
  const [showName, setShowName] = useState("");
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
      return createFromRSS(rssUrl, savePath, folderName, artworkUrl, showName);
    },
    onSuccess: (data) => onCreated(data.folder),
  });

  const defaultPath = (name: string) =>
    `${defaultSavePath.replace(/\/+$/, "")}/${name}`;

  const selectResult = (result: PodcastSearchResult) => {
    setRssUrl(result.feed_url);
    setArtworkUrl(result.artwork_url);
    setShowName(result.name);
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
                <p className="text-destructive text-xs">{errorMessage(createMutation.error)}</p>
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
