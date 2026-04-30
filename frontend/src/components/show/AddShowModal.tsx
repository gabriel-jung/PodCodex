import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { createFromRSS, createFromYouTube, getHealth, registerShow, searchPodcasts } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import type { PodcastSearchResult } from "@/api/types";
import { Button } from "@/components/ui/button";
import { errorMessage, SUB_LANGUAGES } from "@/lib/utils";
import FolderLocationFields from "@/components/common/FolderLocationFields";
import FolderPicker from "@/components/common/FolderPicker";
import MissingDependency from "@/components/common/MissingDependency";
import BundleImportPanel from "./BundleImportPanel";
import { PlaySquare, Search, Rss, FolderOpen, Loader2, Package } from "lucide-react";

export interface AddShowModalProps {
  defaultSavePath: string;
  onClose: () => void;
  onCreated: (folder: string) => void;
  onOpenFile?: (path: string) => void;
}

type SourceMode = "podcast" | "youtube" | "local" | "bundle";

export default function AddShowModal({ defaultSavePath, onClose, onCreated, onOpenFile }: AddShowModalProps) {
  const [step, setStep] = useState<"search" | "location">("search");
  const [sourceMode, setSourceMode] = useState<SourceMode>("podcast");
  const [searchQuery, setSearchQuery] = useState("");
  const [rssUrl, setRssUrl] = useState("");
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [artworkUrl, setArtworkUrl] = useState("");
  const [showName, setShowName] = useState("");
  const [savePath, setSavePath] = useState(defaultSavePath || "~");
  const [folderName, setFolderName] = useState("");
  const [language, setLanguage] = useState("");
  const [customLang, setCustomLang] = useState("");
  const [localPickerOpen, setLocalPickerOpen] = useState<"folder" | "file" | null>(null);

  const { data: health } = useQuery({ queryKey: queryKeys.health(), queryFn: getHealth });
  const hasYtDlp = health?.capabilities?.yt_dlp ?? false;

  const switchMode = (mode: SourceMode) => {
    setSourceMode(mode);
    setRssUrl("");
    setYoutubeUrl("");
    setArtworkUrl("");
    setShowName("");
    setFolderName("");
  };

  const searchMutation = useMutation({
    mutationFn: (q: string) => searchPodcasts(q),
  });

  const effectiveLang = language === "other" ? customLang : language;

  const createMutation = useMutation({
    mutationFn: () => {
      if (sourceMode === "youtube") {
        return createFromYouTube(youtubeUrl, savePath, folderName, artworkUrl, showName, effectiveLang);
      }
      return createFromRSS(rssUrl, savePath, folderName, artworkUrl, showName, effectiveLang);
    },
    onSuccess: (data) => onCreated(data.folder),
  });

  const selectResult = (result: PodcastSearchResult) => {
    setRssUrl(result.feed_url);
    setArtworkUrl(result.artwork_url);
    setShowName(result.name);
    setFolderName(result.name);
    setStep("location");
  };

  const handleSearch = () => {
    if (searchQuery.trim()) searchMutation.mutate(searchQuery.trim());
  };

  const handleYouTubeNext = () => {
    if (!folderName) {
      const slug = youtubeUrl.replace(/https?:\/\//, "").replace(/[^a-zA-Z0-9]+/g, "_").slice(0, 40);
      setFolderName(slug || "youtube");
    }
    setStep("location");
  };

  const importMutation = useMutation({
    mutationFn: (path: string) => registerShow(path),
    onSuccess: (_, path) => onCreated(path),
  });

  const handleLocalImport = (path: string) => importMutation.mutate(path);

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
        <div className="bg-card border border-border rounded-xl p-6 max-w-lg w-full shadow-2xl max-h-[80vh] flex flex-col">
          <div className="flex items-center justify-between mb-1">
            <h3 className="font-medium">
              {step === "search" ? "Add a show" : "Save location"}
            </h3>
            <Button onClick={onClose} variant="ghost" size="sm">x</Button>
          </div>

          {step === "search" && (
            <div className="flex flex-col gap-4 overflow-hidden">
              {/* Source mode toggle: Podcast | YouTube | Local | Bundle */}
              <div className="flex gap-1 bg-muted rounded-lg p-1">
                <button
                  className={`flex-1 text-sm py-1.5 px-3 rounded-md transition flex items-center justify-center gap-1.5 ${sourceMode === "podcast" ? "bg-background shadow-sm font-medium" : "text-muted-foreground hover:text-foreground"}`}
                  onClick={() => switchMode("podcast")}
                >
                  <Rss className="h-3.5 w-3.5" /> Podcast
                </button>
                <button
                  className={`flex-1 text-sm py-1.5 px-3 rounded-md transition flex items-center justify-center gap-1.5 ${sourceMode === "youtube" ? "bg-background shadow-sm font-medium" : "text-muted-foreground hover:text-foreground"}`}
                  onClick={() => switchMode("youtube")}
                >
                  <PlaySquare className="h-3.5 w-3.5" /> YouTube
                </button>
                <button
                  className={`flex-1 text-sm py-1.5 px-3 rounded-md transition flex items-center justify-center gap-1.5 ${sourceMode === "local" ? "bg-background shadow-sm font-medium" : "text-muted-foreground hover:text-foreground"}`}
                  onClick={() => switchMode("local")}
                >
                  <FolderOpen className="h-3.5 w-3.5" /> Local
                </button>
                <button
                  className={`flex-1 text-sm py-1.5 px-3 rounded-md transition flex items-center justify-center gap-1.5 ${sourceMode === "bundle" ? "bg-background shadow-sm font-medium" : "text-muted-foreground hover:text-foreground"}`}
                  onClick={() => switchMode("bundle")}
                >
                  <Package className="h-3.5 w-3.5" /> Bundle
                </button>
              </div>

              {sourceMode === "podcast" && (
                <>
                  <p className="text-xs text-muted-foreground">
                    Search the Apple Podcasts catalog or paste an RSS feed URL to get started.
                  </p>

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
                            <img src={r.artwork_url} alt={r.name} className="w-10 h-10 rounded shrink-0" />
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
                          if (!folderName) setFolderName("podcast");
                          setStep("location");
                        }}
                        disabled={!rssUrl.trim()}
                        size="sm"
                      >
                        Next
                      </Button>
                    </div>
                  </div>
                </>
              )}

              {sourceMode === "local" && (
                <>
                  <p className="text-xs text-muted-foreground">
                    Pick an existing folder of episodes (creates a show) or a single audio file (one-off transcription).
                  </p>

                  <div className="flex flex-col gap-3">
                    <Button
                      onClick={() => setLocalPickerOpen("folder")}
                      variant="outline"
                      className="justify-start gap-2"
                    >
                      <FolderOpen className="w-4 h-4" /> Existing folder…
                    </Button>
                    {onOpenFile && (
                      <Button
                        onClick={() => setLocalPickerOpen("file")}
                        variant="outline"
                        className="justify-start gap-2"
                      >
                        <Search className="w-4 h-4" /> Single audio file…
                      </Button>
                    )}
                  </div>

                  {importMutation.isPending && (
                    <p className="text-xs text-muted-foreground">Registering show…</p>
                  )}
                  {importMutation.isError && (
                    <p className="text-destructive text-xs">{errorMessage(importMutation.error)}</p>
                  )}
                </>
              )}

              {sourceMode === "bundle" && (
                <BundleImportPanel
                  onImported={(showsDir, finalFolder) => {
                    if (finalFolder && showsDir) {
                      onCreated(`${showsDir.replace(/\/+$/, "")}/${finalFolder}`);
                    } else {
                      // Index-only import: nothing to navigate to.
                      onClose();
                    }
                  }}
                />
              )}

              {sourceMode === "youtube" && (
                <>
                  {!hasYtDlp ? (
                    <MissingDependency
                      extra="youtube"
                      label="yt-dlp"
                      description="YouTube support requires the youtube plugin to fetch channels, playlists, and download audio."
                    />
                  ) : (
                    <>
                      <p className="text-xs text-muted-foreground">
                        Paste a YouTube channel, playlist, or video URL. Audio will be extracted with yt-dlp.
                      </p>

                      <div className="flex flex-col gap-3">
                        <input
                          value={youtubeUrl}
                          onChange={(e) => setYoutubeUrl(e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && youtubeUrl.trim() && handleYouTubeNext()}
                          placeholder="https://youtube.com/@channel or playlist URL"
                          className="input w-full"
                          autoFocus
                        />
                        <Button
                          onClick={handleYouTubeNext}
                          disabled={!youtubeUrl.trim()}
                          className="self-end"
                          size="sm"
                        >
                          Next
                        </Button>
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          )}

          {step === "location" && (
            <div className="space-y-4">
              {createMutation.isPending ? (
                <div className="flex flex-col items-center gap-3 py-6">
                  <Loader2 className="w-8 h-8 animate-spin text-primary" />
                  <p className="text-sm text-muted-foreground">
                    {sourceMode === "youtube"
                      ? "Fetching videos from YouTube..."
                      : "Fetching episodes from feed..."}
                  </p>
                  {sourceMode === "youtube" && (
                    <p className="text-xs text-muted-foreground">This can take a moment for large channels.</p>
                  )}
                </div>
              ) : (
                <>
                  <FolderLocationFields
                    folderName={folderName}
                    onFolderNameChange={setFolderName}
                    parentPath={savePath}
                    onParentPathChange={setSavePath}
                    placeholder="My Podcast"
                    autoFocus
                  />

                  <div>
                    <label className="text-xs text-muted-foreground block mb-1.5">Language</label>
                    <div className="flex flex-wrap gap-1.5">
                      {SUB_LANGUAGES.slice(0, 5).map((l) => (
                        <button
                          key={l.code}
                          onClick={() => { setLanguage(l.code); setCustomLang(""); }}
                          className={`px-2.5 py-1 text-xs rounded-md border transition ${language === l.code ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
                        >
                          {l.label}
                        </button>
                      ))}
                      <button
                        onClick={() => setLanguage("other")}
                        className={`px-2.5 py-1 text-xs rounded-md border transition ${language === "other" ? "bg-primary text-primary-foreground border-primary" : "border-border hover:bg-accent"}`}
                      >
                        Other
                      </button>
                    </div>
                    {language === "other" && (
                      <input
                        value={customLang}
                        onChange={(e) => setCustomLang(e.target.value.toLowerCase().slice(0, 5))}
                        placeholder="ISO code (e.g. nl, zh, ar)"
                        className="input w-32 mt-1.5"
                      />
                    )}
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
                      disabled={!folderName.trim() || !savePath.trim()}
                      className="flex-1"
                    >
                      Add show
                    </Button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {localPickerOpen === "folder" && (
        <FolderPicker
          open
          title="Choose a show folder"
          description="Select a folder that already contains podcast episodes or transcripts."
          onClose={() => setLocalPickerOpen(null)}
          onSelect={(p) => { setLocalPickerOpen(null); handleLocalImport(p); }}
          initialPath={defaultSavePath || "~"}
        />
      )}

      {localPickerOpen === "file" && onOpenFile && (
        <FolderPicker
          open
          mode="file"
          title="Choose an audio file"
          description="Select a single audio file to transcribe and process."
          onClose={() => setLocalPickerOpen(null)}
          onSelect={(p) => { setLocalPickerOpen(null); onOpenFile(p); }}
          initialPath={defaultSavePath || "~"}
        />
      )}
    </>
  );
}
