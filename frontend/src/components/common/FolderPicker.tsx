import { useEffect, useState, useRef } from "react";
import { listDirectory, createDirectory } from "@/api/client";
import type { DirEntry, DirListing, FileEntry } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Folder, FolderPlus, Music, ChevronLeft, ChevronRight, ArrowUp, Home, HardDrive, X } from "lucide-react";

interface FolderPickerProps {
  open: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  initialPath?: string;
  mode?: "folder" | "file";
  title?: string;
  description?: string;
}

const QUICK_ACCESS = [
  { label: "Home", path: "~", icon: Home },
  { label: "Drive C", path: "/mnt/c", icon: HardDrive },
  { label: "Drive D", path: "/mnt/d", icon: HardDrive },
];

export default function FolderPicker({ open, onClose, onSelect, initialPath, mode = "folder", title, description }: FolderPickerProps) {
  const [currentPath, setCurrentPath] = useState(initialPath || "~");
  const [listing, setListing] = useState<DirListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [editingPath, setEditingPath] = useState(false);
  const [pathInput, setPathInput] = useState("");
  const [newFolderName, setNewFolderName] = useState("");
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const pathInputRef = useRef<HTMLInputElement>(null);

  const navigateTo = (path: string, addToHistory = true) => {
    if (addToHistory && listing?.path) {
      const newHistory = history.slice(0, historyIndex + 1);
      newHistory.push(listing.path);
      setHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
    }
    setCurrentPath(path);
  };

  const canGoBack = historyIndex >= 0;
  const canGoForward = historyIndex < history.length - 1;

  const goBack = () => {
    if (!canGoBack) return;
    const prev = history[historyIndex];
    setHistoryIndex(historyIndex - 1);
    setCurrentPath(prev);
  };

  const goForward = () => {
    if (!canGoForward) return;
    const next = history[historyIndex + 2] || history[historyIndex + 1];
    setHistoryIndex(historyIndex + 1);
    // We need to navigate forward — the "current" after going back is one ahead
    if (history[historyIndex + 1]) {
      setCurrentPath(history[historyIndex + 1]);
    }
  };

  const refresh = (path: string) => {
    setLoading(true);
    listDirectory(path, mode === "file")
      .then((data) => {
        setListing(data);
        setPathInput(data.path);
      })
      .catch(() => setListing(null))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!open) return;
    refresh(currentPath);
  }, [currentPath, open, mode]);

  useEffect(() => {
    if (editingPath) pathInputRef.current?.select();
  }, [editingPath]);

  if (!open) return null;

  const handleGoToPath = () => {
    if (pathInput.trim()) {
      navigateTo(pathInput.trim());
      setEditingPath(false);
    }
  };

  const handleSelect = () => {
    if (listing) {
      onSelect(listing.path);
      onClose();
    }
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim() || !listing) return;
    setCreateError(null);
    const res = await createDirectory(listing.path, newFolderName.trim());
    if (res.error) {
      setCreateError(res.error);
    } else {
      setNewFolderName("");
      setCreatingFolder(false);
      refresh(currentPath);
    }
  };

  // Build breadcrumb segments from the resolved path.
  const breadcrumbs = (() => {
    const p = listing?.path || "";
    if (!p) return [];
    const parts = p.split("/").filter(Boolean);
    const crumbs: { label: string; path: string }[] = [{ label: "/", path: "/" }];
    for (let i = 0; i < parts.length; i++) {
      crumbs.push({ label: parts[i], path: "/" + parts.slice(0, i + 1).join("/") });
    }
    return crumbs;
  })();

  const displayTitle = title || (mode === "file" ? "Select audio file" : "Select folder");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-card border border-border rounded-xl w-[700px] max-h-[80vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="px-4 py-3 border-b border-border flex items-center justify-between">
          <h3 className="font-semibold text-sm">{displayTitle}</h3>
          {description && (
            <span className="text-xs text-muted-foreground ml-3 mr-auto">{description}</span>
          )}
          <Button onClick={onClose} variant="ghost" size="sm" className="h-7 w-7 p-0">
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Toolbar: nav buttons + breadcrumb/path */}
        <div className="px-4 py-2 border-b border-border flex items-center gap-1">
          <Button onClick={goBack} variant="ghost" size="sm" disabled={!canGoBack} className="h-7 w-7 p-0">
            <ChevronLeft className="w-4 h-4" />
          </Button>
          <Button onClick={goForward} variant="ghost" size="sm" disabled={!canGoForward} className="h-7 w-7 p-0">
            <ChevronRight className="w-4 h-4" />
          </Button>
          <Button
            onClick={() => listing?.parent && navigateTo(listing.parent)}
            variant="ghost" size="sm"
            disabled={!listing?.parent}
            className="h-7 w-7 p-0"
          >
            <ArrowUp className="w-4 h-4" />
          </Button>

          <div className="flex-1 ml-2">
            {editingPath ? (
              <input
                ref={pathInputRef}
                value={pathInput}
                onChange={(e) => setPathInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleGoToPath();
                  if (e.key === "Escape") setEditingPath(false);
                }}
                onBlur={() => setEditingPath(false)}
                className="input w-full font-mono text-xs h-7"
                autoFocus
              />
            ) : (
              <button
                onClick={() => setEditingPath(true)}
                className="flex items-center gap-0.5 w-full text-left text-sm px-2 py-1
                           rounded hover:bg-accent transition overflow-x-auto"
              >
                {breadcrumbs.map((crumb, i) => (
                  <span key={crumb.path} className="flex items-center gap-0.5 shrink-0">
                    {i > 0 && <span className="text-muted-foreground mx-0.5">/</span>}
                    <span
                      onClick={(e) => { e.stopPropagation(); navigateTo(crumb.path); }}
                      className="text-primary hover:underline cursor-pointer"
                    >
                      {crumb.label === "/" ? "root" : crumb.label}
                    </span>
                  </span>
                ))}
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-1 min-h-0">
          {/* Sidebar */}
          <div className="w-40 border-r border-border py-2 flex flex-col shrink-0">
            <span className="text-2xs text-muted-foreground uppercase tracking-wider px-3 mb-1">Quick access</span>
            {QUICK_ACCESS.map((item) => (
              <button
                key={item.path}
                onClick={() => navigateTo(item.path)}
                className={`flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-accent transition text-left
                  ${listing?.path?.startsWith(item.path === "~" ? "" : item.path) ? "bg-accent text-foreground" : "text-muted-foreground"}`}
              >
                <item.icon className="w-3.5 h-3.5" />
                {item.label}
              </button>
            ))}
          </div>

          {/* Directory + file list */}
          <div className="flex-1 overflow-y-auto">
            {loading && (
              <div className="p-5 text-muted-foreground text-sm">Loading...</div>
            )}

            {listing?.error && (
              <div className="p-5 text-destructive text-sm">{listing.error}</div>
            )}

            {listing && !loading && (
              <div className="py-1">
                {listing.dirs.map((dir) => (
                  <DirRow
                    key={dir.path}
                    dir={dir}
                    onNavigate={() => navigateTo(dir.path)}
                    onSelect={mode === "folder" ? () => { onSelect(dir.path); onClose(); } : undefined}
                  />
                ))}

                {mode === "file" && listing.files.map((file) => (
                  <FileRow
                    key={file.path}
                    file={file}
                    onSelect={() => { onSelect(file.path); onClose(); }}
                  />
                ))}

                {listing.dirs.length === 0 && (!listing.files || listing.files.length === 0) && !listing.error && (
                  <div className="px-5 py-8 text-muted-foreground text-sm text-center">
                    {mode === "file" ? "No folders or audio files" : "Empty folder"}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* New folder inline input */}
        {creatingFolder && mode === "folder" && (
          <div className="px-4 py-2 border-t border-border flex gap-2 items-center">
            <input
              value={newFolderName}
              onChange={(e) => { setNewFolderName(e.target.value); setCreateError(null); }}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCreateFolder();
                if (e.key === "Escape") { setCreatingFolder(false); setNewFolderName(""); setCreateError(null); }
              }}
              className="input flex-1 text-sm"
              placeholder="Folder name"
              autoFocus
            />
            <Button onClick={handleCreateFolder} size="sm" disabled={!newFolderName.trim()}>
              Create
            </Button>
            <Button onClick={() => { setCreatingFolder(false); setNewFolderName(""); setCreateError(null); }} variant="ghost" size="sm">
              Cancel
            </Button>
            {createError && <span className="text-destructive text-xs">{createError}</span>}
          </div>
        )}

        {/* Footer */}
        <div className="px-4 py-3 border-t border-border flex items-center justify-end gap-2">
          {mode === "folder" && !creatingFolder && (
            <Button onClick={() => setCreatingFolder(true)} variant="outline" size="sm" className="gap-1.5 mr-auto">
              <FolderPlus className="w-3.5 h-3.5" />
              New folder
            </Button>
          )}
          <Button onClick={onClose} variant="ghost" size="sm">
            Cancel
          </Button>
          {mode === "folder" && (
            <Button onClick={handleSelect} size="sm">
              Select this folder
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

function DirRow({
  dir,
  onNavigate,
  onSelect,
}: {
  dir: DirEntry;
  onNavigate: () => void;
  onSelect?: () => void;
}) {
  return (
    <div
      className="w-full text-left px-4 py-1.5 text-sm hover:bg-accent
                 transition flex items-center gap-2 group"
    >
      <Folder className={`w-4 h-4 shrink-0 ${dir.is_show ? "text-primary" : dir.has_audio ? "text-yellow-400" : "text-muted-foreground"}`} />

      <button
        onClick={onNavigate}
        className="flex-1 text-left text-secondary-foreground hover:text-foreground truncate"
      >
        {dir.name}
      </button>

      {dir.is_show && (
        <span className="text-2xs bg-primary/20 text-primary px-1.5 py-0.5 rounded-full">
          show
        </span>
      )}
      {dir.has_audio && !dir.is_show && (
        <span className="text-2xs bg-yellow-600/20 text-yellow-400 px-1.5 py-0.5 rounded-full">
          audio
        </span>
      )}

      {onSelect && (
        <button
          onClick={onSelect}
          className="text-xs text-primary hover:text-primary/80 opacity-0
                     group-hover:opacity-100 transition px-2"
        >
          Select
        </button>
      )}
    </div>
  );
}

function FileRow({
  file,
  onSelect,
}: {
  file: FileEntry;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className="w-full text-left px-4 py-1.5 text-sm hover:bg-accent
                 transition flex items-center gap-2"
    >
      <Music className="w-4 h-4 shrink-0 text-success" />
      <span className="flex-1 truncate">{file.name}</span>
    </button>
  );
}
