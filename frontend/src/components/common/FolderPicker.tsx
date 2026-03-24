import { useEffect, useState } from "react";
import { listDirectory, type DirEntry, type DirListing, type FileEntry } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Folder, Music } from "lucide-react";

interface FolderPickerProps {
  open: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  initialPath?: string;
  mode?: "folder" | "file";
  title?: string;
  description?: string;
}

export default function FolderPicker({ open, onClose, onSelect, initialPath, mode = "folder", title, description }: FolderPickerProps) {
  const [currentPath, setCurrentPath] = useState(initialPath || "~");
  const [listing, setListing] = useState<DirListing | null>(null);
  const [loading, setLoading] = useState(false);
  const [pathInput, setPathInput] = useState("");

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    listDirectory(currentPath, mode === "file")
      .then((data) => {
        setListing(data);
        setPathInput(data.path);
      })
      .catch(() => setListing(null))
      .finally(() => setLoading(false));
  }, [currentPath, open, mode]);

  if (!open) return null;

  const handleGoToPath = () => {
    if (pathInput.trim()) setCurrentPath(pathInput.trim());
  };

  const handleSelect = () => {
    if (listing) {
      onSelect(listing.path);
      onClose();
    }
  };

  const displayTitle = title || (mode === "file" ? "Select audio file" : "Select folder");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-card border border-border rounded-xl w-[600px] max-h-[80vh] flex flex-col shadow-2xl">
        {/* Header */}
        <div className="px-5 py-4 border-b border-border">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold">{displayTitle}</h3>
            <Button onClick={onClose} variant="ghost" size="sm">x</Button>
          </div>
          {description && (
            <p className="text-xs text-muted-foreground mt-1">{description}</p>
          )}
        </div>

        {/* Path bar */}
        <div className="px-5 py-3 border-b border-border flex gap-2">
          <input
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleGoToPath()}
            className="input flex-1 font-mono"
          />
          <Button onClick={handleGoToPath} variant="outline" size="sm">
            Go
          </Button>
        </div>

        {/* Directory + file list */}
        <div className="flex-1 overflow-y-auto min-h-[300px]">
          {loading && (
            <div className="p-5 text-muted-foreground text-sm">Loading...</div>
          )}

          {listing?.error && (
            <div className="p-5 text-destructive text-sm">{listing.error}</div>
          )}

          {listing && !loading && (
            <div className="py-1">
              {listing.parent && (
                <button
                  onClick={() => setCurrentPath(listing.parent!)}
                  className="w-full text-left px-5 py-2 text-sm text-muted-foreground
                             hover:bg-accent transition flex items-center gap-2"
                >
                  <span className="text-muted-foreground">↑</span>
                  <span>..</span>
                </button>
              )}

              {listing.dirs.map((dir) => (
                <DirRow
                  key={dir.path}
                  dir={dir}
                  onNavigate={() => setCurrentPath(dir.path)}
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
                  {mode === "file" ? "No folders or audio files" : "No subdirectories"}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-border flex items-center justify-between">
          <span className="text-xs text-muted-foreground truncate max-w-[300px]">
            {listing?.path}
          </span>
          <div className="flex gap-2">
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
      className="w-full text-left px-5 py-2 text-sm hover:bg-accent
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
        <span className="text-xs bg-primary/20 text-primary px-2 py-0.5 rounded-full">
          show
        </span>
      )}
      {dir.has_audio && !dir.is_show && (
        <span className="text-xs bg-yellow-600/20 text-yellow-400 px-2 py-0.5 rounded-full">
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
      className="w-full text-left px-5 py-2 text-sm hover:bg-accent
                 transition flex items-center gap-2"
    >
      <Music className="w-4 h-4 shrink-0 text-green-400" />
      <span className="flex-1 truncate">{file.name}</span>
    </button>
  );
}
