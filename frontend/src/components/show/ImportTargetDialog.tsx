import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { conflictSuggestion, createLocalShow } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import type { ShowSummary } from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { showEpisodeCountLabel } from "@/lib/showCounts";
import { errorMessage } from "@/lib/utils";
import { FolderPlus, Folder } from "lucide-react";

/** Destination picker for standalone audio imports: choose an existing local
 *  show or create a new one on the spot. Every local (non-feed) show can act
 *  as an import bucket; "Files" is just the default first one. */
export default function ImportTargetDialog({
  fileCount,
  localShows,
  defaultFolder,
  onConfirm,
  onClose,
}: {
  fileCount: number;
  /** Shows the server will accept imports into (`accepts_imports`). */
  localShows: ShowSummary[];
  /** Preselected folder (last used target), if it still exists. */
  defaultFolder: string | null;
  onConfirm: (folder: string) => void;
  onClose: () => void;
}) {
  const preselect =
    defaultFolder && localShows.some((s) => s.path === defaultFolder)
      ? defaultFolder
      : localShows[0]?.path ?? null;
  const [selected, setSelected] = useState<string | null>(preselect);
  const [creating, setCreating] = useState(localShows.length === 0);
  const [newName, setNewName] = useState("");

  const queryClient = useQueryClient();
  const createMutation = useMutation({
    mutationFn: (name: string) => createLocalShow(name),
    onSuccess: (res) => {
      // Only creation changes the shows list; picking an existing show
      // leaves it untouched until the imports land.
      queryClient.invalidateQueries({ queryKey: queryKeys.shows() });
      onConfirm(res.folder);
    },
    onError: (err) => {
      // A folder of that name already sits in the library (registered or
      // not). Offer the free name the server worked out.
      const free = conflictSuggestion(err);
      if (free) setNewName(free);
    },
  });

  const trimmed = newName.trim();
  const canSubmit = creating ? !!trimmed : selected != null;

  const submit = () => {
    if (!canSubmit || createMutation.isPending) return;
    if (creating) createMutation.mutate(trimmed);
    else if (selected) onConfirm(selected);
  };

  return (
    <Dialog open onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Where should this go?</DialogTitle>
          <DialogDescription>
            Pick the show that will hold{" "}
            {fileCount === 1 ? "this audio file" : `these ${fileCount} audio files`}.
          </DialogDescription>
        </DialogHeader>

        {localShows.length > 0 && (
          <div className="max-h-56 overflow-y-auto -mx-1 px-1 space-y-0.5">
            {localShows.map((show) => {
              const active = !creating && selected === show.path;
              const countLabel = showEpisodeCountLabel(show);
              return (
                <button
                  key={show.path}
                  onClick={() => { setCreating(false); setSelected(show.path); }}
                  className={`w-full flex items-center gap-2 rounded px-2 py-2 text-left transition ${
                    active ? "bg-accent text-foreground" : "hover:bg-accent"
                  }`}
                >
                  <Folder className="w-3.5 h-3.5 shrink-0 text-muted-foreground" />
                  <span className="text-xs flex-1 truncate">{show.name}</span>
                  {countLabel && (
                    <span className="text-xs text-muted-foreground/60 shrink-0">
                      {countLabel}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        )}

        {creating ? (
          <input
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
            placeholder="New show name"
            className="input w-full"
            aria-label="New show name"
          />
        ) : (
          <button
            onClick={() => setCreating(true)}
            className="w-full flex items-center gap-2 rounded px-2 py-2 text-left transition hover:bg-accent text-muted-foreground"
          >
            <FolderPlus className="w-3.5 h-3.5 shrink-0" />
            <span className="text-xs">New show…</span>
          </button>
        )}

        {createMutation.isError && (
          <p className="text-destructive text-xs">
            {conflictSuggestion(createMutation.error)
              ? "That name is taken, so here is a free one."
              : errorMessage(createMutation.error)}
          </p>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={submit} disabled={!canSubmit || createMutation.isPending}>
            {createMutation.isPending ? "Creating…" : "Import"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
