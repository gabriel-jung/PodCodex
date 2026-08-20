import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { conflictSuggestion, importLocalFile } from "@/api/client";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { errorMessage, splitPath } from "@/lib/utils";
import type { ImportConflict } from "@/hooks/useAudioImportQueue";

/** Rename prompt shown when importing a standalone audio file collides with
 *  an existing episode in the target show. Retries the import with the
 *  chosen name; a repeat collision refreshes the suggestion in place. */
export default function ImportFileDialog({
  conflict,
  onImported,
  onClose,
}: {
  conflict: ImportConflict;
  onImported: (folder: string, stem: string) => void;
  onClose: () => void;
}) {
  const { filePath, suggested, folder } = conflict;
  const [name, setName] = useState(suggested);
  const fileName = splitPath(filePath).basename || filePath;

  const retryMutation = useMutation({
    mutationFn: (newName: string) => importLocalFile(filePath, newName, folder),
    onSuccess: (res) => onImported(res.folder, res.stem),
    onError: (err) => {
      const next = conflictSuggestion(err);
      if (next) setName(next);
    },
  });

  const stillConflicts = retryMutation.isError && !!conflictSuggestion(retryMutation.error);
  const trimmed = name.trim();

  const submit = () => {
    if (trimmed && !retryMutation.isPending) retryMutation.mutate(trimmed);
  };

  return (
    <Dialog open onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>That name is taken</DialogTitle>
          <DialogDescription>
            That show already has an episode named after{" "}
            <span className="font-mono">{fileName}</span>. Give this one a
            different name.
          </DialogDescription>
        </DialogHeader>
        <input
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") submit(); }}
          className="input w-full"
          aria-label="New episode name"
        />
        {retryMutation.isError && (
          <p className="text-destructive text-xs">
            {stillConflicts
              ? "That one is taken too, so here is a new suggestion."
              : errorMessage(retryMutation.error)}
          </p>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={submit} disabled={!trimmed || retryMutation.isPending}>
            {retryMutation.isPending ? "Importing…" : "Import"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
