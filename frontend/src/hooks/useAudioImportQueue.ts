/** Sequential standalone-audio import queue, shared by HomePage and ShowPage. */

import { useCallback, useState } from "react";
import { conflictSuggestion, importLocalFile } from "@/api/client";
import type { FilesImportResponse } from "@/api/types";
import { errorMessage, splitPath } from "@/lib/utils";

export interface ImportConflict {
  filePath: string;
  suggested: string;
  folder: string;
  remaining: string[];
  imported: FilesImportResponse[];
  errors: string[];
}

/** Imports audio files into a show one at a time. A 409 (name taken) pauses
 *  the queue into `conflict`; the page renders `ImportFileDialog` and calls
 *  `resumeAfterConflict` (file imported under a new name) or `skipConflict`
 *  (file skipped) to continue. Other failures land in `errors` (rendered by
 *  `ImportErrorsBanner`, cleared via `dismissErrors`) and the queue moves
 *  on. `onFinished` fires once per batch with everything that imported and
 *  every error message. */
export function useAudioImportQueue(
  onFinished: (imported: FilesImportResponse[], errors: string[]) => void,
) {
  const [conflict, setConflict] = useState<ImportConflict | null>(null);
  const [errors, setErrors] = useState<string[]>([]);

  const runFrom = useCallback(
    async (
      paths: string[],
      folder: string,
      imported: FilesImportResponse[],
      collected: string[],
    ) => {
      for (let i = 0; i < paths.length; i++) {
        try {
          const res = await importLocalFile(paths[i], undefined, folder);
          imported = [...imported, res];
        } catch (err) {
          const suggested = conflictSuggestion(err);
          if (suggested) {
            setConflict({
              filePath: paths[i],
              suggested,
              folder,
              remaining: paths.slice(i + 1),
              imported,
              errors: collected,
            });
            return;
          }
          const name = splitPath(paths[i]).basename || paths[i];
          collected = [...collected, `${name}: ${errorMessage(err)}`];
        }
      }
      setErrors(collected);
      onFinished(imported, collected);
    },
    [onFinished],
  );

  const run = useCallback(
    (paths: string[], folder: string) => runFrom(paths, folder, [], []),
    [runFrom],
  );

  const resumeAfterConflict = useCallback(
    (folder: string, stem: string) => {
      if (!conflict) return;
      const { remaining, imported, errors: collected } = conflict;
      setConflict(null);
      void runFrom(remaining, conflict.folder, [...imported, { folder, stem }], collected);
    },
    [conflict, runFrom],
  );

  const skipConflict = useCallback(() => {
    if (!conflict) return;
    const { remaining, imported, errors: collected } = conflict;
    setConflict(null);
    void runFrom(remaining, conflict.folder, imported, collected);
  }, [conflict, runFrom]);

  const dismissErrors = useCallback(() => setErrors([]), []);

  return { run, conflict, resumeAfterConflict, skipConflict, errors, dismissErrors };
}
