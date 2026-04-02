/** Hook for managing drag-and-drop state on a container element. */

import { useCallback, useEffect, useRef, useState } from "react";

interface UseDropZoneOptions {
  /** File extensions to accept (e.g. [".json", ".mp3"]). Empty = accept all. */
  accept?: string[];
  /** Called with accepted files on drop. */
  onDrop: (files: File[]) => void;
  /** Disable the drop zone. */
  disabled?: boolean;
}

export function useDropZone({ accept = [], onDrop, disabled }: UseDropZoneOptions) {
  const [isDragging, setIsDragging] = useState(false);
  const dragCounter = useRef(0);

  const matchesExtension = useCallback(
    (file: File) => {
      if (accept.length === 0) return true;
      const name = file.name.toLowerCase();
      return accept.some((ext) => name.endsWith(ext.toLowerCase()));
    },
    [accept],
  );

  useEffect(() => {
    if (disabled) return;

    const onDragEnter = (e: DragEvent) => {
      e.preventDefault();
      dragCounter.current++;
      if (dragCounter.current === 1) setIsDragging(true);
    };

    const onDragOver = (e: DragEvent) => {
      e.preventDefault();
    };

    const onDragLeave = (e: DragEvent) => {
      e.preventDefault();
      dragCounter.current--;
      if (dragCounter.current === 0) setIsDragging(false);
    };

    const onDropHandler = (e: DragEvent) => {
      e.preventDefault();
      dragCounter.current = 0;
      setIsDragging(false);

      const files = Array.from(e.dataTransfer?.files ?? []).filter(matchesExtension);
      if (files.length > 0) onDrop(files);
    };

    document.addEventListener("dragenter", onDragEnter);
    document.addEventListener("dragover", onDragOver);
    document.addEventListener("dragleave", onDragLeave);
    document.addEventListener("drop", onDropHandler);

    return () => {
      document.removeEventListener("dragenter", onDragEnter);
      document.removeEventListener("dragover", onDragOver);
      document.removeEventListener("dragleave", onDragLeave);
      document.removeEventListener("drop", onDropHandler);
    };
  }, [disabled, matchesExtension, onDrop]);

  return { isDragging };
}
