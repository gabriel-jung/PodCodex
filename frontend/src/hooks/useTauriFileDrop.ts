/** Tauri-native file drop hook — gives real filesystem paths, unlike browser drag-drop. */

import { useEffect, useState } from "react";
import { usePlatform } from "@/platform";

export interface UseTauriFileDropOptions {
  /** Extensions to accept (e.g. [".mp3", ".wav"]). Empty array accepts all. */
  accept?: string[];
  /** Called with accepted paths on drop. */
  onDrop: (paths: string[]) => void;
  disabled?: boolean;
}

export function useTauriFileDrop({ accept = [], onDrop, disabled }: UseTauriFileDropOptions) {
  const platform = usePlatform();
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    if (disabled || !platform.isTauri) return;

    let unlisten: (() => void) | null = null;
    let cancelled = false;

    (async () => {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const fn = await getCurrentWindow().onDragDropEvent((event) => {
        const payload = event.payload;
        if (payload.type === "over") {
          setIsHovering(true);
        } else if (payload.type === "leave") {
          setIsHovering(false);
        } else if (payload.type === "drop") {
          setIsHovering(false);
          const paths = payload.paths.filter((p) => {
            if (accept.length === 0) return true;
            const low = p.toLowerCase();
            return accept.some((ext) => low.endsWith(ext.toLowerCase()));
          });
          if (paths.length > 0) onDrop(paths);
        }
      });
      if (cancelled) fn();
      else unlisten = fn;
    })();

    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, [disabled, platform.isTauri, accept, onDrop]);

  return { isHovering };
}
