/** Unsaved-edit registry: the one place navigation asks before discarding.
 *
 *  Every in-app move is a router navigation (step switch, prev/next episode,
 *  sidebar, breadcrumb, command palette, Settings tabs), and EpisodePage
 *  remounts its panel per step and episode, so a panel with pending edits is
 *  simply unmounted. A per-component `beforeunload` listener never fires for
 *  any of that, and never fires at all in the Tauri webview.
 *
 *  Surfaces call `useDirtyEdit(dirty, label)`. RootLayout's router blocker
 *  and the browser `beforeunload` hook consult `dirtyEdits.isDirty()`;
 *  modal close paths call `confirmDiscard()` directly. Nothing else should
 *  reimplement the question.
 */

import { useEffect, useId } from "react";
import { confirmDialog } from "@/components/ui/confirm-dialog";

const entries = new Map<string, string>();

export const dirtyEdits = {
  set(id: string, label: string): void {
    entries.set(id, label);
  },
  clear(id: string): void {
    entries.delete(id);
  },
  isDirty(): boolean {
    return entries.size > 0;
  },
  /** Distinct labels, in registration order, for the discard prompt. */
  labels(): string[] {
    return [...new Set(entries.values())];
  },
};

/** Register this component's unsaved edits while `dirty` holds. Cleared on
 *  the falling edge and on unmount, so a saved or discarded edit never
 *  lingers and blocks a later navigation. */
export function useDirtyEdit(dirty: boolean, label: string): void {
  const id = useId();
  useEffect(() => {
    if (!dirty) return;
    dirtyEdits.set(id, label);
    return () => dirtyEdits.clear(id);
  }, [dirty, label, id]);
}

/** Ask whether to throw away the registered edits. Resolves true on
 *  Discard, false on Cancel, Escape or the backdrop. */
export function confirmDiscard(labels: string[] = dirtyEdits.labels()): Promise<boolean> {
  return new Promise((resolve) => {
    const what = labels.length ? labels.join(", ") : "your edits";
    confirmDialog.open({
      title: "Discard unsaved changes?",
      description: `Unsaved: ${what}. Leaving now throws them away.`,
      confirmLabel: "Discard",
      cancelLabel: "Keep editing",
      variant: "destructive",
      onConfirm: () => resolve(true),
      onCancel: () => resolve(false),
    });
  });
}
