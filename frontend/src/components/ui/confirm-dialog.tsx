import { useCallback, useSyncExternalStore, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Button } from "./button";

interface ConfirmOptions {
  title: string;
  description?: string;
  content?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "default" | "destructive";
  onConfirm: () => void | Promise<void>;
}

// ── Global state (like Whispering's confirmationDialog) ──

let current: ConfirmOptions | null = null;
const listeners = new Set<() => void>();
function notify() { listeners.forEach((cb) => cb()); }

export const confirmDialog = {
  open(options: ConfirmOptions) {
    current = options;
    notify();
  },
  close() {
    current = null;
    notify();
  },
};

function useConfirmState() {
  return useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => listeners.delete(cb); },
    () => current,
  );
}

// ── Rendered once in RootLayout ──

export function ConfirmDialogHost() {
  const state = useConfirmState();

  const handleConfirm = useCallback(async () => {
    if (!state) return;
    await state.onConfirm();
    confirmDialog.close();
  }, [state]);

  if (!state) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 animate-in fade-in duration-150"
        onClick={() => confirmDialog.close()}
      />
      {/* Dialog */}
      <div className="relative bg-card border border-border rounded-lg shadow-lg p-6 max-w-md w-full mx-4 animate-in zoom-in-95 fade-in duration-150">
        <h3 className="text-lg font-semibold">{state.title}</h3>
        {state.description && (
          <p className="text-sm text-muted-foreground mt-2">{state.description}</p>
        )}
        {state.content && <div className="mt-3">{state.content}</div>}
        <div className="flex justify-end gap-3 mt-6">
          <Button variant="ghost" size="sm" onClick={() => confirmDialog.close()}>
            {state.cancelLabel || "Cancel"}
          </Button>
          <Button
            variant={state.variant || "default"}
            size="sm"
            onClick={handleConfirm}
          >
            {state.confirmLabel || "Confirm"}
          </Button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
