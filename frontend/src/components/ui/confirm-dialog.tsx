import { useState, useSyncExternalStore, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "./button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./dialog";
import { ErrorAlert } from "./error-alert";

interface ConfirmOptions {
  title: string;
  description?: string;
  content?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "default" | "destructive";
  onConfirm: () => void | Promise<void>;
  /** Cancel button, Escape or the backdrop. Not called after a confirm. */
  onCancel?: () => void;
}

// ── Global state (like Whispering's confirmationDialog) ──

/** Each open gets a fresh id so the host's per-request state (pending,
 *  error) resets when one confirmation replaces another. */
interface ConfirmRequest extends ConfirmOptions {
  id: number;
}

let current: ConfirmRequest | null = null;
let nextId = 0;
const listeners = new Set<() => void>();
function notify() { listeners.forEach((cb) => cb()); }

export const confirmDialog = {
  open(options: ConfirmOptions) {
    current = { ...options, id: ++nextId };
    notify();
  },
  close() {
    current = null;
    notify();
  },
  /** Dismiss without confirming, and tell the requester so. */
  cancel() {
    const req = current;
    current = null;
    notify();
    req?.onCancel?.();
  },
};

function useConfirmState() {
  return useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => listeners.delete(cb); },
    () => current,
  );
}

// ── Rendered once in RootLayout ──

/** Built on the shadcn Dialog so focus moves into the dialog and is trapped
 *  there, Escape and the backdrop dismiss, and the title and description
 *  are announced. Every destructive confirmation in the app goes through
 *  here, so this is where keyboard and screen-reader users meet it. */
export function ConfirmDialogHost() {
  const state = useConfirmState();
  if (!state) return null;
  return <ConfirmDialogBody key={state.id} request={state} />;
}

function ConfirmDialogBody({ request }: { request: ConfirmRequest }) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const handleConfirm = async () => {
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      await request.onConfirm();
      confirmDialog.close();
    } catch (e) {
      // Stay open with the failure in view; Confirm becomes a retry.
      setError(e);
    } finally {
      setPending(false);
    }
  };

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        // Escape and the backdrop: keep a running onConfirm undisturbed.
        if (!open && !pending) confirmDialog.cancel();
      }}
    >
      <DialogContent
        showCloseButton={false}
        className="bg-popover border-border max-w-md"
        // Radix warns when no description element exists; an explicit
        // undefined tells it none is coming.
        {...(request.description ? {} : { "aria-describedby": undefined })}
      >
        <DialogHeader>
          <DialogTitle>{request.title}</DialogTitle>
          {request.description && (
            <DialogDescription>{request.description}</DialogDescription>
          )}
        </DialogHeader>
        {request.content}
        {error != null && <ErrorAlert error={error} compact />}
        <DialogFooter className="mt-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => confirmDialog.cancel()}
            disabled={pending}
          >
            {request.cancelLabel || "Cancel"}
          </Button>
          <Button
            variant={request.variant || "default"}
            size="sm"
            onClick={handleConfirm}
            disabled={pending}
          >
            {pending && <Loader2 className="w-3.5 h-3.5 mr-1 animate-spin" />}
            {error != null ? "Retry" : request.confirmLabel || "Confirm"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
