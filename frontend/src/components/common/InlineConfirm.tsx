import { useEffect, useRef } from "react";

interface Props {
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
  destructive?: boolean;
}

/** Row-level confirmation strip — replaces the row content until resolved.
 *  Esc cancels, Enter confirms. Keeps destructive actions explicit without
 *  pulling in a modal. */
export default function InlineConfirm({
  message,
  confirmLabel = "Delete",
  cancelLabel = "Keep",
  onConfirm,
  onCancel,
  destructive = true,
}: Props) {
  const confirmRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    confirmRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
      else if (e.key === "Enter") onConfirm();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel, onConfirm]);

  return (
    <div className="flex items-center gap-2 w-full text-xs">
      <span className="flex-1 text-muted-foreground truncate">{message}</span>
      <button
        onClick={onCancel}
        className="px-2 py-0.5 rounded text-muted-foreground hover:text-foreground transition"
      >
        {cancelLabel}
      </button>
      <button
        ref={confirmRef}
        onClick={onConfirm}
        className={`px-2 py-0.5 rounded transition ${
          destructive
            ? "text-destructive hover:bg-destructive/10"
            : "text-foreground hover:bg-accent"
        }`}
      >
        {confirmLabel}
      </button>
    </div>
  );
}
