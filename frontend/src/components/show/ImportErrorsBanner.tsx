import { X } from "lucide-react";

/** Dismissible list of standalone-import failures, one line per file.
 *  Renders nothing when there are no errors. */
export default function ImportErrorsBanner({
  errors,
  onDismiss,
  className = "",
}: {
  errors: string[];
  onDismiss: () => void;
  className?: string;
}) {
  if (errors.length === 0) return null;
  return (
    <div
      className={`flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive ${className}`}
    >
      <div className="flex-1 space-y-0.5">
        {errors.map((e) => (
          <p key={e}>Couldn't import {e}</p>
        ))}
      </div>
      <button
        onClick={onDismiss}
        className="shrink-0 hover:text-foreground transition"
        aria-label="Dismiss import errors"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
