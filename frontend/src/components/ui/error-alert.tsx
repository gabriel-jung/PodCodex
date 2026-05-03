/** Consistent error display with Retry / Dismiss / Details actions. */

import { useState } from "react";
import { AlertCircle, RotateCcw, X, ChevronDown, ChevronRight } from "lucide-react";
import { errorMessage } from "@/lib/utils";
import { Button } from "./button";

export interface ErrorAlertProps {
  /** The error to display. Accepts Error, string, or unknown. */
  error: unknown;
  /** Called when the user clicks Retry. */
  onRetry?: () => void;
  /** Called when the user clicks Dismiss. */
  onDismiss?: () => void;
  /** Optional details (stack trace, response body) shown in an expandable panel. */
  details?: string | null;
  /** Compact layout for inline use. */
  compact?: boolean;
  className?: string;
}

export function ErrorAlert({ error, onRetry, onDismiss, details, compact, className }: ErrorAlertProps) {
  const [showDetails, setShowDetails] = useState(false);
  const msg = errorMessage(error);

  if (compact) {
    return (
      <div className={`flex items-center gap-2 text-destructive text-xs ${className ?? ""}`}>
        <AlertCircle className="w-3.5 h-3.5 shrink-0" />
        <span className="flex-1 min-w-0 truncate" title={msg}>{msg}</span>
        {onRetry && (
          <button onClick={onRetry} className="hover:underline shrink-0" title="Retry">Retry</button>
        )}
        {onDismiss && (
          <button onClick={onDismiss} className="text-muted-foreground hover:text-foreground shrink-0" title="Dismiss" aria-label="Dismiss">
            <X className="w-3 h-3" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div className={`rounded-md border border-destructive/30 bg-destructive/5 p-3 ${className ?? ""}`}>
      <div className="flex items-start gap-2.5">
        <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-destructive">{msg}</p>
          {details && (
            <button
              onClick={() => setShowDetails((v) => !v)}
              className="mt-1.5 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
            >
              {showDetails ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              {showDetails ? "Hide details" : "Show details"}
            </button>
          )}
          {showDetails && details && (
            <pre className="mt-2 p-2 bg-muted rounded text-2xs leading-normal text-muted-foreground max-h-60 overflow-auto font-mono whitespace-pre-wrap">
              {details}
            </pre>
          )}
        </div>
        <div className="flex gap-1 shrink-0">
          {onRetry && (
            <Button onClick={onRetry} variant="outline" size="sm" className="h-7 text-xs">
              <RotateCcw className="w-3 h-3 mr-1" /> Retry
            </Button>
          )}
          {onDismiss && (
            <Button onClick={onDismiss} variant="ghost" size="sm" className="h-7 w-7 p-0" aria-label="Dismiss">
              <X className="w-3.5 h-3.5" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
