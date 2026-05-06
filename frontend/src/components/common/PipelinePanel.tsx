import { ChevronDown, ChevronRight } from "lucide-react";
import ProgressBar from "@/components/editor/ProgressBar";
import { EmptyState } from "@/components/ui/empty-state";
import type { PanelStatus } from "@/lib/stepStatus";

const STATUS_META: Record<PanelStatus, { text: string; color: string }> = {
  ready: { text: "ready", color: "text-success" },
  review: { text: "needs review", color: "text-info" },
  none: { text: "not started", color: "text-muted-foreground/60" },
};

interface PipelinePanelProps {
  /** Panel title shown in the header. */
  title: string;
  /** One-line description shown below the title. */
  description: string;
  /** Stage status — "none" hides the collapse chevron and shows settings inline.
   *  "ready"/"review" enables collapse and renders a status badge in the header. */
  status: PanelStatus;
  /** Controls are expanded (editable). */
  expanded: boolean;
  /** Toggle expanded state. */
  onToggle: () => void;
  /** Label for the chevron toggle when done (e.g. "Re-run correction"). */
  rerunLabel: string;
  /** Label shown above controls when step hasn't been run yet. */
  settingsLabel?: string;
  /** Active task ID — shows progress bar, hides controls. */
  taskId: string | null;
  /** Called when task completes. */
  onTaskComplete?: () => void;
  /** Called when user clicks Retry on a stuck/failed task. */
  onRetry?: () => void;
  /** Called when user dismisses a stuck/failed task. */
  onDismiss?: () => void;
  /** Controls section — rendered inside the collapsible area. */
  controls?: React.ReactNode;
  /** Main content — shown below controls (e.g. TranscriptViewer, results). */
  children?: React.ReactNode;
  /** Empty state message when step not done and controls collapsed. */
  emptyMessage?: string;
  /** Prerequisite message — when set, renders only the header + this message. */
  prerequisite?: string;
  /** Rich blocker content (e.g. install button) — takes precedence over prerequisite. */
  blocker?: React.ReactNode;
}

export default function PipelinePanel({
  title,
  description,
  status,
  expanded,
  onToggle,
  rerunLabel,
  settingsLabel,
  taskId,
  onTaskComplete,
  controls,
  children,
  emptyMessage,
  prerequisite,
  blocker,
  onRetry,
  onDismiss,
}: PipelinePanelProps) {
  const hasOutput = status !== "none";
  const header = (
    <div className="sticky top-0 z-10 bg-background px-4 py-2 border-b border-border">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-semibold">{title}</span>
        <span className={`text-2xs shrink-0 ${STATUS_META[status].color}`}>{STATUS_META[status].text}</span>
      </div>
      <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
    </div>
  );

  if (blocker || prerequisite) {
    return (
      <div className="flex flex-col h-full">
        {header}
        <div className="p-6">{blocker || <span className="text-muted-foreground">{prerequisite}</span>}</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {header}

      {!taskId && controls && (
        <div className="border-b border-border bg-secondary/30">
          {hasOutput ? (
            <button
              onClick={onToggle}
              className="w-full px-4 py-1.5 flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition"
            >
              {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
              <span className="font-medium">{rerunLabel}</span>
            </button>
          ) : settingsLabel ? (
            <div className="px-4 pt-2 pb-1">
              <span className="text-xs font-medium text-muted-foreground">{settingsLabel}</span>
            </div>
          ) : null}

          {expanded && controls}
        </div>
      )}

      {taskId && <ProgressBar taskId={taskId} onComplete={onTaskComplete} onRetry={onRetry} onDismiss={onDismiss} onCancel={onDismiss} />}

      {children}

      {!hasOutput && !expanded && !taskId && emptyMessage && (
        <EmptyState title={emptyMessage} dashed />
      )}
    </div>
  );
}
