import { ChevronDown, ChevronRight } from "lucide-react";
import ProgressBar from "@/components/editor/ProgressBar";
import { EmptyState } from "@/components/ui/empty-state";

interface PipelinePanelProps {
  /** Panel title shown in the header. */
  title: string;
  /** One-line description shown below the title. */
  description: string;
  /** Whether this step has already been completed. Controls chevron vs open. */
  done: boolean;
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
  done,
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
  if (blocker || prerequisite) {
    return (
      <div className="flex flex-col h-full">
        <div className="sticky top-0 z-10 bg-background px-4 py-2 border-b border-border">
          <span className="text-sm font-semibold">{title}</span>
          <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
        </div>
        <div className="p-6">{blocker || <span className="text-muted-foreground">{prerequisite}</span>}</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Step header — sticks to the top of the scroll container so the
          panel title stays visible while the transcript/controls scroll. */}
      <div className="sticky top-0 z-10 bg-background px-4 py-2 border-b border-border">
        <span className="text-sm font-semibold">{title}</span>
        <p className="text-xs text-muted-foreground mt-0.5">{description}</p>
      </div>

      {/* Controls — collapsible when step is done.
          Tinted background reads as a distinct "settings strip" so the eye
          can separate it from the editor region below. */}
      {!taskId && controls && (
        <div className="border-b border-border bg-secondary/30">
          {done ? (
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

      {/* Progress */}
      {taskId && <ProgressBar taskId={taskId} onComplete={onTaskComplete} onRetry={onRetry} onDismiss={onDismiss} onCancel={onDismiss} />}

      {/* Main content (editor, results, etc.) */}
      {children}

      {/* Empty state */}
      {!done && !expanded && !taskId && emptyMessage && (
        <EmptyState title={emptyMessage} dashed />
      )}
    </div>
  );
}
