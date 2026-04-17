import { Button } from "@/components/ui/button";
import { errorMessage } from "@/lib/utils";

interface PipelineRunFooterProps {
  onRun: () => void;
  isPending: boolean;
  /** Mutation error, or null. */
  mutationError: unknown;

  /** True when the step has already been run at least once — flips the label
   *  and surfaces the "saves a new version" caveat. */
  hasExisting: boolean;
  /** Label for the first run (no prior version). */
  initialLabel: string;
  /** Label for subsequent runs. */
  rerunLabel: string;

  /** Additional disable reason OR'd with pending. Use for backend-missing or
   *  step-specific guards like "target language empty". */
  disabled?: boolean;
  disabledTitle?: string;
}

/**
 * Shared run-button row for LLM pipeline panels. Renders:
 *   [Button] [caveat if hasExisting]
 *   [error message on next line if the mutation failed]
 */
export default function PipelineRunFooter({
  onRun,
  isPending,
  mutationError,
  hasExisting,
  initialLabel,
  rerunLabel,
  disabled,
  disabledTitle,
}: PipelineRunFooterProps) {
  return (
    <div className="flex items-baseline gap-3 flex-wrap pt-1">
      <Button
        onClick={onRun}
        disabled={isPending || disabled}
        size="sm"
        title={disabledTitle}
      >
        {isPending ? "Starting…" : hasExisting ? rerunLabel : initialLabel}
      </Button>
      {hasExisting && (
        <span className="text-xs text-muted-foreground">Saves a new version — previous ones stay in History.</span>
      )}
      {mutationError != null && (
        <p className="text-destructive text-xs w-full">{errorMessage(mutationError)}</p>
      )}
    </div>
  );
}
