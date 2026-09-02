import { useEffect } from "react";
import { ErrorAlert } from "@/components/ui/error-alert";
import { useMutationErrorStore } from "@/stores";

/** Long enough to read, short enough that a stale failure never lingers
 *  under the next thing the user does. Dismiss closes it early. */
const AUTO_DISMISS_MS = 12_000;

/** Shell-level surface for failed writes (see MutationCache.onError in
 *  main.tsx). Sits above the task bar so it is visible on every route and
 *  never covers the content the failure is about. */
export default function MutationErrorBanner() {
  const failure = useMutationErrorStore((s) => s.failure);
  const dismiss = useMutationErrorStore((s) => s.dismiss);

  useEffect(() => {
    if (!failure) return;
    const id = window.setTimeout(dismiss, AUTO_DISMISS_MS);
    return () => window.clearTimeout(id);
  }, [failure, dismiss]);

  if (!failure) return null;
  return (
    <div role="alert" className="border-t border-border bg-card px-4 py-1.5">
      <ErrorAlert error={failure.message} onDismiss={dismiss} compact />
    </div>
  );
}
