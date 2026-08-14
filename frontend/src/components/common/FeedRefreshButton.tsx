import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { StaleUpdatedLabel } from "@/components/common/StaleUpdatedLabel";
import { useFeedRefreshing } from "@/hooks/useFeedRefresh";

interface Props {
  onRefresh: () => void;
  title: string;
  /** Feed timestamp shown via StaleUpdatedLabel when idle; falls back to idleLabel. */
  lastUpdate?: string | null;
  idleLabel: string;
  refreshingLabel?: string;
  /** Extra classes on the label span (e.g. "hidden md:inline"). */
  labelClassName?: string;
  /** Extra disable condition (e.g. show meta not loaded yet). */
  disabled?: boolean;
}

/** Shared header button for feed refreshes: spinner while running, stale
 *  "Updated X ago" label while idle. Derives its own in-flight state from
 *  the shared feed-refresh mutation key, so only this button re-renders
 *  when a refresh starts anywhere in the app. Pair with useFeedRefresh/-All. */
export function FeedRefreshButton({
  onRefresh,
  title,
  lastUpdate,
  idleLabel,
  refreshingLabel = "Refreshing...",
  labelClassName,
  disabled = false,
}: Props) {
  const refreshing = useFeedRefreshing();
  return (
    <Button onClick={onRefresh} disabled={refreshing || disabled} variant="outline" size="sm" title={title}>
      <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} />
      <span className={labelClassName}>
        {refreshing ? (
          refreshingLabel
        ) : lastUpdate ? (
          <StaleUpdatedLabel timestamp={lastUpdate} />
        ) : (
          idleLabel
        )}
      </span>
    </Button>
  );
}
