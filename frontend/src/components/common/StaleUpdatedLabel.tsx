import { isStale, timeAgo } from "@/lib/utils";

interface Props {
  timestamp: string | null | undefined;
  prefix?: string;
  className?: string;
}

/** "Updated X ago" with amber text when the timestamp is older than a day.
 *  Used in feed-freshness labels across home, show, and detail surfaces. */
export function StaleUpdatedLabel({ timestamp, prefix = "Updated", className = "" }: Props) {
  if (!timestamp) return null;
  const stale = isStale(timestamp);
  return (
    <span
      title={timestamp}
      className={`${stale ? "text-amber-500" : "text-muted-foreground"} ${className}`}
    >
      {prefix} {timeAgo(timestamp)}
    </span>
  );
}
