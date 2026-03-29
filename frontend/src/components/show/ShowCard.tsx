import type { ShowSummary } from "@/api/types";
import { timeAgo } from "@/lib/utils";

export interface ShowCardProps {
  show: ShowSummary;
  onClick: () => void;
}

export default function ShowCard({ show, onClick }: ShowCardProps) {
  return (
    <button
      onClick={onClick}
      className="text-left p-5 rounded-xl bg-card border border-border hover:border-muted-foreground/30 transition group flex items-center gap-4"
    >
      {show.artwork_url ? (
        <img src={show.artwork_url} alt="" className="w-12 h-12 rounded-lg shrink-0" />
      ) : (
        <div className="w-12 h-12 shrink-0" />
      )}
      <div className="min-w-0">
        <h3 className="font-medium truncate group-hover:text-primary transition">{show.name}</h3>
        <div className="mt-1 flex gap-3 text-xs text-muted-foreground">
          {show.episode_count > 0 && <span>{show.episode_count} episode{show.episode_count !== 1 && "s"}</span>}
          {show.last_rss_update && <span title={show.last_rss_update}>updated {timeAgo(show.last_rss_update)}</span>}
        </div>
      </div>
    </button>
  );
}
