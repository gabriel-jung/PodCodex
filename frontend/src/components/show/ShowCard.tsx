import { artworkUrl } from "@/api/filesystem";
import type { ShowSummary } from "@/api/types";
import { timeAgo } from "@/lib/utils";
import { SourceIcon } from "./SourceIcon";

export interface ShowCardProps {
  show: ShowSummary;
  onClick: () => void;
  vertical?: boolean;
}

export default function ShowCard({ show, onClick, vertical }: ShowCardProps) {
  if (vertical) {
    return (
      <button
        onClick={onClick}
        className="text-left rounded-xl bg-card border border-border hover:border-muted-foreground/30 transition group overflow-hidden"
      >
        <div className="p-3 pb-0">
        {show.artwork_url ? (
          <img src={artworkUrl(show.path)} alt={show.name} className="w-full h-32 object-cover rounded-lg" />
        ) : (
          <div className="w-full h-32 bg-muted rounded-lg" />
        )}
      </div>
        <div className="p-3 space-y-0.5">
          <h3 className="font-medium text-sm truncate group-hover:text-primary transition flex items-center gap-1.5">
            <SourceIcon show={show} />
            {show.name}
          </h3>
          {show.episode_count != null && show.episode_count > 0 && (
            <p className="text-xs text-muted-foreground">{show.episode_count} episode{show.episode_count !== 1 && "s"}</p>
          )}
          {show.last_rss_update && (
            <p className="text-xs text-muted-foreground" title={show.last_rss_update}>Updated {timeAgo(show.last_rss_update)}</p>
          )}
        </div>
      </button>
    );
  }

  return (
    <button
      onClick={onClick}
      className="text-left p-3 rounded-xl bg-card border border-border hover:border-muted-foreground/30 transition group flex items-center gap-3"
    >
      {show.artwork_url ? (
        <img src={artworkUrl(show.path)} alt={show.name} className="h-12 w-auto max-w-24 rounded-lg shrink-0 object-contain" />
      ) : (
        <div className="w-12 h-12 rounded-lg bg-muted shrink-0" />
      )}
      <div className="min-w-0 space-y-0.5">
        <h3 className="font-medium text-sm truncate group-hover:text-primary transition flex items-center gap-1.5">
          <SourceIcon show={show} />
          {show.name}
        </h3>
        {show.episode_count != null && show.episode_count > 0 && (
          <p className="text-xs text-muted-foreground">{show.episode_count} episode{show.episode_count !== 1 && "s"}</p>
        )}
        {show.last_rss_update && (
          <p className="text-xs text-muted-foreground" title={show.last_rss_update}>Updated {timeAgo(show.last_rss_update)}</p>
        )}
      </div>
    </button>
  );
}
