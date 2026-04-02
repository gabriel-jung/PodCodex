import type { ShowSummary } from "@/api/types";
import { timeAgo } from "@/lib/utils";
import { FolderOpen, PlaySquare, Rss } from "lucide-react";

export interface ShowCardProps {
  show: ShowSummary;
  onClick: () => void;
  vertical?: boolean;
}

function SourceIcon({ show }: { show: ShowSummary }) {
  if (show.has_youtube) return <PlaySquare className="w-3.5 h-3.5 text-red-500" title="YouTube" />;
  if (show.has_rss) return <Rss className="w-3.5 h-3.5 text-orange-500" title="RSS" />;
  return <FolderOpen className="w-3.5 h-3.5 text-muted-foreground" title="Local" />;
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
          <img src={show.artwork_url} alt="" className="w-full h-32 object-cover rounded-lg" />
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
        <img src={show.artwork_url} alt="" className="w-14 h-10 object-cover rounded-lg shrink-0" />
      ) : (
        <div className="w-14 h-10 rounded-lg bg-muted shrink-0" />
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
