import type { ShowSummary } from "@/api/types";
import { timeAgo } from "@/lib/utils";
import { FolderOpen, PlaySquare, Rss } from "lucide-react";

export interface ShowListRowProps {
  show: ShowSummary;
  onClick: () => void;
}

function SourceIcon({ show }: { show: ShowSummary }) {
  if (show.has_youtube) return <PlaySquare className="w-3.5 h-3.5 text-red-500" title="YouTube" />;
  if (show.has_rss) return <Rss className="w-3.5 h-3.5 text-orange-500" title="RSS" />;
  return <FolderOpen className="w-3.5 h-3.5 text-muted-foreground" title="Local" />;
}

export default function ShowListRow({ show, onClick }: ShowListRowProps) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left px-4 py-3 flex items-center gap-4 hover:bg-accent/50 transition border-b border-border last:border-0"
    >
      {show.artwork_url ? (
        <img src={show.artwork_url} alt="" className="w-8 h-8 rounded shrink-0" />
      ) : (
        <div className="w-8 h-8 rounded bg-muted shrink-0" />
      )}
      <div className="min-w-0 flex-1">
        <span className="font-medium text-sm truncate block">{show.name}</span>
      </div>
      <div className="flex items-center gap-3 text-xs text-muted-foreground shrink-0">
        <SourceIcon show={show} />
        {show.episode_count != null && show.episode_count > 0 && (
          <span>{show.episode_count} ep{show.episode_count !== 1 && "s"}</span>
        )}
        {show.last_rss_update && (
          <span title={show.last_rss_update}>updated {timeAgo(show.last_rss_update)}</span>
        )}
      </div>
    </button>
  );
}
