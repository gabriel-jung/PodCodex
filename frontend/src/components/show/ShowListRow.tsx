import { memo } from "react";
import { artworkUrl } from "@/api/filesystem";
import type { ShowSummary } from "@/api/types";
import { SourceIcon } from "./SourceIcon";
import { StaleUpdatedLabel } from "@/components/common/StaleUpdatedLabel";

export interface ShowListRowProps {
  show: ShowSummary;
  onClick: (path: string) => void;
}

function ShowListRowInner({ show, onClick }: ShowListRowProps) {
  return (
    <button
      onClick={() => onClick(show.path)}
      className="w-full text-left px-4 py-3 flex items-center gap-4 hover:bg-accent/50 transition border-b border-border last:border-0"
    >
      {show.artwork_url ? (
        <img src={artworkUrl(show.path)} alt={show.name} className="w-8 h-8 rounded shrink-0" />
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
        <StaleUpdatedLabel timestamp={show.last_rss_update} prefix="updated" />
      </div>
    </button>
  );
}

const ShowListRow = memo(ShowListRowInner);
export default ShowListRow;
