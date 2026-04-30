import { memo } from "react";
import { artworkUrl } from "@/api/filesystem";
import type { ShowSummary } from "@/api/types";
import { SourceIcon } from "./SourceIcon";
import { StaleUpdatedLabel } from "@/components/common/StaleUpdatedLabel";

export interface ShowCardProps {
  show: ShowSummary;
  onClick: (path: string) => void;
  vertical?: boolean;
}

function ShowCardInner({ show, onClick, vertical }: ShowCardProps) {
  const handleClick = () => onClick(show.path);
  if (vertical) {
    return (
      <button
        onClick={handleClick}
        className="text-left rounded-lg bg-card border border-border hover:border-muted-foreground/30 transition group overflow-hidden"
      >
        <div className="p-3 pb-0">
          {show.artwork_url ? (
            <img src={artworkUrl(show.path)} alt={show.name} className="w-full aspect-square object-cover rounded-lg" />
          ) : (
            <div className="w-full aspect-square bg-muted rounded-lg" />
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
          <StaleUpdatedLabel timestamp={show.last_rss_update} className="text-xs block" />
        </div>
      </button>
    );
  }

  return (
    <button
      onClick={handleClick}
      className="text-left p-3 rounded-lg bg-card border border-border hover:border-muted-foreground/30 transition group flex items-center gap-3"
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
        <StaleUpdatedLabel timestamp={show.last_rss_update} className="text-xs block" />
      </div>
    </button>
  );
}

const ShowCard = memo(ShowCardInner);
export default ShowCard;
