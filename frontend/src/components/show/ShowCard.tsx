import { memo } from "react";
import { artworkUrl } from "@/api/filesystem";
import type { ShowSummary } from "@/api/types";
import { useLayoutStore } from "@/stores";
import { SourceIcon } from "./SourceIcon";
import { showEpisodeCountLabel } from "@/lib/showCounts";
import ShowProgressStrip from "./ShowProgressStrip";

export interface ShowCardProps {
  show: ShowSummary;
  onClick: (path: string) => void;
  vertical?: boolean;
}

function ShowCardInner({ show, onClick, vertical }: ShowCardProps) {
  const compact = useLayoutStore((s) => s.compact);
  const handleClick = () => onClick(show.path);
  const countLabel = showEpisodeCountLabel(show);

  if (vertical) {
    return (
      <button
        data-uniform-card
        onClick={handleClick}
        className="text-left rounded-lg bg-card border border-border hover:border-muted-foreground/30 transition group overflow-hidden flex flex-col"
      >
        <div className="p-3 pb-0 shrink-0">
          {show.artwork_url ? (
            <img src={artworkUrl(show.path)} alt={show.name} className="w-full aspect-square object-cover rounded-lg" />
          ) : (
            <div className="w-full aspect-square bg-muted rounded-lg" />
          )}
        </div>
        <div className="p-3 space-y-1 flex-1">
          <h3 className="font-medium text-sm truncate group-hover:text-primary transition flex items-center gap-1.5">
            <SourceIcon show={show} />
            {show.name}
          </h3>
          {countLabel && (
            <p className="text-xs text-muted-foreground truncate">{countLabel}</p>
          )}
          {!compact && <ShowProgressStrip show={show} />}
        </div>
      </button>
    );
  }

  return (
    <button
      onClick={handleClick}
      className="text-left rounded-lg bg-card border border-border hover:border-muted-foreground/30 transition group flex items-center gap-3 p-3"
    >
      <div className="h-20 w-20 shrink-0 rounded-lg overflow-hidden bg-muted">
        {show.artwork_url ? (
          <img
            src={artworkUrl(show.path)}
            alt={show.name}
            className="w-full h-full object-cover"
          />
        ) : null}
      </div>
      <div className="min-w-0 flex-1 space-y-1">
        <h3 className="font-medium text-sm truncate group-hover:text-primary transition flex items-center gap-1.5">
          <SourceIcon show={show} />
          {show.name}
        </h3>
        {countLabel && (
          <p className="text-xs text-muted-foreground">{countLabel}</p>
        )}
        {!compact && <ShowProgressStrip show={show} />}
      </div>
    </button>
  );
}

const ShowCard = memo(ShowCardInner);
export default ShowCard;
