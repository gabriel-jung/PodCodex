import { memo } from "react";
import { showArtworkSrc, useArtworkEpoch } from "@/lib/showArtwork";
import type { ShowSummary } from "@/api/types";
import { useLayoutStore } from "@/stores";
import { SourceIcon } from "./SourceIcon";
import { showEpisodeCountLabel } from "@/lib/showCounts";
import ShowProgressStrip from "./ShowProgressStrip";

export interface ShowListRowProps {
  show: ShowSummary;
  onClick: (path: string) => void;
}

function ShowListRowInner({ show, onClick }: ShowListRowProps) {
  const compact = useLayoutStore((s) => s.compact);
  const artworkEpoch = useArtworkEpoch();
  const countLabel = showEpisodeCountLabel(show);
  return (
    <button
      onClick={() => onClick(show.path)}
      className="w-full text-left flex items-start gap-4 hover:bg-accent/50 transition border-b border-border last:border-0 px-4 py-3"
    >
      <img src={showArtworkSrc(show.artwork_url, show.path, artworkEpoch)} alt={show.name} className="w-8 h-8 rounded shrink-0" />
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="flex items-center gap-1.5">
          <SourceIcon show={show} />
          <span className="font-medium text-sm truncate">{show.name}</span>
        </div>
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
          {countLabel && <span>{countLabel}</span>}
          {!compact && <ShowProgressStrip show={show} dense />}
        </div>
      </div>
    </button>
  );
}

const ShowListRow = memo(ShowListRowInner);
export default ShowListRow;
