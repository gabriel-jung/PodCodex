import { memo } from "react";
import { artworkUrl } from "@/api/filesystem";
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
  const countLabel = showEpisodeCountLabel(show);
  return (
    <button
      onClick={() => onClick(show.path)}
      className="w-full text-left flex items-start gap-4 hover:bg-accent/50 transition border-b border-border last:border-0 px-4 py-3"
    >
      {show.artwork_url ? (
        <img src={artworkUrl(show.path)} alt={show.name} className="w-8 h-8 rounded shrink-0" />
      ) : (
        <div className="w-8 h-8 rounded bg-muted shrink-0" />
      )}
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
