/** Context menu for an episode row/card: Open, Play, Download, Delete. */

import { memo } from "react";
import { MoreVertical, Play, Download, Trash2, ExternalLink } from "lucide-react";
import type { Episode } from "@/api/types";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";

export interface EpisodeMenuProps {
  ep: Episode;
  onOpen: () => void;
  onPlay?: () => void;
  onDownload?: () => void;
  /** Delete the audio file only, keeping transcripts and versions. */
  onDelete?: () => void;
  /** Delete the whole episode: audio, output dir, versions, index chunks. */
  onDeleteEpisode?: () => void;
  /** Override the trigger button. If omitted renders a compact "⋯" button. */
  children?: React.ReactNode;
}

function EpisodeMenuInner({ ep, onOpen, onPlay, onDownload, onDelete, onDeleteEpisode, children }: EpisodeMenuProps) {
  const canPlay = !!onPlay && !!ep.audio_path;
  const canDownload = !!onDownload && !ep.downloaded;
  const canDelete = !!onDelete && !!ep.audio_path;
  // Gated on having a local footprint, not on audio_path: a subtitle-only
  // import has no audio file but does have versions and index chunks, and is
  // exactly the episode that would otherwise be impossible to remove. A feed
  // row that was never downloaded has nothing to delete, and offering the
  // action there means a destructive confirm that changes nothing (output_dir
  // is non-null even for those rows, so it cannot be part of the test).
  const hasLocalContent =
    !!ep.audio_path ||
    ep.transcribed ||
    ep.corrected ||
    ep.synthesized ||
    ep.indexed ||
    ep.translations.length > 0;
  const canDeleteEpisode = !!onDeleteEpisode && hasLocalContent;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        {children ?? (
          <button
            onClick={(e) => e.stopPropagation()}
            className="text-muted-foreground hover:text-foreground transition p-1 rounded hover:bg-accent/50"
            title="More actions"
            aria-label="More actions"
          >
            <MoreVertical className="w-3.5 h-3.5" />
          </button>
        )}
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onSelect={onOpen}>
          <ExternalLink className="w-3.5 h-3.5" /> Open episode
        </DropdownMenuItem>
        {canPlay && (
          <DropdownMenuItem onSelect={onPlay}>
            <Play className="w-3.5 h-3.5" /> Play
          </DropdownMenuItem>
        )}
        {canDownload && (
          <DropdownMenuItem onSelect={onDownload}>
            <Download className="w-3.5 h-3.5" /> Download audio
          </DropdownMenuItem>
        )}
        {(canDelete || canDeleteEpisode) && <DropdownMenuSeparator />}
        {canDelete && (
          <DropdownMenuItem variant="destructive" onSelect={onDelete}>
            <Trash2 className="w-3.5 h-3.5" /> Delete audio
          </DropdownMenuItem>
        )}
        {canDeleteEpisode && (
          <DropdownMenuItem variant="destructive" onSelect={onDeleteEpisode}>
            <Trash2 className="w-3.5 h-3.5" /> Delete episode...
          </DropdownMenuItem>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export const EpisodeMenu = memo(EpisodeMenuInner);
