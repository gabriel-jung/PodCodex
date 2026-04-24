/** Context menu for an episode row/card — Play, Download, Process (per-step), Delete. */

import { memo } from "react";
import { MoreVertical, Play, Download, Trash2, Sparkles, FileText, Languages, Database, ExternalLink } from "lucide-react";
import type { Episode } from "@/api/types";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubTrigger,
  DropdownMenuSubContent,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";

export type PipelineStep = "transcribe" | "correct" | "translate" | "index";

export interface EpisodeMenuProps {
  ep: Episode;
  onOpen: () => void;
  onPlay?: () => void;
  onDownload?: () => void;
  onDelete?: () => void;
  onProcess?: (step: PipelineStep) => void;
  /** Override the trigger button. If omitted renders a compact "⋯" button. */
  children?: React.ReactNode;
}

const STEP_LABELS: Record<PipelineStep, { icon: typeof Sparkles; label: string }> = {
  transcribe: { icon: Sparkles, label: "Transcribe" },
  correct: { icon: FileText, label: "Correct" },
  translate: { icon: Languages, label: "Translate" },
  index: { icon: Database, label: "Index" },
};

function EpisodeMenuInner({ ep, onOpen, onPlay, onDownload, onDelete, onProcess, children }: EpisodeMenuProps) {
  const canProcess = !!onProcess && (ep.downloaded || ep.has_subtitles || (ep.transcribed && ep.output_dir));
  const canPlay = !!onPlay && !!ep.audio_path;
  const canDownload = !!onDownload && !ep.downloaded;
  const canDelete = !!onDelete && !!ep.audio_path;

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
        {canProcess && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <Sparkles className="w-3.5 h-3.5" /> Process
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent>
                {(Object.entries(STEP_LABELS) as [PipelineStep, typeof STEP_LABELS[PipelineStep]][]).map(([step, { icon: Icon, label }]) => (
                  <DropdownMenuItem key={step} onSelect={() => onProcess?.(step)}>
                    <Icon className="w-3.5 h-3.5" /> {label}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          </>
        )}
        {canDelete && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem variant="destructive" onSelect={onDelete}>
              <Trash2 className="w-3.5 h-3.5" /> Delete audio
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export const EpisodeMenu = memo(EpisodeMenuInner);
