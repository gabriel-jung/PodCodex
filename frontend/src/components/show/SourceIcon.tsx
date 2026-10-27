import type { ShowSummary } from "@/api/types";
import { FolderOpen, PlaySquare, Rss } from "lucide-react";

export function SourceIcon({ show }: { show: ShowSummary }) {
  if (show.has_youtube) return <PlaySquare aria-label="YouTube" className="w-3.5 h-3.5 shrink-0 text-destructive" />;
  if (show.has_rss) return <Rss aria-label="RSS" className="w-3.5 h-3.5 shrink-0 text-warning" />;
  return <FolderOpen aria-label="Local" className="w-3.5 h-3.5 shrink-0 text-muted-foreground" />;
}
