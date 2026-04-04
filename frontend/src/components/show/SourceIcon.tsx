import type { ShowSummary } from "@/api/types";
import { FolderOpen, PlaySquare, Rss } from "lucide-react";

export function SourceIcon({ show }: { show: ShowSummary }) {
  if (show.has_youtube) return <PlaySquare className="w-3.5 h-3.5 text-red-500" title="YouTube" />;
  if (show.has_rss) return <Rss className="w-3.5 h-3.5 text-orange-500" title="RSS" />;
  return <FolderOpen className="w-3.5 h-3.5 text-muted-foreground" title="Local" />;
}
