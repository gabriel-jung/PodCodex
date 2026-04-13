/** Browsable history of recent batch runs. Shows outcomes and links to affected episodes. */

import { useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  History,
  Trash2,
  XCircle,
} from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useBatchHistoryStore, type BatchHistoryEntry } from "@/stores";
import { EmptyState } from "@/components/ui/empty-state";
import { timeAgo, capitalize } from "@/lib/utils";

export default function BatchHistoryModal() {
  const isOpen = useBatchHistoryStore((s) => s.isOpen);
  const close = useBatchHistoryStore((s) => s.close);
  const entries = useBatchHistoryStore((s) => s.entries);
  const clear = useBatchHistoryStore((s) => s.clear);

  return (
    <Dialog open={isOpen} onOpenChange={(o) => !o && close()}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle>Recent batches</DialogTitle>
              <DialogDescription>Last {entries.length} batch run{entries.length === 1 ? "" : "s"}. Click an episode to open it.</DialogDescription>
            </div>
            {entries.length > 0 && (
              <Button onClick={clear} variant="ghost" size="sm" className="text-muted-foreground">
                <Trash2 className="w-3.5 h-3.5 mr-1" /> Clear
              </Button>
            )}
          </div>
        </DialogHeader>

        {entries.length === 0 ? (
          <EmptyState
            icon={History}
            title="No batches yet"
            description="Batch runs from any show will appear here after they finish."
          />
        ) : (
          <div className="max-h-[60vh] overflow-y-auto space-y-2 -mx-1 px-1">
            {entries.map((entry) => (
              <BatchHistoryRow key={entry.id} entry={entry} onNavigate={close} />
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function BatchHistoryRow({ entry, onNavigate }: { entry: BatchHistoryEntry; onNavigate: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();
  const remove = useBatchHistoryStore((s) => s.remove);

  const hasFailures = entry.failed.length > 0;
  const stepLabel = capitalize(entry.step);

  const goToShow = () => {
    onNavigate();
    navigate({ to: "/show/$folder", params: { folder: encodeURIComponent(entry.folder) } });
  };

  const goToEpisode = (stem: string) => {
    onNavigate();
    navigate({
      to: "/show/$folder/episode/$stem",
      params: { folder: encodeURIComponent(entry.folder), stem: encodeURIComponent(stem) },
    });
  };

  return (
    <div className="border border-border rounded-md p-3 hover:border-border/80 transition">
      <div className="flex items-center gap-3">
        {entry.status === "completed" && !hasFailures ? (
          <CheckCircle2 className="w-4 h-4 text-success shrink-0" />
        ) : hasFailures ? (
          <AlertTriangle className="w-4 h-4 text-destructive shrink-0" />
        ) : (
          <XCircle className="w-4 h-4 text-muted-foreground shrink-0" />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-medium">{stepLabel}</span>
            <button
              onClick={goToShow}
              className="text-xs text-muted-foreground hover:text-foreground hover:underline truncate"
            >
              {entry.showName || entry.folder}
            </button>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground mt-0.5">
            <span>{timeAgo(entry.timestamp) || "just now"}</span>
            <span>·</span>
            <span>{entry.totalCount} episode{entry.totalCount === 1 ? "" : "s"}</span>
            {entry.successCount > 0 && (
              <span className="text-success">{entry.successCount} done</span>
            )}
            {hasFailures && (
              <span className="text-destructive">{entry.failed.length} failed</span>
            )}
          </div>
        </div>
        {hasFailures && (
          <button
            onClick={() => setExpanded((v) => !v)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            Failed
          </button>
        )}
        <Button
          onClick={() => remove(entry.id)}
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
          aria-label="Remove entry"
        >
          <Trash2 className="w-3 h-3" />
        </Button>
      </div>
      {expanded && hasFailures && (
        <ul className="mt-2 pl-7 space-y-1">
          {entry.failed.map((f, i) => (
            <li key={i} className="flex items-center gap-2 text-xs">
              <button
                onClick={() => goToEpisode(f.stem)}
                className="truncate text-foreground hover:underline text-left max-w-[40%] shrink-0"
              >
                {f.title}
              </button>
              <span className="text-destructive/80 truncate flex-1 min-w-0" title={f.error}>
                {f.error}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
