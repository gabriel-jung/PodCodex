import { useState } from "react";
import { HelpCircle, Trash2 } from "lucide-react";
import type { VersionEntry } from "@/api/types";
import { versionInfo, versionLabel, versionDate, isEdited } from "@/lib/utils";
import InlineConfirm from "@/components/common/InlineConfirm";

interface Props {
  version: VersionEntry;
  onOpen?: () => void;
  onDelete?: () => void;
  /** Compact padding for the History dropdown. */
  dense?: boolean;
  /** Subtle left accent for the latest version. */
  isLatest?: boolean;
}

export default function VersionRow({
  version,
  onOpen,
  onDelete,
  dense = false,
  isLatest = false,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const edited = isEdited(version);
  const dotColor = edited ? "bg-success" : "bg-info";
  const hash = version.content_hash.replace("sha256:", "").slice(0, 6);
  const padding = dense ? "px-3 py-1.5" : "px-4 py-2";
  const accent = isLatest ? "border-success/60" : "border-transparent";

  if (confirming && onDelete) {
    return (
      <div className={`${padding} border-l-2 ${accent}`}>
        <InlineConfirm
          message="Delete this version permanently?"
          onConfirm={() => {
            setConfirming(false);
            onDelete();
          }}
          onCancel={() => setConfirming(false)}
        />
      </div>
    );
  }

  const Label = onOpen ? "button" : "span";

  return (
    <div>
      <div
        className={`${padding} flex items-center gap-2 group/row hover:bg-accent/40 transition border-l-2 ${accent}`}
      >
        <span className={`shrink-0 w-1.5 h-1.5 rounded-full ${dotColor}`} />
        <Label
          onClick={onOpen}
          className={`flex-1 text-left truncate text-xs ${onOpen ? "hover:underline cursor-pointer" : ""}`}
        >
          <span className="text-foreground">{versionLabel(version)}</span>
          <span className="text-muted-foreground"> · {versionDate(version)}</span>
          {edited && <span className="ml-1.5 text-2xs text-success">edited</span>}
        </Label>
        <span className="shrink-0 font-mono text-2xs text-muted-foreground/60 tabular-nums">
          {hash}
        </span>
        <button
          onClick={() => setExpanded(!expanded)}
          className="shrink-0 text-muted-foreground/40 hover:text-muted-foreground p-0.5"
          title="Version details"
        >
          <HelpCircle className="w-3 h-3" />
        </button>
        {onDelete && (
          <button
            onClick={() => setConfirming(true)}
            className="shrink-0 text-muted-foreground/40 hover:text-destructive p-0.5 opacity-0 group-hover/row:opacity-100 transition"
            title="Delete this version"
          >
            <Trash2 className="w-3 h-3" />
          </button>
        )}
      </div>
      {expanded && (
        <div className={`${dense ? "px-3" : "px-4"} pb-2 text-xs`}>
          <div className="bg-secondary/50 rounded border border-border/50 px-2 py-1.5 space-y-0.5 ml-4">
            {versionInfo(version).map(({ key, value }) => (
              <div key={key} className="flex gap-2">
                <span className="text-muted-foreground shrink-0 w-24">{key}</span>
                <span className="truncate">{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
