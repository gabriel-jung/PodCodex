/**
 * VersionControlBar — the row above the action bar that picks a version to
 * view, optionally a version to compare against (diff reference), exposes a
 * delete button for the current version, and reveals file metadata.
 *
 * Pure presentational. Parent owns all state.
 */

import type { VersionEntry } from "@/api/types";
import VersionPicker, { type PrependOption } from "@/components/common/VersionPicker";
import SectionHeader from "@/components/common/SectionHeader";
import { confirmDialog } from "@/components/ui/confirm-dialog";
import { versionInfo, versionOption } from "@/lib/utils";
import { Trash2 } from "lucide-react";

export type VersionInfoItem = ReturnType<typeof versionInfo>[number];

export interface VersionControlBarProps {
  versions: VersionEntry[] | undefined;
  selectedVersionId: string | null;
  onSelectVersion: (id: string | null) => void;
  hasCompareOptions: boolean;
  compareVersions: VersionEntry[];
  refChoice: string;
  onRefChoiceChange: (choice: string) => void;
  compareExtras: PrependOption[];
  refNoneSentinel: string;
  /** Source label fallback when no versions exist. */
  sourceLabel?: string;
  infoItems: VersionInfoItem[];
  expandedInfo: boolean;
  setExpandedInfo: (v: boolean) => void;
  /** Present iff the parent supports version deletion. */
  onDeleteVersion?: (id: string) => void;
}

export default function VersionControlBar({
  versions,
  selectedVersionId,
  onSelectVersion,
  hasCompareOptions,
  compareVersions,
  refChoice,
  onRefChoiceChange,
  compareExtras,
  refNoneSentinel,
  sourceLabel,
  infoItems,
  expandedInfo,
  setExpandedInfo,
  onDeleteVersion,
}: VersionControlBarProps) {
  const hasVersions = !!versions && versions.length > 0;

  return (
    <>
      {hasVersions ? (
        <div className="flex items-center gap-2 flex-wrap">
          <SectionHeader className="shrink-0 w-20">Version</SectionHeader>
          <VersionPicker
            versions={versions!}
            value={selectedVersionId}
            onChange={onSelectVersion}
            widthFit
          />
          {hasCompareOptions && (
            <>
              <span className="text-xs text-muted-foreground/60">vs</span>
              <VersionPicker
                versions={compareVersions}
                value={refChoice}
                onChange={(v) => onRefChoiceChange(v ?? refNoneSentinel)}
                showLatest={false}
                prependOptions={compareExtras}
                widthFit
              />
            </>
          )}
          <div className="flex-1" />
          {infoItems.length > 0 && (
            <button
              onClick={() => setExpandedInfo(!expandedInfo)}
              className="text-xs text-muted-foreground/60 hover:text-muted-foreground transition shrink-0"
            >
              File details
            </button>
          )}
          {onDeleteVersion && (
            <button
              onClick={() => {
                const targetId = selectedVersionId ?? versions![0].id;
                const target = versions!.find((v) => v.id === targetId);
                if (!target) return;
                confirmDialog.open({
                  title: "Delete this version?",
                  description: `${versionOption(target)}. Removes both the file and the database entry, and cannot be undone.`,
                  confirmLabel: "Delete",
                  variant: "destructive",
                  onConfirm: () => {
                    onDeleteVersion(targetId);
                  },
                });
              }}
              className="p-1 rounded text-muted-foreground/60 hover:text-destructive hover:bg-destructive/10 transition shrink-0"
              aria-label="Delete version"
              title="Delete current version (file + db entry)"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          )}
        </div>
      ) : sourceLabel ? (
        <div className="flex items-center gap-2">
          <SectionHeader className="shrink-0 w-20">Version</SectionHeader>
          <span className="text-xs text-muted-foreground font-mono">{sourceLabel}</span>
        </div>
      ) : null}

      {hasCompareOptions && !hasVersions && (
        <div className="flex items-center gap-2">
          <SectionHeader className="shrink-0 w-20">Compare</SectionHeader>
          <VersionPicker
            versions={[]}
            value={refChoice}
            onChange={(v) => onRefChoiceChange(v ?? refNoneSentinel)}
            showLatest={false}
            prependOptions={compareExtras}
          />
        </div>
      )}

      {infoItems.length > 0 && expandedInfo && (
        <div className="bg-secondary/50 rounded border border-border/50 px-3 py-2 text-xs space-y-0.5">
          {infoItems.map(({ key, value }) => (
            <div key={key} className="flex gap-2">
              <span className="text-muted-foreground shrink-0 w-20">{key}</span>
              <span className="truncate">{value}</span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
