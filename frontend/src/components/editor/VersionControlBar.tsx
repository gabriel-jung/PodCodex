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
import { VERIFIED_CAPTION } from "@/lib/verified";
import { Star, Trash2 } from "lucide-react";

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
  /** Enables the "★ Verified" toggle. Only set for transcript / corrected
   *  editors, the only steps a verified pointer may reference. */
  verifiableStep?: "transcript" | "corrected";
  /** The episode's currently-verified version id (any step). The star is
   *  rendered filled when this matches the picker's effective version. */
  verifiedVersionId?: string | null;
  /** Whether the episode's verified pointer targets this editor's step.
   *  Used to render the filled-star state without needing step equality
   *  outside this component. */
  verifiedStepMatches?: boolean;
  /** Called when the user clicks the star. ``targetId`` is the version the
   *  click acts on (the explicitly-selected version, or the latest); flip
   *  ``isCurrentlyVerified`` to know whether to clear or set the pointer. */
  onToggleVerified?: (targetId: string, isCurrentlyVerified: boolean) => void;
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
  verifiableStep,
  verifiedVersionId,
  verifiedStepMatches,
  onToggleVerified,
}: VersionControlBarProps) {
  const hasVersions = !!versions && versions.length > 0;
  // Star acts on the explicitly-picked version when set, otherwise on the
  // latest (which "" / null in the picker tracks).
  const effectiveTargetId =
    hasVersions ? (selectedVersionId ?? versions![0].id) : null;
  const showStar = !!verifiableStep && !!onToggleVerified && !!effectiveTargetId;
  const targetIsVerified =
    !!verifiedStepMatches && verifiedVersionId === effectiveTargetId;

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
          {showStar && (
            <button
              onClick={() => onToggleVerified!(effectiveTargetId!, targetIsVerified)}
              className={
                targetIsVerified
                  ? "flex items-center gap-1 px-1.5 py-0.5 rounded text-xs text-verified hover:bg-verified/10 transition shrink-0"
                  : "flex items-center gap-1 px-1.5 py-0.5 rounded text-xs text-muted-foreground/60 hover:text-verified hover:bg-verified/10 transition shrink-0"
              }
              aria-label={targetIsVerified ? "Unmark as verified" : "Mark as verified"}
              title={
                targetIsVerified
                  ? "Verified version. Click to clear."
                  : `Mark as verified (${VERIFIED_CAPTION}).`
              }
            >
              <Star
                className="w-3 h-3"
                fill={targetIsVerified ? "currentColor" : "none"}
              />
              {targetIsVerified ? "Verified" : "Mark verified"}
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
