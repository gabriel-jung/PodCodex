import { useState, useRef, useEffect, type RefObject } from "react";
import { Button } from "@/components/ui/button";
import { Search, X, SlidersHorizontal, Clock, Undo2, HelpCircle, Trash2, Download, History } from "lucide-react";
import type { VersionEntry } from "@/api/types";
import { exportTextUrl, exportSrtUrl, exportVttUrl, exportZipUrl } from "@/api/client";

function Tip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <span className="relative inline-flex">
      <button
        type="button"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="text-muted-foreground/40 hover:text-muted-foreground transition"
      >
        <HelpCircle className="w-3 h-3" />
      </button>
      {show && (
        <div className="absolute left-0 top-full mt-1 z-50 bg-popover text-popover-foreground text-xs rounded-md border border-border shadow-lg px-2.5 py-1.5 max-w-64 whitespace-normal">
          {text}
        </div>
      )}
    </span>
  );
}

/** Build a compact label for the version list row. */
function versionLabel(v: VersionEntry): string {
  const p = v.params as Record<string, unknown>;
  if (v.manual_edit || v.type === "validated") return "Manual edit";
  if (p.skipped) return "Skipped (copied)";
  const parts: string[] = [];
  if (v.model) parts.push(v.model);
  if (p.provider) parts.push(String(p.provider));
  else if (p.mode) parts.push(String(p.mode));
  if (p.language) parts.push(String(p.language));
  else if (p.source_lang && p.target_lang) parts.push(`${p.source_lang} → ${p.target_lang}`);
  else if (p.source_lang) parts.push(String(p.source_lang));
  if (p.diarize === false) parts.push("no diar");
  return parts.join(", ") || "Pipeline";
}

/** Params to hide from the version info box (internal / not user-relevant). */
const HIDDEN_PARAMS = new Set(["meta", "batch_size", "batch_minutes", "engine", "skipped"]);

/** Format all params as key: value lines for the info box. */
function versionInfo(v: VersionEntry): { key: string; value: string }[] {
  const rows: { key: string; value: string }[] = [];
  if (v.model) rows.push({ key: "Model", value: v.model });
  rows.push({ key: "Type", value: v.type === "validated" ? "Saved edit" : "Generated" });
  rows.push({ key: "Segments", value: String(v.segment_count) });
  rows.push({ key: "Hash", value: v.content_hash.replace("sha256:", "").slice(0, 8) });
  const p = v.params as Record<string, unknown>;
  for (const [k, val] of Object.entries(p)) {
    if (HIDDEN_PARAMS.has(k) || val === null || val === undefined) continue;
    const label = k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    rows.push({ key: label, value: typeof val === "boolean" ? (val ? "yes" : "no") : String(val) });
  }
  return rows;
}

interface EditorToolbarProps {
  totalSegments: number;
  visibleCount: number;
  isDirty: boolean;
  flaggedCount: number;
  deletedCount: number;
  speakers: string[];
  speakerFilter: string;
  onSpeakerFilterChange: (speaker: string) => void;
  showFlaggedOnly: boolean;
  onFlaggedFilterChange: (show: boolean) => void;
  showChangedOnly: boolean;
  onChangedFilterChange: (show: boolean) => void;
  hasReference: boolean;
  changedCount: number;
  densityThreshold: number;
  onDensityChange: (value: number) => void;
  maxDensityThreshold: number;
  onMaxDensityChange: (value: number) => void;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  searchRef?: RefObject<HTMLInputElement | null>;
  onSave: () => void;
  onUndo?: () => void;
  onDeleteFlagged: () => void;
  onEstimateTimestamps?: () => void;
  isSaving: boolean;
  audioPath?: string;
  exportSource?: string;
  versions?: VersionEntry[];
  onLoadVersion?: (id: string) => void;
}

export default function EditorToolbar({
  totalSegments,
  visibleCount,
  isDirty,
  flaggedCount,
  deletedCount,
  speakers,
  speakerFilter,
  onSpeakerFilterChange,
  showFlaggedOnly,
  onFlaggedFilterChange,
  showChangedOnly,
  onChangedFilterChange,
  hasReference,
  changedCount,
  densityThreshold,
  onDensityChange,
  maxDensityThreshold,
  onMaxDensityChange,
  searchQuery,
  onSearchChange,
  searchRef,
  onSave,
  onUndo,
  onDeleteFlagged,
  onEstimateTimestamps,
  isSaving,
  audioPath,
  exportSource,
  versions,
  onLoadVersion,
}: EditorToolbarProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showExport, setShowExport] = useState(false);
  const [showVersions, setShowVersions] = useState(false);
  const [expandedVersion, setExpandedVersion] = useState<string | null>(null);
  const exportRef = useRef<HTMLDivElement>(null);
  const versionsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showExport && !showVersions) return;
    const handler = (e: MouseEvent) => {
      if (showExport && exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setShowExport(false);
      }
      if (showVersions && versionsRef.current && !versionsRef.current.contains(e.target as Node)) {
        setShowVersions(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showExport, showVersions]);

  return (
    <div className="border-b border-border text-xs">
      {/* Row 1: Info & actions */}
      <div className="px-4 py-2 flex items-center gap-3">
        <span className="text-muted-foreground whitespace-nowrap">
          {totalSegments} segment{totalSegments !== 1 ? "s" : ""}
          {deletedCount > 0 && ` (${deletedCount} deleted)`}
        </span>

        {onEstimateTimestamps && (
          <Button onClick={onEstimateTimestamps} variant="outline" size="sm" className="h-6" title="Recalculate timestamps based on text length">
            <Clock className="w-3 h-3 mr-1" /> Estimate timestamps
          </Button>
        )}

        <div className="flex-1" />

        {versions && versions.length > 0 && onLoadVersion && (
          <div className="relative" ref={versionsRef}>
            <Button
              variant="outline"
              size="sm"
              className="h-6"
              onClick={() => setShowVersions(!showVersions)}
            >
              <History className="w-3 h-3 mr-1" /> History ({versions.length})
            </Button>
            {showVersions && (
              <div className="absolute right-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg py-1 min-w-72 max-h-80 overflow-y-auto">
                <div className="px-3 py-1 text-muted-foreground/60 flex items-center gap-3 border-b border-border/50 mb-1">
                  <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-blue-500" /> generated</span>
                  <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-green-500" /> saved edit</span>
                </div>
                {versions.map((v) => {
                  const d = new Date(v.timestamp);
                  const dateStr = d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
                  const timeStr = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
                  const label = versionLabel(v);
                  const isExpanded = expandedVersion === v.id;
                  const info = isExpanded ? versionInfo(v) : [];
                  return (
                    <div key={v.id} className="border-b border-border/30 last:border-0">
                      <div className="flex items-center gap-1.5 px-3 py-1.5 text-xs hover:bg-accent transition">
                        <span className={`shrink-0 w-1.5 h-1.5 rounded-full ${v.type === "validated" ? "bg-green-500" : "bg-blue-500"}`} />
                        <button
                          className="flex-1 text-left truncate"
                          onClick={() => { onLoadVersion(v.id); setShowVersions(false); }}
                        >
                          <span className="text-muted-foreground">{dateStr}, {timeStr}</span>
                          {" — "}
                          <span className="font-medium">{label}</span>
                          <span className="ml-1 text-muted-foreground/60">({v.segment_count} seg)</span>
                        </button>
                        <button
                          className="shrink-0 text-muted-foreground/40 hover:text-muted-foreground p-0.5"
                          onClick={() => setExpandedVersion(isExpanded ? null : v.id)}
                          title="Version details"
                        >
                          <HelpCircle className="w-3 h-3" />
                        </button>
                      </div>
                      {isExpanded && (
                        <div className="px-3 pb-2 ml-3 text-xs">
                          <div className="bg-secondary/50 rounded border border-border/50 px-2 py-1.5 space-y-0.5">
                            {info.map(({ key, value }) => (
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
                })}
              </div>
            )}
          </div>
        )}
        {versions && versions.length > 0 && audioPath && (
          <div className="w-px h-4 bg-border" />
        )}
        {audioPath && (
          <div className="relative" ref={exportRef}>
            <Button
              variant="outline"
              size="sm"
              className="h-6"
              onClick={() => setShowExport(!showExport)}
            >
              <Download className="w-3 h-3 mr-1" /> Export
            </Button>
            {showExport && (
              <div className="absolute right-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg py-1 min-w-36">
                {[
                  { label: "Plain Text", url: exportTextUrl(audioPath, exportSource) },
                  { label: "SRT Subtitles", url: exportSrtUrl(audioPath, exportSource) },
                  { label: "WebVTT Subtitles", url: exportVttUrl(audioPath, exportSource) },
                ].map(({ label, url }) => (
                  <a
                    key={label}
                    href={url}
                    download
                    className="block px-3 py-1.5 text-xs hover:bg-accent transition"
                    onClick={() => setShowExport(false)}
                  >
                    {label}
                  </a>
                ))}
                <div className="border-t border-border my-1" />
                <a
                  href={exportZipUrl(audioPath)}
                  download
                  className="block px-3 py-1.5 text-xs hover:bg-accent transition"
                  onClick={() => setShowExport(false)}
                >
                  ZIP (all files)
                </a>
              </div>
            )}
          </div>
        )}
        <Button onClick={onSave} disabled={isSaving} size="sm" className="h-6">
          {isSaving ? "Saving..." : isDirty ? "Save*" : "Save"}
        </Button>
        {!isDirty && !isSaving && <span className="text-green-500">up to date</span>}
      </div>

      {/* Row 2: Filters, undo, search */}
      <div className="px-4 py-1.5 flex items-center gap-3 border-t border-border/50">
        {speakers.length > 1 && (
          <select
            value={speakerFilter}
            onChange={(e) => onSpeakerFilterChange(e.target.value)}
            className="bg-secondary text-secondary-foreground rounded px-2 py-1 border border-border"
          >
            <option value="">All speakers</option>
            {speakers.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        )}

        {hasReference && changedCount > 0 && (
          <label className="flex items-center gap-1 cursor-pointer whitespace-nowrap">
            <input type="checkbox" checked={showChangedOnly} onChange={(e) => onChangedFilterChange(e.target.checked)} className="accent-blue-500" />
            <span className="text-blue-600 dark:text-blue-400">{changedCount} changed</span>
            <Tip text="Segments where the text differs from the previous step's version. Useful for reviewing what the AI changed." />
          </label>
        )}

        {flaggedCount > 0 && (
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-yellow-500/10 border border-yellow-500/20">
            <label className="flex items-center gap-1 cursor-pointer whitespace-nowrap">
              <input type="checkbox" checked={showFlaggedOnly} onChange={(e) => onFlaggedFilterChange(e.target.checked)} className="accent-yellow-500" />
              <span className="text-yellow-600 dark:text-yellow-400">{flaggedCount} flagged</span>
            </label>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className={`p-0.5 rounded hover:bg-yellow-500/20 transition ${showAdvanced ? "text-yellow-600 dark:text-yellow-400" : "text-yellow-600/50 dark:text-yellow-400/50"}`}
              title="Adjust density thresholds"
            >
              <SlidersHorizontal className="w-3 h-3" />
            </button>
            <button
              onClick={onDeleteFlagged}
              className="p-0.5 rounded text-yellow-600/50 dark:text-yellow-400/50 hover:text-yellow-600 dark:hover:text-yellow-400 hover:bg-yellow-500/20 transition"
              title="Delete all flagged segments"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        )}

        <div className="flex-1" />

        {visibleCount !== totalSegments && (
          <span className="text-muted-foreground">{visibleCount} shown</span>
        )}

        {onUndo && (
          <Button onClick={onUndo} variant="ghost" size="sm" className="h-6" title="Undo last action">
            <Undo2 className="w-3 h-3" />
          </Button>
        )}

        <div className="relative flex items-center">
          <Search className="absolute left-1.5 w-3 h-3 text-muted-foreground pointer-events-none" />
          <input
            ref={searchRef}
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search..."
            className="bg-secondary text-secondary-foreground rounded pl-6 pr-6 py-1 border border-border w-36 text-xs"
          />
          {searchQuery && (
            <button onClick={() => onSearchChange("")} className="absolute right-1.5 text-muted-foreground hover:text-foreground">
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      {/* Density sliders (expandable) */}
      {showAdvanced && (
        <div className="px-4 py-2 flex flex-col gap-1.5 border-t border-border/50">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground whitespace-nowrap w-24 flex items-center gap-1">
              Too sparse &lt;
              <Tip text="Flag segments where there's too little text for the time span. Normal speech is about 10-15 char/s." />
            </span>
            <input type="range" min={0} max={10} step={0.5} value={densityThreshold} onChange={(e) => onDensityChange(Number(e.target.value))} className="flex-1 max-w-48 accent-yellow-500 h-1" />
            <span className="text-muted-foreground tabular-nums w-14">{densityThreshold} char/s</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground whitespace-nowrap w-24 flex items-center gap-1">
              Too dense &gt;
              <Tip text="Flag segments where there's too much text for the time span. Usually means repeated/hallucinated text." />
            </span>
            <input type="range" min={10} max={100} step={5} value={maxDensityThreshold} onChange={(e) => onMaxDensityChange(Number(e.target.value))} className="flex-1 max-w-48 accent-yellow-500 h-1" />
            <span className="text-muted-foreground tabular-nums w-14">{maxDensityThreshold} char/s</span>
          </div>
        </div>
      )}
    </div>
  );
}
