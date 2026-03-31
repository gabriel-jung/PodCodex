import { useMemo, useState, useEffect } from "react";
import type { Segment } from "@/api/types";

const UNKNOWN_SPEAKERS = new Set(["UNKNOWN", "UNK", "None", "none", ""]);

export interface FilterState {
  page: number;
  setPage: (p: number) => void;
  pageSize: number;
  setPageSize: (s: number) => void;
  speakerFilter: string;
  setSpeakerFilter: (s: string) => void;
  showFlaggedOnly: boolean;
  setShowFlaggedOnly: (b: boolean) => void;
  showChangedOnly: boolean;
  setShowChangedOnly: (b: boolean) => void;
  densityThreshold: number;
  setDensityThreshold: (n: number) => void;
  maxDensityThreshold: number;
  setMaxDensityThreshold: (n: number) => void;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
}

export interface FilteredResult {
  displaySegments: { segment: Segment; originalIndex: number; displayIndex: number }[];
  pageSegments: { segment: Segment; originalIndex: number; displayIndex: number }[];
  totalPages: number;
  flaggedCount: number;
}

export function flagReason(
  seg: Segment,
  densityThreshold: number,
  maxDensityThreshold: number,
): string | null {
  if (seg.speaker === "[BREAK]") return null;
  if (UNKNOWN_SPEAKERS.has(seg.speaker)) return "Unknown speaker";
  if (seg.speaker === "[remove]") return "Marked for removal";
  const dur = seg.end - seg.start;
  if (dur > 0) {
    const density = seg.text.length / dur;
    if (density < densityThreshold) return "Too little text for duration (sparse)";
    if (density > maxDensityThreshold) return "Too much text for duration (dense)";
  }
  return null;
}

export function useSegmentFiltering(): FilterState {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(20);
  const [speakerFilter, setSpeakerFilter] = useState("");
  const [showFlaggedOnly, setShowFlaggedOnly] = useState(false);
  const [showChangedOnly, setShowChangedOnly] = useState(false);
  const [densityThreshold, setDensityThreshold] = useState(2);
  const [maxDensityThreshold, setMaxDensityThreshold] = useState(50);
  const [searchQuery, setSearchQuery] = useState("");

  // Reset page when filters change
  useEffect(() => setPage(0), [speakerFilter, showFlaggedOnly, showChangedOnly, searchQuery]);

  return {
    page, setPage,
    pageSize, setPageSize,
    speakerFilter, setSpeakerFilter,
    showFlaggedOnly, setShowFlaggedOnly,
    showChangedOnly, setShowChangedOnly,
    densityThreshold, setDensityThreshold,
    maxDensityThreshold, setMaxDensityThreshold,
    searchQuery, setSearchQuery,
  };
}

export function useFilteredSegments(
  editedSegments: Segment[],
  originalIndices: number[],
  filters: FilterState,
  opts: {
    showFlags: boolean;
    dismissedFlags: Set<number>;
    isChanged: (seg: Segment, origIdx: number) => boolean;
  },
): FilteredResult {
  const { speakerFilter, showFlaggedOnly, showChangedOnly, searchQuery, page, pageSize, densityThreshold, maxDensityThreshold } = filters;
  const { showFlags, dismissedFlags, isChanged } = opts;
  const searchLower = searchQuery.toLowerCase().trim();

  const isFlaggedSeg = (seg: Segment, origIdx?: number): boolean => {
    if (origIdx != null && dismissedFlags.has(origIdx)) return false;
    return flagReason(seg, densityThreshold, maxDensityThreshold) !== null;
  };

  const displaySegments = useMemo(() => {
    const result: { segment: Segment; originalIndex: number; displayIndex: number }[] = [];

    let idx = 0;
    for (let e = 0; e < editedSegments.length; e++) {
      const seg = editedSegments[e];
      const origIdx = originalIndices[e];

      if (speakerFilter && seg.speaker !== speakerFilter && seg.speaker !== "[BREAK]") continue;
      if (showFlaggedOnly && showFlags && !isFlaggedSeg(seg, origIdx) && seg.speaker !== "[BREAK]") continue;
      if (showChangedOnly && !isChanged(seg, origIdx) && seg.speaker !== "[BREAK]") continue;
      if (searchLower && !seg.text.toLowerCase().includes(searchLower) && seg.speaker !== "[BREAK]") continue;
      if (seg.speaker === "[BREAK]" && result.length > 0 && result[result.length - 1].segment.speaker === "[BREAK]") continue;

      result.push({ segment: seg, originalIndex: origIdx, displayIndex: idx });
      idx++;
    }

    while (result.length > 0 && result[0].segment.speaker === "[BREAK]") result.shift();
    while (result.length > 0 && result[result.length - 1].segment.speaker === "[BREAK]") result.pop();

    return result;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editedSegments, originalIndices, speakerFilter, showFlaggedOnly, showChangedOnly, showFlags, searchLower, densityThreshold, maxDensityThreshold, dismissedFlags]);

  const totalPages = Math.ceil(displaySegments.length / pageSize);
  const pageSegments = displaySegments.slice(page * pageSize, (page + 1) * pageSize);

  // Clamp page when total pages shrinks
  useEffect(() => {
    if (totalPages > 0 && page >= totalPages) filters.setPage(totalPages - 1);
  }, [totalPages, page, filters]);

  const flaggedCount = useMemo(() => {
    return editedSegments.filter((seg, i) => isFlaggedSeg(seg, originalIndices[i])).length;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editedSegments, originalIndices, densityThreshold, maxDensityThreshold, dismissedFlags]);

  return { displaySegments, pageSegments, totalPages, flaggedCount };
}
