import type { UseMutationResult } from "@tanstack/react-query";
import type { Segment, VoiceSample, SynthesisStatus } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Play, Check, Upload } from "lucide-react";
import { audioFileUrl } from "@/api/client";
import SectionHeader from "@/components/common/SectionHeader";
import { errorMessage } from "@/lib/utils";

/** Key for a segment selection — "speaker:start:end" */
export function segKey(seg: { speaker?: string; start: number; end: number }) {
  return `${seg.speaker || ""}:${seg.start}:${seg.end}`;
}

export interface VoiceExtractionSectionProps {
  // Segment data
  segmentsBySpeaker: Record<string, Segment[]>;
  allSpeakers: string[];

  // Selection state
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  expandedSeg: string | null;
  setExpandedSeg: (key: string | null) => void;
  showCount: Record<string, number>;
  setShowCount: (v: Record<string, number>) => void;

  // Time range filter
  timeFrom: string;
  setTimeFrom: (v: string) => void;
  timeTo: string;
  setTimeTo: (v: string) => void;

  // Speaker overrides
  speakerOverrides: Record<string, string>;
  setSpeakerOverrides: (v: Record<string, string>) => void;

  // Mutations
  extractMutation: UseMutationResult<unknown, Error, void>;
  uploadMutation: UseMutationResult<unknown, Error, { speaker: string; file: File }>;

  // Status & voice samples
  status: SynthesisStatus | undefined;
  voiceSamples: Record<string, VoiceSample[]> | undefined;

  // Audio playback
  seekTo: (path: string, time: number) => void;
  audioPath: string;
}

export default function VoiceExtractionSection({
  segmentsBySpeaker,
  allSpeakers,
  selected,
  setSelected,
  expandedSeg,
  setExpandedSeg,
  showCount,
  setShowCount,
  timeFrom,
  setTimeFrom,
  timeTo,
  setTimeTo,
  speakerOverrides,
  setSpeakerOverrides,
  extractMutation,
  uploadMutation,
  status,
  voiceSamples,
  seekTo,
  audioPath,
}: VoiceExtractionSectionProps) {
  return (
    <section className="space-y-3">
      <SectionHeader>1. Select Voice Samples</SectionHeader>
      <p className="text-xs text-muted-foreground">
        Choose the best segments for each speaker to use as voice cloning references.
        Play segments to audition them, check the ones you want, then extract.
      </p>

      {/* Time range filter */}
      <div className="flex items-center gap-2 text-xs">
        <span className="text-muted-foreground">Focus:</span>
        <input
          value={timeFrom}
          onChange={(e) => setTimeFrom(e.target.value)}
          placeholder="from (mm:ss)"
          className="input py-0.5 text-xs w-24"
        />
        <span className="text-muted-foreground">to</span>
        <input
          value={timeTo}
          onChange={(e) => setTimeTo(e.target.value)}
          placeholder="to (mm:ss)"
          className="input py-0.5 text-xs w-24"
        />
        {(timeFrom || timeTo) && (
          <button
            onClick={() => { setTimeFrom(""); setTimeTo(""); }}
            className="text-muted-foreground hover:text-foreground"
          >
            Clear
          </button>
        )}
      </div>

      {Object.keys(segmentsBySpeaker).length > 0 ? (
        <div className="space-y-3">
          {Object.entries(segmentsBySpeaker).map(([speaker, segs]) => {
            const speakerSelected = segs.filter((s) => selected.has(segKey(s))).length;
            return (
              <div key={speaker} className="text-sm">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="font-medium">{speaker}</span>
                  <span className="text-muted-foreground text-xs">
                    {segs.length} segment{segs.length !== 1 ? "s" : ""}
                    {speakerSelected > 0 && ` · ${speakerSelected} selected`}
                  </span>
                </div>
                <div className="flex flex-col gap-1 max-h-64 overflow-y-auto">
                  {(() => {
                    const filtered = segs
                      .filter((s) => (s.end - s.start) >= 2)
                      .sort((a, b) => (b.end - b.start) - (a.end - a.start));
                    const limit = showCount[speaker] || 10;
                    const visible = filtered.slice(0, limit);
                    const hasMore = filtered.length > limit;
                    return (
                      <>
                        {visible.map((seg) => {
                          const key = segKey(seg);
                          const isSelected = selected.has(key);
                          const isExpanded = expandedSeg === key;
                          const dur = seg.end - seg.start;
                          return (
                            <div
                              key={key}
                              className={`flex flex-col rounded text-xs transition cursor-pointer ${
                                isSelected
                                  ? "bg-primary/15 border border-primary/30"
                                  : "bg-secondary/50 border border-transparent hover:bg-secondary"
                              }`}
                              onClick={() => {
                                const next = new Set(selected);
                                if (isSelected) next.delete(key);
                                else next.add(key);
                                setSelected(next);
                                setExpandedSeg(isExpanded ? null : key);
                              }}
                            >
                              <div className="flex items-center gap-2 px-2 py-1">
                                <div className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 ${
                                  isSelected ? "bg-primary border-primary" : "border-border"
                                }`}>
                                  {isSelected && <Check className="w-3 h-3 text-primary-foreground" />}
                                </div>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    seekTo(audioPath, seg.start);
                                  }}
                                  className="shrink-0 p-0.5 rounded hover:bg-accent transition"
                                  title="Play this segment"
                                >
                                  <Play className="w-3 h-3" />
                                </button>
                                <span className="text-muted-foreground tabular-nums shrink-0 w-12">{dur.toFixed(1)}s</span>
                                <span className={isExpanded ? "flex-1" : "truncate flex-1"}>{seg.text}</span>
                                {/* Speaker reassignment */}
                                {allSpeakers.length > 1 && (
                                  <select
                                    value={speakerOverrides[key] || seg.speaker || ""}
                                    onClick={(e) => e.stopPropagation()}
                                    onChange={(e) => {
                                      e.stopPropagation();
                                      const next = { ...speakerOverrides };
                                      if (e.target.value === seg.speaker) delete next[key];
                                      else next[key] = e.target.value;
                                      setSpeakerOverrides(next);
                                    }}
                                    className="shrink-0 bg-transparent border border-border rounded px-1 py-0 text-2xs w-20 text-muted-foreground"
                                    title="Reassign speaker"
                                  >
                                    {allSpeakers.map((sp) => (
                                      <option key={sp} value={sp}>{sp}</option>
                                    ))}
                                  </select>
                                )}
                              </div>
                              {isExpanded && seg.text.length > 60 && (
                                <div className="px-2 pb-1.5 pl-[4.5rem] text-muted-foreground leading-relaxed">
                                  {seg.text}
                                </div>
                              )}
                            </div>
                          );
                        })}
                        {hasMore && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setShowCount({ ...showCount, [speaker]: limit + 20 });
                            }}
                            className="text-xs text-muted-foreground hover:text-foreground transition py-1 text-center"
                          >
                            Show more ({filtered.length - limit} remaining)
                          </button>
                        )}
                      </>
                    );
                  })()}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="text-xs text-muted-foreground italic">No transcript segments available.</p>
      )}

      <div className="flex items-center gap-3">
        <Button
          onClick={() => extractMutation.mutate()}
          disabled={selected.size === 0 || extractMutation.isPending}
          size="sm"
        >
          {extractMutation.isPending
            ? "Extracting..."
            : `Extract ${selected.size} sample${selected.size !== 1 ? "s" : ""}`}
        </Button>
        {status?.voice_samples_extracted && (
          <span className="text-xs text-success">Samples on disk</span>
        )}
        {extractMutation.isError && (
          <span className="text-xs text-destructive">
            {errorMessage(extractMutation.error)}
          </span>
        )}
      </div>

      {/* Extracted + uploaded samples */}
      <div className="space-y-2 border-t border-border/50 pt-3">
        <div className="flex items-center gap-3">
          <span className="text-xs text-muted-foreground font-medium">
            {voiceSamples && Object.keys(voiceSamples).length > 0
              ? "Extracted samples:"
              : "No samples yet"}
          </span>
        </div>
        {voiceSamples && Object.entries(voiceSamples).map(([speaker, samples]) => (
          <div key={speaker} className="text-sm">
            <div className="flex items-center gap-2">
              <span className="font-medium">{speaker}</span>
              <span className="text-muted-foreground text-xs">
                ({samples.length} sample{samples.length !== 1 ? "s" : ""})
              </span>
              <label className="cursor-pointer text-muted-foreground hover:text-foreground transition" title={`Upload audio for ${speaker}`}>
                <Upload className="w-3 h-3" />
                <input
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) uploadMutation.mutate({ speaker, file });
                    e.target.value = "";
                  }}
                />
              </label>
            </div>
            <div className="flex gap-2 mt-1 flex-wrap">
              {samples.map((s, i) => (
                <button
                  key={i}
                  onClick={() => {
                    const audio = new Audio(audioFileUrl(s.file));
                    audio.play();
                  }}
                  className="text-xs px-2 py-1 rounded bg-secondary hover:bg-accent transition border border-border"
                  title={s.text || `Sample ${i + 1}`}
                >
                  {s.duration.toFixed(1)}s
                </button>
              ))}
            </div>
          </div>
        ))}

        {/* Upload for a new/any speaker */}
        {allSpeakers.length > 0 && (
          <div className="flex items-center gap-2 text-xs pt-1">
            <label className="flex items-center gap-1.5 cursor-pointer text-muted-foreground hover:text-foreground transition">
              <Upload className="w-3.5 h-3.5" />
              <span>Upload a sample</span>
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) {
                    const speaker = allSpeakers[0];
                    const choice = prompt(`Upload for which speaker?\n${allSpeakers.join(", ")}`, speaker);
                    if (choice && allSpeakers.includes(choice)) {
                      uploadMutation.mutate({ speaker: choice, file });
                    }
                  }
                  e.target.value = "";
                }}
              />
            </label>
            {uploadMutation.isPending && <span className="text-muted-foreground">Uploading…</span>}
            {uploadMutation.isError && <span className="text-destructive">{errorMessage(uploadMutation.error)}</span>}
          </div>
        )}
      </div>
    </section>
  );
}
