import type { UseMutationResult } from "@tanstack/react-query";
import type { Segment, VoiceSample, SynthesisStatus } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Play, Check, Upload, ArrowRightLeft } from "lucide-react";
import { audioFileUrl } from "@/api/client";
import SectionHeader from "@/components/common/SectionHeader";
import { errorMessage } from "@/lib/utils";
import { segKey } from "@/lib/segKey";

const DEFAULT_SEGMENT_LIMIT = 3;

export interface VoiceExtractionSectionProps {
  // Segments already grouped by OUTPUT speaker (after applying reassignments).
  segmentsBySpeaker: Record<string, Segment[]>;
  allSpeakers: string[];

  // Selection state
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  expandedSeg: string | null;
  setExpandedSeg: (key: string | null) => void;
  showCount: Record<string, number>;
  setShowCount: (v: Record<string, number>) => void;

  // Speaker overrides — per-segment reassignment of output speaker.
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
  speakerOverrides,
  setSpeakerOverrides,
  extractMutation,
  uploadMutation,
  status,
  voiceSamples,
  seekTo,
  audioPath,
}: VoiceExtractionSectionProps) {
  // Output-speaker order: detected speakers with segments first (stable),
  // then reassignment-only keys, then detected speakers that have no
  // assigned segments yet (so users can still upload for them).
  const populatedSpeakers = allSpeakers.filter((s) => s in segmentsBySpeaker);
  const reassignedOnly = Object.keys(segmentsBySpeaker).filter(
    (s) => !allSpeakers.includes(s),
  );
  const emptySpeakers = allSpeakers.filter((s) => !(s in segmentsBySpeaker));
  const renderedSpeakers = [...populatedSpeakers, ...reassignedOnly, ...emptySpeakers];

  return (
    <section className="space-y-3 border-t border-border/50 pt-3">
      <SectionHeader help="Pick the clearest clips per speaker. The model clones voices from these. You can upload a custom clip or reassign segments.">
        2. Voice samples
      </SectionHeader>

      {renderedSpeakers.length > 0 ? (
        <div className="space-y-4">
          {renderedSpeakers.map((speaker) => (
            <SpeakerBlock
              key={speaker}
              speaker={speaker}
              segments={segmentsBySpeaker[speaker] ?? []}
              selected={selected}
              setSelected={setSelected}
              expandedSeg={expandedSeg}
              setExpandedSeg={setExpandedSeg}
              showCount={showCount}
              setShowCount={setShowCount}
              speakerOverrides={speakerOverrides}
              setSpeakerOverrides={setSpeakerOverrides}
              allSpeakers={allSpeakers}
              samples={voiceSamples?.[speaker] ?? []}
              uploadMutation={uploadMutation}
              seekTo={seekTo}
              audioPath={audioPath}
            />
          ))}
        </div>
      ) : (
        <p className="text-xs text-muted-foreground italic">No transcript segments available.</p>
      )}

      <div className="flex items-center gap-3 flex-wrap">
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
          <span className="text-xs text-destructive w-full">
            {errorMessage(extractMutation.error)}
          </span>
        )}
      </div>
    </section>
  );
}

interface SpeakerBlockProps {
  speaker: string;
  segments: Segment[];
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  expandedSeg: string | null;
  setExpandedSeg: (key: string | null) => void;
  showCount: Record<string, number>;
  setShowCount: (v: Record<string, number>) => void;
  speakerOverrides: Record<string, string>;
  setSpeakerOverrides: (v: Record<string, string>) => void;
  allSpeakers: string[];
  samples: VoiceSample[];
  uploadMutation: UseMutationResult<unknown, Error, { speaker: string; file: File }>;
  seekTo: (path: string, time: number) => void;
  audioPath: string;
}

function SpeakerBlock({
  speaker,
  segments,
  selected,
  setSelected,
  expandedSeg,
  setExpandedSeg,
  showCount,
  setShowCount,
  speakerOverrides,
  setSpeakerOverrides,
  allSpeakers,
  samples,
  uploadMutation,
  seekTo,
  audioPath,
}: SpeakerBlockProps) {
  const filtered = segments
    .filter((s) => s.end - s.start >= 2)
    .sort((a, b) => b.end - b.start - (a.end - a.start));
  const limit = showCount[speaker] || DEFAULT_SEGMENT_LIMIT;
  const visible = filtered.slice(0, limit);
  const hasMore = filtered.length > limit;
  const speakerSelected = segments.filter((s) => selected.has(segKey(s))).length;

  const uploading =
    uploadMutation.isPending && uploadMutation.variables?.speaker === speaker;
  const uploadError =
    uploadMutation.isError && uploadMutation.variables?.speaker === speaker;

  return (
    <div className="text-sm border border-border/60 rounded-md p-2.5 bg-secondary/20">
      <div className="flex items-center gap-2 mb-2 min-w-0">
        <span className="font-medium shrink-0">{speaker}</span>
        <span className="text-muted-foreground text-xs flex-1 min-w-0 truncate">
          {segments.length === 0
            ? "no segments"
            : `${segments.length} segment${segments.length !== 1 ? "s" : ""}`}
          {speakerSelected > 0 && ` · ${speakerSelected} selected`}
          {samples.length > 0 && ` · ${samples.length} sample${samples.length !== 1 ? "s" : ""} extracted`}
        </span>
        <label
          className="shrink-0 flex items-center gap-1 cursor-pointer text-xs text-muted-foreground hover:text-foreground transition"
          title={`Upload audio for ${speaker}`}
        >
          <Upload className="w-3 h-3" />
          <span>Upload</span>
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

      {samples.length > 0 && (
        <div className="flex gap-2 mb-2 flex-wrap">
          {samples.map((s, i) => (
            <button
              key={i}
              onClick={() => {
                const audio = new Audio(audioFileUrl(s.file));
                audio.play();
              }}
              className="text-xs px-2 py-1 rounded bg-background hover:bg-accent transition border border-border"
              title={s.text || `Sample ${i + 1}`}
            >
              {s.duration.toFixed(1)}s
            </button>
          ))}
        </div>
      )}

      {uploading && (
        <p className="text-xs text-muted-foreground mb-1">Uploading…</p>
      )}
      {uploadError && (
        <p className="text-xs text-destructive mb-1">
          {errorMessage(uploadMutation.error)}
        </p>
      )}

      {segments.length === 0 ? (
        <p className="text-xs text-muted-foreground italic">
          No segments assigned. Upload a custom clip above, or reassign a segment from another speaker.
        </p>
      ) : (
        <div className="flex flex-col gap-1">
          {visible.map((seg) => {
            const key = segKey(seg);
            const isSelected = selected.has(key);
            const isExpanded = expandedSeg === key;
            const dur = seg.end - seg.start;
            const originalSpeaker = seg.speaker || "";
            const reassigned = originalSpeaker && originalSpeaker !== speaker;
            return (
              <div
                key={key}
                className={`flex flex-col rounded text-xs transition cursor-pointer ${
                  isSelected
                    ? "bg-primary/15 border border-primary/30"
                    : "bg-background border border-transparent hover:bg-secondary"
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
                  <div
                    className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 ${
                      isSelected ? "bg-primary border-primary" : "border-border"
                    }`}
                  >
                    {isSelected && <Check className="w-3 h-3 text-primary-foreground" />}
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      seekTo(audioPath, seg.start);
                    }}
                    className="shrink-0 p-0.5 rounded hover:bg-accent transition"
                    title="Play this segment"
                    aria-label={`Play segment at ${seg.start.toFixed(1)}s`}
                  >
                    <Play className="w-3 h-3" />
                  </button>
                  <span className="text-muted-foreground tabular-nums shrink-0 w-12">
                    {dur.toFixed(1)}s
                  </span>
                  <span className={isExpanded ? "flex-1 whitespace-normal" : "truncate flex-1"}>
                    {seg.text}
                  </span>
                  {reassigned && (
                    <span
                      className="shrink-0 text-2xs px-1 py-0.5 rounded bg-accent/50 text-muted-foreground"
                      title={`Originally attributed to ${originalSpeaker}`}
                    >
                      from {originalSpeaker}
                    </span>
                  )}
                  {allSpeakers.length > 1 && (
                    <ReassignControl
                      segKey={key}
                      currentSpeaker={speaker}
                      originalSpeaker={originalSpeaker}
                      allSpeakers={allSpeakers}
                      speakerOverrides={speakerOverrides}
                      setSpeakerOverrides={setSpeakerOverrides}
                    />
                  )}
                </div>
              </div>
            );
          })}
          {hasMore && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowCount({ ...showCount, [speaker]: limit + 10 });
              }}
              className="text-xs text-muted-foreground hover:text-foreground transition py-1 text-center"
            >
              Show more ({filtered.length - limit} remaining)
            </button>
          )}
        </div>
      )}
    </div>
  );
}

interface ReassignControlProps {
  segKey: string;
  currentSpeaker: string;
  originalSpeaker: string;
  allSpeakers: string[];
  speakerOverrides: Record<string, string>;
  setSpeakerOverrides: (v: Record<string, string>) => void;
}

function ReassignControl({
  segKey,
  currentSpeaker,
  originalSpeaker,
  allSpeakers,
  speakerOverrides,
  setSpeakerOverrides,
}: ReassignControlProps) {
  return (
    <div className="relative shrink-0" onClick={(e) => e.stopPropagation()}>
      <ArrowRightLeft className="w-3 h-3 text-muted-foreground/60 absolute left-1 top-1/2 -translate-y-1/2 pointer-events-none" />
      <select
        value={currentSpeaker}
        onChange={(e) => {
          e.stopPropagation();
          const next = { ...speakerOverrides };
          if (e.target.value === originalSpeaker) delete next[segKey];
          else next[segKey] = e.target.value;
          setSpeakerOverrides(next);
        }}
        className="bg-transparent border border-border rounded pl-5 pr-1 py-0 text-2xs w-24 text-muted-foreground hover:text-foreground cursor-pointer"
        title={`Reassign to a different speaker (currently ${currentSpeaker})`}
      >
        {allSpeakers.map((sp) => (
          <option key={sp} value={sp}>
            {sp}
          </option>
        ))}
      </select>
    </div>
  );
}
