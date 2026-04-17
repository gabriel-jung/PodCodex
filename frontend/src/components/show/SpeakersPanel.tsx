import { useState, useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import type { ShowMeta, SpeakerRosterEntry } from "@/api/types";
import { getSpeakerRoster, updateShowMeta } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { Button } from "@/components/ui/button";
import { ChevronRight, Plus, Star, X } from "lucide-react";
import { formatDuration, errorMessage } from "@/lib/utils";
import { speakerColor, speakerTint } from "@/lib/speakerColor";

interface SpeakersPanelProps {
  folder: string;
  meta: ShowMeta;
}

const SORT_OPTIONS = [
  { key: "talk",     label: "talk time", cmp: (a: SpeakerRosterEntry, b: SpeakerRosterEntry) => b.total_seconds - a.total_seconds },
  { key: "segments", label: "segments",  cmp: (a: SpeakerRosterEntry, b: SpeakerRosterEntry) => b.segment_count - a.segment_count },
  { key: "episodes", label: "episodes",  cmp: (a: SpeakerRosterEntry, b: SpeakerRosterEntry) => b.episode_count - a.episode_count },
  { key: "name",     label: "name",      cmp: (a: SpeakerRosterEntry, b: SpeakerRosterEntry) => a.name.localeCompare(b.name) },
] as const;
type SortKey = (typeof SORT_OPTIONS)[number]["key"];

export default function SpeakersPanel({ folder, meta }: SpeakersPanelProps) {
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const roster = useQuery({
    queryKey: queryKeys.speakerRoster(folder),
    queryFn: () => getSpeakerRoster(folder),
    staleTime: 60_000,
  });

  const [sort, setSort] = useState<SortKey>("talk");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [newSpeaker, setNewSpeaker] = useState("");

  const saveMutation = useMutation({
    mutationFn: (nextMeta: ShowMeta) => updateShowMeta(folder, nextMeta),
    onMutate: async (nextMeta) => {
      await queryClient.cancelQueries({ queryKey: queryKeys.showMeta(folder) });
      const prev = queryClient.getQueryData<ShowMeta>(queryKeys.showMeta(folder));
      queryClient.setQueryData(queryKeys.showMeta(folder), nextMeta);
      return { prev };
    },
    onError: (_err, _next, ctx) => {
      if (ctx?.prev) queryClient.setQueryData(queryKeys.showMeta(folder), ctx.prev);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.showMeta(folder) });
      queryClient.invalidateQueries({ queryKey: queryKeys.speakerRoster(folder) });
    },
  });

  const known = meta.speakers;
  const toggleKnown = (name: string) => {
    const next = known.includes(name) ? known.filter((s) => s !== name) : [...known, name];
    saveMutation.mutate({ ...meta, speakers: next });
  };

  const addSpeaker = () => {
    const trimmed = newSpeaker.trim();
    if (trimmed && !known.includes(trimmed)) {
      saveMutation.mutate({ ...meta, speakers: [...known, trimmed] });
      setNewSpeaker("");
    }
  };

  const sorted: SpeakerRosterEntry[] = useMemo(() => {
    const list = [...(roster.data?.speakers ?? [])];
    const cmp = SORT_OPTIONS.find((o) => o.key === sort)?.cmp;
    if (cmp) list.sort(cmp);
    return list;
  }, [roster.data, sort]);

  const totalTalk = (roster.data?.speakers ?? []).reduce((s, x) => s + x.total_seconds, 0);

  const toggleExpand = (name: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });
  };

  const openEpisode = (stem: string) => {
    navigate({
      to: "/show/$folder/episode/$stem",
      params: { folder: encodeURIComponent(folder), stem: encodeURIComponent(stem) },
    });
  };

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h3 className="text-sm font-medium">Speakers</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            {roster.data
              ? `${roster.data.episodes_with_transcripts} of ${roster.data.episodes_scanned} episode${roster.data.episodes_scanned === 1 ? "" : "s"} transcribed`
              : roster.isLoading ? "Scanning transcripts…" : "\u00A0"}
            {saveMutation.isPending && <span className="ml-2 text-warning">Saving…</span>}
            {saveMutation.isError && (
              <span className="ml-2 text-destructive">{errorMessage(saveMutation.error)}</span>
            )}
          </p>
        </div>
        <form
          className="flex gap-2"
          onSubmit={(e) => { e.preventDefault(); addSpeaker(); }}
        >
          <input
            value={newSpeaker}
            onChange={(e) => setNewSpeaker(e.target.value)}
            placeholder="Add known speaker…"
            className="input w-56"
          />
          <Button type="submit" variant="outline" size="sm" disabled={!newSpeaker.trim()}>
            <Plus className="w-3.5 h-3.5 mr-1" />Add
          </Button>
        </form>
      </div>

      <div className="flex items-center gap-1 text-xs">
        <span className="text-muted-foreground mr-1">Sort:</span>
        {SORT_OPTIONS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setSort(key)}
            className={`px-2 py-0.5 rounded-full border transition ${
              sort === key
                ? "bg-primary/15 border-primary/40 text-primary"
                : "border-border text-muted-foreground hover:text-foreground hover:bg-muted"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {roster.isLoading && (
        <p className="text-xs text-muted-foreground">Loading roster…</p>
      )}
      {roster.isError && (
        <p className="text-xs text-destructive">{errorMessage(roster.error)}</p>
      )}
      {roster.data && sorted.length === 0 && (
        <p className="text-xs text-muted-foreground">
          No speakers found yet — transcribe episodes to populate the roster.
        </p>
      )}

      <ul className="space-y-1.5">
        {sorted.map((sp) => {
          const isKnown = known.includes(sp.name);
          const isExpanded = expanded.has(sp.name);
          const pct = totalTalk > 0 ? (sp.total_seconds / totalTalk) * 100 : 0;
          const hasEpisodes = sp.episodes.length > 0;
          const color = speakerColor(sp.name);
          const tint = speakerTint(sp.name);

          return (
            <li key={sp.name} className="rounded-md border border-border bg-card overflow-hidden">
              <div
                className={`flex items-center gap-3 px-3 py-2 ${hasEpisodes ? "cursor-pointer hover:bg-muted/40" : ""} transition`}
                onClick={() => hasEpisodes && toggleExpand(sp.name)}
              >
                <ChevronRight
                  className={`w-3.5 h-3.5 shrink-0 text-muted-foreground transition-transform ${
                    hasEpisodes ? "" : "opacity-0"
                  } ${isExpanded ? "rotate-90" : ""}`}
                />

                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <span
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ background: color }}
                    aria-hidden
                  />
                  <span className="font-medium text-sm truncate" style={{ color: isKnown ? color : undefined }}>
                    {sp.name}
                  </span>
                  {sp.episode_count === 0 && (
                    <span className="text-[10px] italic text-muted-foreground/70 shrink-0">
                      unseen
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-4 text-xs text-muted-foreground shrink-0 font-mono">
                  <span title="Episodes">{sp.episode_count} ep</span>
                  <span title="Segments">{sp.segment_count.toLocaleString()} seg</span>
                  <span className="text-foreground/80 w-16 text-right" title="Talk time">
                    {sp.total_seconds > 0 ? formatDuration(sp.total_seconds) || "<1m" : "—"}
                  </span>
                </div>

                <button
                  onClick={(e) => { e.stopPropagation(); toggleKnown(sp.name); }}
                  className={`shrink-0 h-7 w-7 flex items-center justify-center rounded-full transition ${
                    isKnown
                      ? "text-primary hover:bg-primary/10"
                      : "text-muted-foreground/60 hover:text-foreground hover:bg-muted"
                  }`}
                  title={isKnown ? "Remove from known speakers" : "Mark as known speaker"}
                  aria-label={isKnown ? "Remove known speaker" : "Mark as known speaker"}
                >
                  <Star className={`w-3.5 h-3.5 ${isKnown ? "fill-current" : ""}`} />
                </button>

                {isKnown && sp.episode_count === 0 && (
                  <button
                    onClick={(e) => { e.stopPropagation(); toggleKnown(sp.name); }}
                    className="shrink-0 h-7 w-7 flex items-center justify-center rounded-full text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition"
                    title="Delete"
                    aria-label="Delete"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              {pct > 0 && (
                <div className="h-0.5 bg-muted/40">
                  <div
                    className="h-full transition-all"
                    style={{ width: `${pct}%`, background: tint }}
                  />
                </div>
              )}

              {isExpanded && hasEpisodes && (
                <ul className="border-t border-border/50 bg-muted/20 divide-y divide-border/40">
                  {sp.episodes.map((ep) => (
                    <li key={ep.stem}>
                      <button
                        onClick={() => openEpisode(ep.stem)}
                        className="w-full flex items-center gap-3 px-3 py-1.5 text-left text-xs hover:bg-muted/40 transition"
                      >
                        <span className="truncate flex-1">{ep.title}</span>
                        <span className="font-mono text-muted-foreground shrink-0">
                          {ep.segment_count} seg
                        </span>
                        <span className="font-mono text-muted-foreground w-14 text-right shrink-0">
                          {formatDuration(ep.total_seconds) || "<1m"}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
