/** Global command palette — Cmd+K / Ctrl+K to open. */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useQuery, useQueries } from "@tanstack/react-query";
import {
  Home,
  Settings,
  Sun,
  Moon,
  Monitor,
  Play,
  Pause,
  Podcast,
  Mic,
  Keyboard,
  History,
  Quote,
} from "lucide-react";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command";
import { listShows, getEpisodes } from "@/api/shows";
import { exactSearch } from "@/api/client";
import { queryKeys } from "@/api/queryKeys";
import { useAudioStore, useBatchHistoryStore } from "@/stores";
import { useTheme } from "@/hooks/useTheme";
import { errorMessage, formatTime } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";
import { countLabel } from "@/lib/showCounts";

/** Episode items shown at most; the list is for jumping, not browsing. */
const MAX_EPISODE_ITEMS = 30;

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const navigate = useNavigate();
  const playEpisode = useAudioStore((s) => s.playEpisode);
  const { theme, setTheme } = useTheme();
  const isPlaying = useAudioStore((s) => s.isPlaying);

  const { data: shows } = useQuery({
    queryKey: queryKeys.shows(),
    queryFn: listShows,
    enabled: open,
  });

  // Episode lists are the heavy `/unified` read, one per show, so they load
  // only once the user types something to match against, and share the show
  // page's cache entry (and its staleness) instead of refetching every open.
  const episodeQuery = query.trim().toLowerCase();
  const wantEpisodes = open && !!shows && episodeQuery.length >= 2;
  // `combine` keeps the result referentially stable while the data is, so
  // the match below reruns on new data or a new query, not on every render.
  const episodeLists = useQueries({
    queries: (shows ?? []).map((show) => ({
      queryKey: queryKeys.episodesForFolder(show.path),
      queryFn: () => getEpisodes(show.path),
      enabled: wantEpisodes,
    })),
    combine: (results) => results.map((q) => q.data),
  });

  // Matched here and capped: cmdk would otherwise mount and score one item
  // per episode of every show on each keystroke.
  const episodes = useMemo(() => {
    if (!shows || !wantEpisodes) return [];
    const out: { id: string; stem?: string | null; title: string; showPath: string; showName: string }[] = [];
    episodeLists.forEach((list, i) => {
      const showName = shows[i].name || shows[i].path;
      for (const ep of list ?? []) {
        if (out.length >= MAX_EPISODE_ITEMS) return;
        if (`${ep.title} ${showName}`.toLowerCase().includes(episodeQuery)) {
          out.push({ id: ep.id, stem: ep.stem, title: ep.title, showPath: shows[i].path, showName });
        }
      }
    });
    return out;
  }, [shows, wantEpisodes, episodeLists, episodeQuery]);

  // Debounce the query for transcript fan-out — running N exact searches per
  // keystroke across every show wastes work when the user is still typing.
  // Trim before storing so whitespace variants share a single cache entry.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query.trim()), 300);
    return () => clearTimeout(t);
  }, [query]);

  const canSearchTranscripts = open && debouncedQuery.length >= 3 && !!shows;
  const transcriptQueries = useQueries({
    queries: (shows ?? []).map((show) => ({
      queryKey: ["transcriptSearch", show.path, debouncedQuery],
      queryFn: () =>
        exactSearch({
          query: debouncedQuery,
          show_id: show.id,
          top_k: 3,
        }),
      // A show with no id was never indexed (indexing mints one).
      enabled: canSearchTranscripts && !!show.id,
      staleTime: 30_000,
      retry: false,
    })),
  });

  const transcriptHits = useMemo(() => {
    if (!shows) return [];
    type Hit = { showPath: string; showName: string; text: string; episode: string; stem: string; audioPath: string; speaker: string; start: number; end: number; score: number };
    const out: Hit[] = [];
    transcriptQueries.forEach((q, i) => {
      const show = shows[i];
      (q.data ?? []).forEach((r) => {
        out.push({
          showPath: show.path,
          showName: show.name || show.path,
          text: r.text,
          episode: r.episode,
          stem: r.episode_stem,
          audioPath: r.audio_path,
          speaker: r.speaker,
          start: r.start,
          end: r.end,
          score: r.score,
        });
      });
    });
    return out.sort((a, b) => b.score - a.score).slice(0, 8);
  }, [shows, transcriptQueries]);

  const transcriptLoading = canSearchTranscripts && transcriptQueries.some((q) => q.isFetching);
  // These run with retry: false, so a backend error or a missing rag extra
  // rejects at once; reading `data ?? []` alone would report it as a miss.
  const transcriptError = canSearchTranscripts
    ? transcriptQueries.find((q) => q.isError)?.error
    : undefined;

  const handleOpenChange = useCallback((next: boolean) => {
    setOpen(next);
    if (!next) setQuery("");
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((prev) => {
          if (prev) setQuery("");
          return !prev;
        });
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  // `episode` is the display title; the route and the player need the stem
  // and the hit's own audio file, not whatever track is already loaded.
  const openTranscriptHit = (hit: { showPath: string; showName: string; episode: string; stem: string; audioPath: string; start: number }) => {
    navigate({
      to: "/show/$folder/episode/$stem",
      params: { folder: encodeURIComponent(hit.showPath), stem: encodeURIComponent(hit.stem) },
    });
    if (hit.audioPath) {
      playEpisode(hit.audioPath, hit.start, {
        title: hit.episode,
        showName: hit.showName,
        folder: hit.showPath,
        stem: hit.stem,
      });
    }
  };

  const run = (fn: () => void) => {
    fn();
    handleOpenChange(false);
  };

  return (
    <CommandDialog open={open} onOpenChange={handleOpenChange}>
      <CommandInput
        placeholder="Search shows, episodes, transcripts…"
        value={query}
        onValueChange={setQuery}
      />
      <CommandList>
        <CommandEmpty>
          {transcriptLoading
            ? "Searching transcripts…"
            : transcriptError
              ? `Transcript search failed: ${errorMessage(transcriptError)}`
              : "No results found."}
        </CommandEmpty>

        {transcriptHits.length > 0 && (
          <CommandGroup heading="Transcripts">
            {transcriptHits.map((hit, i) => (
              <CommandItem
                key={`${hit.showPath}-${hit.episode}-${hit.start}-${i}`}
                value={`${query} ${hit.text}`}
                onSelect={() => run(() => openTranscriptHit(hit))}
                className="flex-col items-start gap-0.5"
              >
                <div className="flex items-center gap-2 w-full">
                  <Quote className="shrink-0 opacity-60" />
                  <span className="line-clamp-1 flex-1 text-sm">{hit.text}</span>
                </div>
                <div className="flex items-center gap-2 text-2xs text-muted-foreground pl-7">
                  <span className="truncate max-w-[12rem]">{hit.showName}</span>
                  <span className="opacity-50">·</span>
                  <span className="truncate max-w-[14rem]">{hit.episode}</span>
                  <span className="opacity-50">·</span>
                  <span className="font-mono">{formatTime(hit.start, false)}</span>
                  {hit.speaker && (
                    <>
                      <span className="opacity-50">·</span>
                      <span className="font-medium" style={{ color: speakerColor(hit.speaker) }}>{hit.speaker}</span>
                    </>
                  )}
                </div>
              </CommandItem>
            ))}
          </CommandGroup>
        )}

        <CommandGroup heading="Navigation">
          <CommandItem onSelect={() => run(() => navigate({ to: "/" }))}>
            <Home className="mr-2" />
            Home
          </CommandItem>
          <CommandItem onSelect={() => run(() => navigate({ to: "/settings" }))}>
            <Settings className="mr-2" />
            Settings
          </CommandItem>
          <CommandItem
            onSelect={() =>
              run(() =>
                document.dispatchEvent(new KeyboardEvent("keydown", { key: "?", shiftKey: true })),
              )
            }
          >
            <Keyboard className="mr-2" />
            Keyboard shortcuts
            <CommandShortcut>Shift+?</CommandShortcut>
          </CommandItem>
          <CommandItem
            onSelect={() => run(() => useBatchHistoryStore.getState().open())}
          >
            <History className="mr-2" />
            Recent batches
          </CommandItem>
        </CommandGroup>

        {shows && shows.length > 0 && (
          <CommandGroup heading="Shows">
            {shows.map((show) => (
              <CommandItem
                key={show.path}
                onSelect={() =>
                  run(() =>
                    navigate({
                      to: "/show/$folder",
                      params: { folder: show.path },
                    }),
                  )
                }
              >
                <Podcast className="mr-2" />
                {show.name || show.path}
                <CommandShortcut>{countLabel(show.episode_count ?? 0, "episode")}</CommandShortcut>
              </CommandItem>
            ))}
          </CommandGroup>
        )}

        {episodes.length > 0 && (
          <CommandGroup heading="Episodes">
            {episodes.map((ep) => (
              <CommandItem
                key={`${ep.showPath}/${ep.id}`}
                value={`${ep.title} ${ep.showName}`}
                onSelect={() =>
                  run(() =>
                    navigate({
                      to: "/show/$folder/episode/$stem",
                      params: {
                        folder: encodeURIComponent(ep.showPath),
                        stem: encodeURIComponent(ep.stem || ep.id),
                      },
                    }),
                  )
                }
              >
                <Mic className="mr-2 shrink-0" />
                <span className="truncate">{ep.title}</span>
                <CommandShortcut>{ep.showName}</CommandShortcut>
              </CommandItem>
            ))}
          </CommandGroup>
        )}

        <CommandGroup heading="Audio">
          <CommandItem
            onSelect={() =>
              run(() => {
                if (isPlaying) {
                  useAudioStore.getState().pauseAudio();
                } else {
                  const { currentTime } = useAudioStore.getState();
                  useAudioStore.setState({ pendingSeek: currentTime || 0 });
                }
              })
            }
          >
            {isPlaying ? <Pause className="mr-2" /> : <Play className="mr-2" />}
            {isPlaying ? "Pause" : "Play"}
          </CommandItem>
        </CommandGroup>

        <CommandGroup heading="Theme">
          <CommandItem
            onSelect={() => run(() => setTheme("light"))}
            disabled={theme === "light"}
          >
            <Sun className="mr-2" />
            Light mode
          </CommandItem>
          <CommandItem
            onSelect={() => run(() => setTheme("dark"))}
            disabled={theme === "dark"}
          >
            <Moon className="mr-2" />
            Dark mode
          </CommandItem>
          <CommandItem
            onSelect={() => run(() => setTheme("system"))}
            disabled={theme === "system"}
          >
            <Monitor className="mr-2" />
            System theme
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
