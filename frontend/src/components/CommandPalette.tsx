/** Global command palette — Cmd+K / Ctrl+K to open. */

import { useEffect, useMemo, useRef, useState } from "react";
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
import { exactSearch } from "@/api/search";
import { queryKeys } from "@/api/queryKeys";
import { useAudioStore, useBatchHistoryStore } from "@/stores";
import { useTheme } from "@/hooks/useTheme";
import { formatTime } from "@/lib/utils";
import { speakerColor } from "@/lib/speakerColor";

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const navigate = useNavigate();
  const seekTo = useAudioStore((s) => s.seekTo);
  const { theme, setTheme } = useTheme();
  const isPlaying = useAudioStore((s) => s.isPlaying);

  const { data: shows } = useQuery({
    queryKey: queryKeys.shows(),
    queryFn: listShows,
    enabled: open,
  });

  const episodeQueries = useQueries({
    queries: (shows ?? []).map((show) => ({
      queryKey: queryKeys.episodes(show.path, undefined),
      queryFn: () => getEpisodes(show.path),
      enabled: open && !!shows,
      staleTime: 60_000,
    })),
  });

  const episodes = useMemo(() => {
    if (!shows) return [];
    return episodeQueries.flatMap((q, i) =>
      (q.data ?? []).map((ep) => ({ ...ep, showPath: shows[i].path, showName: shows[i].name || shows[i].path })),
    );
  }, [shows, episodeQueries]);

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
          show: show.name || show.path,
          top_k: 3,
        }),
      enabled: canSearchTranscripts,
      staleTime: 30_000,
      retry: false,
    })),
  });

  const transcriptHits = useMemo(() => {
    if (!shows) return [];
    type Hit = { showPath: string; showName: string; text: string; episode: string; speaker: string; start: number; end: number; score: number };
    const out: Hit[] = [];
    transcriptQueries.forEach((q, i) => {
      const show = shows[i];
      (q.data ?? []).forEach((r) => {
        out.push({
          showPath: show.path,
          showName: show.name || show.path,
          text: r.text,
          episode: r.episode,
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

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  const seekTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (seekTimer.current) clearTimeout(seekTimer.current);
  }, []);

  const openTranscriptHit = (hit: { showPath: string; episode: string; start: number }) => {
    // Episode value from search result is the stem-like string the indexer stored
    navigate({
      to: "/show/$folder/episode/$stem",
      params: { folder: encodeURIComponent(hit.showPath), stem: encodeURIComponent(hit.episode) },
    });
    // Defer the seek until after navigation + audio load; AudioBar picks up pendingSeek.
    if (seekTimer.current) clearTimeout(seekTimer.current);
    seekTimer.current = setTimeout(() => {
      const audioPath = useAudioStore.getState().audioPath;
      if (audioPath) seekTo(audioPath, hit.start);
    }, 400);
  };

  const run = (fn: () => void) => {
    fn();
    setOpen(false);
  };

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput
        placeholder="Search shows, episodes, transcripts…"
        value={query}
        onValueChange={setQuery}
      />
      <CommandList>
        <CommandEmpty>{transcriptLoading ? "Searching transcripts…" : "No results found."}</CommandEmpty>

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
                <CommandShortcut>{show.episode_count} episodes</CommandShortcut>
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
