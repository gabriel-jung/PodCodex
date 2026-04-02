/** Global command palette — Cmd+K / Ctrl+K to open. */

import { useEffect, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Home,
  Settings,
  Sun,
  Moon,
  Monitor,
  Play,
  Pause,
  Podcast,
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
import { listShows } from "@/api/shows";
import { useAudioStore } from "@/stores";
import { useTheme } from "@/hooks/useTheme";

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const { theme, setTheme } = useTheme();
  const isPlaying = useAudioStore((s) => s.isPlaying);

  const { data: shows } = useQuery({
    queryKey: ["shows"],
    queryFn: listShows,
    enabled: open,
  });

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

  const run = (fn: () => void) => {
    fn();
    setOpen(false);
  };

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Type a command or search..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        <CommandGroup heading="Navigation">
          <CommandItem onSelect={() => run(() => navigate({ to: "/" }))}>
            <Home className="mr-2" />
            Home
          </CommandItem>
          <CommandItem onSelect={() => run(() => navigate({ to: "/settings" }))}>
            <Settings className="mr-2" />
            Settings
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

        <CommandGroup heading="Audio">
          <CommandItem
            onSelect={() =>
              run(() => useAudioStore.getState().setIsPlaying(!isPlaying))
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
