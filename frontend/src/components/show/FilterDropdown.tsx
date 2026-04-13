import { useState } from "react";
import { Button } from "@/components/ui/button";
import { SlidersHorizontal } from "lucide-react";
import { useEpisodeStore } from "@/stores";

export default function FilterDropdown() {
  const minDurationMinutes = useEpisodeStore((s) => s.minDurationMinutes);
  const setMinDurationMinutes = useEpisodeStore((s) => s.setMinDurationMinutes);
  const maxDurationMinutes = useEpisodeStore((s) => s.maxDurationMinutes);
  const setMaxDurationMinutes = useEpisodeStore((s) => s.setMaxDurationMinutes);
  const titleInclude = useEpisodeStore((s) => s.titleInclude);
  const setTitleInclude = useEpisodeStore((s) => s.setTitleInclude);
  const titleExclude = useEpisodeStore((s) => s.titleExclude);
  const setTitleExclude = useEpisodeStore((s) => s.setTitleExclude);
  const [open, setOpen] = useState(false);
  const activeCount = [
    minDurationMinutes > 0,
    maxDurationMinutes > 0,
    titleInclude.length > 0,
    titleExclude.length > 0,
  ].filter(Boolean).length;

  const clearAll = () => {
    setMinDurationMinutes(0);
    setMaxDurationMinutes(0);
    setTitleInclude("");
    setTitleExclude("");
  };

  return (
    <div className="relative">
      <Button
        onClick={() => setOpen(!open)}
        variant={activeCount > 0 ? "secondary" : "ghost"}
        size="sm"
        className="text-xs h-7 px-2 gap-1"
      >
        <SlidersHorizontal className="w-3 h-3" />
        Filters
        {activeCount > 0 && <span className="bg-primary text-primary-foreground rounded-full px-1 text-2xs">{activeCount}</span>}
      </Button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute left-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg p-3 min-w-[240px] space-y-3">
            <div className="space-y-2">
              <span className="text-xs font-medium">Duration</span>
              <div className="flex items-center gap-2">
                <input
                  type="number" min={0} step={5}
                  value={minDurationMinutes || ""}
                  onChange={(e) => setMinDurationMinutes(Math.max(0, Number(e.target.value)))}
                  placeholder="min"
                  className="input w-16 text-xs text-center"
                />
                <span className="text-xs text-muted-foreground">to</span>
                <input
                  type="number" min={0} step={5}
                  value={maxDurationMinutes || ""}
                  onChange={(e) => setMaxDurationMinutes(Math.max(0, Number(e.target.value)))}
                  placeholder="max"
                  className="input w-16 text-xs text-center"
                />
                <span className="text-xs text-muted-foreground">min</span>
              </div>
            </div>
            <div className="space-y-2">
              <span className="text-xs font-medium">Title contains</span>
              <input
                value={titleInclude}
                onChange={(e) => setTitleInclude(e.target.value)}
                placeholder="word or phrase..."
                className="input w-full text-xs"
              />
            </div>
            <div className="space-y-2">
              <span className="text-xs font-medium">Title excludes</span>
              <input
                value={titleExclude}
                onChange={(e) => setTitleExclude(e.target.value)}
                placeholder="word or phrase..."
                className="input w-full text-xs"
              />
            </div>
            {activeCount > 0 && (
              <Button onClick={() => { clearAll(); setOpen(false); }} variant="ghost" size="sm" className="text-xs w-full">
                Clear all filters
              </Button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
