import { Eye, EyeOff } from "lucide-react";
import { useLayoutStore } from "@/stores";

/**
 * Eye/EyeOff button that toggles `compact` view mode in the global layout
 * store. Hides per-episode/per-show pipeline progress strips. Used in the
 * HomePage and ShowPage toolbars.
 */
export default function CompactToggle() {
  const compact = useLayoutStore((s) => s.compact);
  const setCompact = useLayoutStore((s) => s.setCompact);
  return (
    <button
      onClick={() => setCompact(!compact)}
      className={`px-1.5 py-1 rounded transition ${compact ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
      title={compact ? "Show details" : "Hide details (compact)"}
      aria-label="Toggle compact view"
    >
      {compact ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
    </button>
  );
}
