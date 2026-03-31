import { ChevronUp, ChevronDown, ArrowUpDown } from "lucide-react";

export default function SortHeader({
  col, label, current, dir, onSort, className = "",
}: {
  col: string;
  label: string;
  current: string;
  dir: "asc" | "desc";
  onSort: (col: any) => void;
  className?: string;
}) {
  const active = current === col;
  return (
    <button
      onClick={() => onSort(col)}
      className={`flex items-center gap-0.5 hover:text-foreground transition group ${active ? "text-foreground" : ""} ${className}`}
    >
      <span>{label}</span>
      {active ? (
        dir === "asc" ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />
      ) : (
        <ArrowUpDown className="w-3 h-3 opacity-0 group-hover:opacity-50 transition-opacity" />
      )}
    </button>
  );
}
