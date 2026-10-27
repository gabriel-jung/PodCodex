import type { Dispatch, SetStateAction } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export type SortKey =
  | "date_desc" | "date_asc"
  | "title_asc" | "title_desc"
  | "duration_desc" | "duration_asc"
  | "number_desc" | "number_asc";

type SortField = "date" | "title" | "duration" | "number";

const OPTIONS: { field: SortField; label: string; defaultDir: "asc" | "desc" }[] = [
  { field: "date", label: "Date", defaultDir: "desc" },
  { field: "title", label: "Title", defaultDir: "asc" },
  { field: "duration", label: "Duration", defaultDir: "desc" },
  { field: "number", label: "Episode number", defaultDir: "desc" },
];

function parseSort(sort: SortKey): { field: SortField; dir: "asc" | "desc" } {
  const m = sort.match(/^(.+)_(asc|desc)$/);
  if (!m) return { field: "date", dir: "desc" };
  return { field: m[1] as SortField, dir: m[2] as "asc" | "desc" };
}

const buildSort = (field: SortField, dir: "asc" | "desc"): SortKey =>
  `${field}_${dir}` as SortKey;

export interface SortDropdownProps {
  sort: SortKey;
  setSort: Dispatch<SetStateAction<SortKey>>;
}

export default function SortDropdown({ sort, setSort }: SortDropdownProps) {
  const { field: activeField, dir: activeDir } = parseSort(sort);
  const activeOption = OPTIONS.find((o) => o.field === activeField) ?? OPTIONS[0];
  const TriggerArrow = activeDir === "asc" ? ArrowUp : ArrowDown;

  const pick = (field: SortField) => {
    if (field === activeField) {
      setSort(buildSort(field, activeDir === "asc" ? "desc" : "asc"));
      return;
    }
    const opt = OPTIONS.find((o) => o.field === field);
    if (opt) setSort(buildSort(field, opt.defaultDir));
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="text-xs h-7 px-2 gap-1.5"
          aria-label={`Sort by ${activeOption.label}`}
        >
          <span>{activeOption.label}</span>
          <TriggerArrow className="w-3 h-3 text-primary" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[160px]">
        {OPTIONS.map((opt) => {
          const active = opt.field === activeField;
          return (
            <DropdownMenuItem
              key={opt.field}
              // Stay open: picking the active field again flips its direction.
              onSelect={(e) => { e.preventDefault(); pick(opt.field); }}
              aria-current={active ? "true" : undefined}
              className={`justify-between gap-3 text-xs ${active ? "bg-accent text-foreground" : "text-muted-foreground"}`}
            >
              <span className="font-medium">{opt.label}</span>
              {active && (activeDir === "asc"
                ? <ArrowUp className="w-3 h-3 text-primary" />
                : <ArrowDown className="w-3 h-3 text-primary" />)}
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
