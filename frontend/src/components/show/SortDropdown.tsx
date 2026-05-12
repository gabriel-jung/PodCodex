import { useState, type Dispatch, type SetStateAction } from "react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";

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
  const [open, setOpen] = useState(false);
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
    <div className="relative">
      <Button
        onClick={() => setOpen(!open)}
        variant="ghost"
        size="sm"
        className="text-xs h-7 px-2 gap-1.5"
        aria-label={`Sort by ${activeOption.label}`}
      >
        <span>{activeOption.label}</span>
        <TriggerArrow className="w-3 h-3 text-primary" />
      </Button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-full mt-1 z-50 bg-popover border border-border rounded-md shadow-lg p-1.5 min-w-[160px]">
            {OPTIONS.map((opt) => {
              const active = opt.field === activeField;
              return (
                <button
                  key={opt.field}
                  onClick={() => pick(opt.field)}
                  className={`w-full flex items-center justify-between gap-3 px-2 py-1.5 text-xs rounded transition ${
                    active
                      ? "bg-accent text-foreground"
                      : "text-muted-foreground hover:bg-accent/60 hover:text-foreground"
                  }`}
                >
                  <span className="font-medium">{opt.label}</span>
                  {active && (activeDir === "asc"
                    ? <ArrowUp className="w-3 h-3 text-primary" />
                    : <ArrowDown className="w-3 h-3 text-primary" />)}
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
