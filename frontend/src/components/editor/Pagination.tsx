import { Button } from "@/components/ui/button";
import { ChevronsLeft, ChevronsRight } from "lucide-react";

const PAGE_SIZES = [10, 20, 50] as const;

interface PaginationProps {
  page: number;
  totalPages: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
}

/** Build a window of page numbers around the current page. */
function pageWindow(current: number, total: number, maxButtons: number = 7): number[] {
  if (total <= maxButtons) return Array.from({ length: total }, (_, i) => i);
  const half = Math.floor(maxButtons / 2);
  let start = Math.max(0, current - half);
  let end = start + maxButtons;
  if (end > total) {
    end = total;
    start = end - maxButtons;
  }
  return Array.from({ length: end - start }, (_, i) => start + i);
}

export default function Pagination({
  page,
  totalPages,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: PaginationProps) {
  const pages = pageWindow(page, totalPages);

  return (
    <div className="flex items-center justify-between px-4 py-2 border-t border-border text-xs text-muted-foreground">
      <div className="flex items-center gap-2">
        <span>Show</span>
        <select
          value={pageSize}
          onChange={(e) => onPageSizeChange(Number(e.target.value))}
          className="bg-secondary text-secondary-foreground rounded px-1.5 py-0.5 border border-border"
        >
          {PAGE_SIZES.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <span>per page</span>
      </div>
      <div className="flex items-center gap-0.5">
        <Button
          onClick={() => onPageChange(0)}
          disabled={page <= 0}
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0"
          title="First page"
        >
          <ChevronsLeft className="w-3.5 h-3.5" />
        </Button>
        {pages.map((p) => (
          <Button
            key={p}
            onClick={() => onPageChange(p)}
            variant={p === page ? "secondary" : "ghost"}
            size="sm"
            className={`h-6 w-6 p-0 text-xs ${p === page ? "font-bold" : ""}`}
          >
            {p + 1}
          </Button>
        ))}
        <Button
          onClick={() => onPageChange(totalPages - 1)}
          disabled={page >= totalPages - 1}
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0"
          title="Last page"
        >
          <ChevronsRight className="w-3.5 h-3.5" />
        </Button>
      </div>
    </div>
  );
}
