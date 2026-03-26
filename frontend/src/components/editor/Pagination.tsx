import { Button } from "@/components/ui/button";

const PAGE_SIZES = [10, 20, 50] as const;

interface PaginationProps {
  page: number;
  totalPages: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
}

export default function Pagination({
  page,
  totalPages,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: PaginationProps) {
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
      <div className="flex items-center gap-1">
        <Button
          onClick={() => onPageChange(page - 1)}
          disabled={page <= 0}
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs"
        >
          Prev
        </Button>
        <span>
          {page + 1} / {Math.max(totalPages, 1)}
        </span>
        <Button
          onClick={() => onPageChange(page + 1)}
          disabled={page >= totalPages - 1}
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs"
        >
          Next
        </Button>
      </div>
    </div>
  );
}
