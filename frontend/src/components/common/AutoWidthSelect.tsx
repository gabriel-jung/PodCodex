/**
 * Native <select> that sizes to its currently-selected label, not to its
 * longest option. An invisible mirror span with the selected label defines
 * the container's intrinsic width; the real select is absolutely positioned
 * over it so its own longest-option measurement doesn't leak into layout.
 */

import type { ReactNode } from "react";
import { selectClass } from "@/lib/utils";

export interface AutoWidthSelectProps {
  value: string;
  onChange: (v: string) => void;
  selectedLabel: string;
  children: ReactNode;
  title?: string;
}

export default function AutoWidthSelect({
  value,
  onChange,
  selectedLabel,
  children,
  title,
}: AutoWidthSelectProps) {
  return (
    <div className="relative shrink-0 max-w-[18rem]">
      <span
        aria-hidden
        className={`${selectClass} text-xs invisible block whitespace-nowrap pr-7`}
      >
        {selectedLabel}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        title={title}
        className={`${selectClass} text-xs absolute inset-0 w-full`}
      >
        {children}
      </select>
    </div>
  );
}
