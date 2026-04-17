import type { Episode } from "@/api/types";
import { isOutdated } from "@/lib/utils";

export function PipelineBar({ ep }: { ep: Episode }) {
  const steps = [
    ep.downloaded,
    ep.transcribed,
    ep.corrected,
    ep.translations.length > 0,
    ep.synthesized,
    ep.indexed,
  ];
  const done = steps.filter(Boolean).length;
  if (done === 0) return null;

  const hasOutdated = isOutdated(ep);

  const pct = (done / steps.length) * 100;
  const color = done === steps.length
    ? (hasOutdated ? "bg-blue-500" : "bg-success")
    : (hasOutdated ? "bg-blue-500" : "bg-primary");

  return (
    <div className="h-1 bg-muted/50 rounded-full overflow-hidden">
      <div className={`h-full ${color} transition-all duration-300`} style={{ width: `${pct}%` }} />
    </div>
  );
}
