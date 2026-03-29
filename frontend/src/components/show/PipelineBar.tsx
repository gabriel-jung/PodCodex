import type { Episode } from "@/api/types";

export function PipelineBar({ ep }: { ep: Episode }) {
  const steps = [ep.downloaded, ep.transcribed, ep.polished, ep.translations.length > 0, ep.synthesized, ep.indexed];
  const done = steps.filter(Boolean).length;
  const pct = (done / steps.length) * 100;
  if (done === 0) return null;
  const color = done === steps.length ? "bg-green-500" : "bg-primary";
  return (
    <div className="h-1 bg-muted/50 rounded-full overflow-hidden">
      <div className={`h-full ${color} transition-all duration-300`} style={{ width: `${pct}%` }} />
    </div>
  );
}
