import type { Episode } from "@/api/types";

export function StatusDots({ ep }: { ep: Episode }) {
  return (
    <div className="flex gap-1.5 items-center">
      {ep.transcribed && <span className="w-2 h-2 rounded-full bg-blue-500" title="Transcribed" />}
      {ep.polished && <span className="w-2 h-2 rounded-full bg-purple-500" title="Polished" />}
      {ep.translations.length > 0 && <span className="w-2 h-2 rounded-full bg-teal-500" title={`Translated (${ep.translations.join(", ")})`} />}
      {ep.synthesized && <span className="w-2 h-2 rounded-full bg-orange-500" title="Synthesized" />}
      {ep.indexed && <span className="w-2 h-2 rounded-full bg-yellow-500" title="Indexed" />}
    </div>
  );
}
