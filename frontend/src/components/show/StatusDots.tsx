import type { Episode } from "@/api/types";

type StepStatus = "none" | "outdated" | "done";

function Dot({ status, color, title }: { status: StepStatus; color: string; title: string }) {
  if (status === "none") return null;
  if (status === "outdated") {
    return (
      <span
        className={`w-2 h-2 rounded-full border-[1.5px] ${color.replace("bg-", "border-")}`}
        title={`${title} (outdated)`}
      />
    );
  }
  return <span className={`w-2 h-2 rounded-full ${color}`} title={title} />;
}

export function StatusDots({ ep }: { ep: Episode }) {
  const translateStatus: StepStatus = ep.translations.length > 0
    ? (ep.translate_status || "done")
    : "none";

  return (
    <div className="flex gap-1.5 items-center">
      <Dot status={ep.transcribe_status || (ep.transcribed ? "done" : "none")} color="bg-blue-500" title="Transcribed" />
      <Dot status={ep.polish_status || (ep.polished ? "done" : "none")} color="bg-purple-500" title="Polished" />
      <Dot
        status={translateStatus}
        color="bg-teal-500"
        title={ep.translations.length > 0 ? `Translated (${ep.translations.join(", ")})` : "Translated"}
      />
      {ep.synthesized && <span className="w-2 h-2 rounded-full bg-orange-500" title="Synthesized" />}
      {ep.indexed && <span className="w-2 h-2 rounded-full bg-yellow-500" title="Indexed" />}
    </div>
  );
}
