import type { Episode } from "@/api/types";
import { langLabel } from "@/lib/utils";

/**
 * What a full episode delete will remove, itemized.
 *
 * Rendered inside the confirm dialog's `content` slot. Everything here comes
 * from the episode row the list already holds, so opening the dialog costs no
 * request.
 */
export function DeleteEpisodeSummary({
  ep,
  isFeedBacked,
}: {
  ep: Episode;
  isFeedBacked: boolean;
}) {
  const items: string[] = [];
  if (ep.audio_path) items.push("The downloaded audio file");
  if (ep.transcribed) items.push("The transcript and every saved version of it");
  if (ep.corrected) items.push("The corrected transcript");
  if (ep.translations.length)
    items.push(`Translations (${ep.translations.map(langLabel).join(", ")})`);
  if (ep.synthesized) items.push("Synthesized audio");
  if (ep.indexed) items.push("Search index entries, so it stops appearing in results");
  if (!items.length) items.push("This episode's folder");

  return (
    <div className="space-y-3 text-xs text-muted-foreground">
      <ul className="space-y-1">
        {items.map((label) => (
          <li key={label} className="flex gap-2">
            <span aria-hidden className="text-destructive">
              &bull;
            </span>
            <span>{label}</span>
          </li>
        ))}
      </ul>
      <p className="text-destructive">This cannot be undone.</p>
      {ep.audio_path && (
        <p>The file you originally imported from is not touched.</p>
      )}
      {isFeedBacked && (
        <p>
          This episode is still in the feed, so it will reappear as an
          undownloaded episode on the next refresh.
        </p>
      )}
    </div>
  );
}
