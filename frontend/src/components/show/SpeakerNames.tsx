import { Users } from "lucide-react";

/** Icon + comma-joined, truncated speaker names. The caller owns the container
 *  (width, text size, `title` tooltip) so it fits both the list row's
 *  fixed-width column and the card's inline line. Pass `iconClass` to match
 *  the icon to the container's text size (w-3 for text-xs, w-2.5 for
 *  text-2xs, per the DESIGN.md pairing table). */
export function SpeakerNames({
  names,
  iconClass = "w-3 h-3",
}: {
  names: string[];
  iconClass?: string;
}) {
  return (
    <>
      <Users className={`${iconClass} shrink-0 opacity-70`} />
      <span className="truncate">{names.join(", ")}</span>
    </>
  );
}
