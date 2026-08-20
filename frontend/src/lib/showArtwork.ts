/** Single facility for resolving a show's cover image.
 *
 * Shows without artwork get the bundled default cover instead of an empty
 * tile. `artwork_url` may also hold the `"local"` marker (an uploaded file
 * served by the same backend route), so its value is only a presence flag
 * here, never a fetchable URL: always go through `/api/shows/artwork`.
 */

import { artworkUrl } from "@/api/filesystem";

const DEFAULT_COVER_URL = "/default-cover.png";

/** Marker value in `artwork_url` meaning "locally uploaded file". Generated
 *  from the backend constant; re-exported here so artwork consumers have one
 *  import for the whole concern. */
export { LOCAL_ARTWORK_MARKER } from "@/api/types";

/** Cover src for a show given its `artwork_url` field and folder path. */
export function showArtworkSrc(
  artworkUrlField: string | undefined | null,
  folder: string,
): string {
  return artworkUrlField ? artworkUrl(folder) : DEFAULT_COVER_URL;
}
