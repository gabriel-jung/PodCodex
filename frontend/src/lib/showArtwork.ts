/** Single facility for resolving a show's cover image.
 *
 * Shows without artwork get the bundled default cover instead of an empty
 * tile. `artwork_url` may also hold the `"local"` marker (an uploaded file
 * served by the same backend route), so its value is only a presence flag
 * here, never a fetchable URL: always go through `/api/shows/artwork`.
 */

import { create } from "zustand";
import { artworkUrl } from "@/api/filesystem";

const DEFAULT_COVER_URL = "/default-cover.png";

/** Marker value in `artwork_url` meaning "locally uploaded file". Generated
 *  from the backend constant; re-exported here so artwork consumers have one
 *  import for the whole concern. */
export { LOCAL_ARTWORK_MARKER } from "@/api/types";

/**
 * Cover-URL revision, folded into every cover URL below. Three states:
 * `OFFLINE_EPOCH` while the backend is known not to be listening, `0` when
 * nothing was ever rendered against a closed port, and a positive counter
 * afterwards.
 *
 * The shell now paints before the sidecar is listening, from the restored
 * query cache, so show cards exist at the first frame. Their `<img>`
 * requests would fire immediately and fail on the closed port, and a failed
 * image never retries on its own: refetching the shows query hands back the
 * same `src` string, so React leaves the attribute alone and the tile stays
 * broken until something remounts it (opening a show and coming back).
 * Changing the URL is what makes the browser ask again.
 *
 * Images are the only resource with this problem. React Query retries its
 * own fetches; the progress WebSocket reconnects with backoff while it has
 * listeners; audio is never restored at launch and is user-initiated. An
 * `<img>` is the one thing that gives up silently and leaves a broken tile.
 */
interface ArtworkEpochState {
  epoch: number;
  markOffline: () => void;
  bump: () => void;
}

/** Epoch while the backend is known to be down. Negative so the positive
 *  counter never collides with it. */
const OFFLINE_EPOCH = -1;

/** A store, not a module variable: the components have to *re-render* for a
 *  new `src` to reach the DOM. Refetching the shows query does not do that
 *  — React Query shares structurally equal results, so a refetch that finds
 *  the same shows hands back the identical object and nothing re-renders. */
const useArtworkEpochStore = create<ArtworkEpochState>((set) => ({
  epoch: 0,
  markOffline: () => set({ epoch: OFFLINE_EPOCH }),
  // Back to a positive counter from either starting state.
  bump: () => set((state) => ({ epoch: Math.max(state.epoch, 0) + 1 })),
}));

/** Called once, from `main.tsx`, before the first render of a launch whose
 *  restored cache carries covers — that first frame lands while the sidecar
 *  is still booting. */
export function markArtworkOffline(): void {
  useArtworkEpochStore.getState().markOffline();
}

/** Called once, from `main.tsx`, when /api/health first succeeds. */
export function bumpArtworkEpoch(): void {
  useArtworkEpochStore.getState().bump();
}

/** Subscribe to the epoch. Call once per component that renders covers and
 *  pass the value to `showArtworkSrc`. */
export function useArtworkEpoch(): number {
  return useArtworkEpochStore((state) => state.epoch);
}

/** Cover src for a show given its `artwork_url` field and folder path. */
export function showArtworkSrc(
  artworkUrlField: string | undefined | null,
  folder: string,
  epoch = 0,
): string {
  if (!artworkUrlField) return DEFAULT_COVER_URL;
  // The backend route would refuse the connection, and the tile would sit
  // broken until the bump. The bundled default cover is on disk and paints
  // at once, so the launch shows a cover-shaped tile that swaps to the real
  // art instead of a broken image that fills in.
  if (epoch === OFFLINE_EPOCH) return DEFAULT_COVER_URL;
  const url = artworkUrl(folder);
  // Omitted while still 0 so a launch that never rendered early covers
  // keeps byte-identical URLs and the browser cache stays warm.
  return epoch ? `${url}&v=${epoch}` : url;
}
