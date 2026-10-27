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
/** What cover URLs are built from: the app-wide epoch, plus a per-show
 *  revision bumped when that show's cover is replaced (same route, same
 *  marker, so the URL would not change otherwise). One object, replaced on
 *  every change, so a subscriber re-renders and only the changed show's
 *  `src` differs. */
export interface ArtworkRevision {
  epoch: number;
  perShow: Readonly<Record<string, number>>;
}

interface ArtworkEpochState {
  rev: ArtworkRevision;
  markOffline: () => void;
  bump: () => void;
  bumpShow: (folder: string) => void;
}

/** Epoch while the backend is known to be down. Negative so the positive
 *  counter never collides with it. */
const OFFLINE_EPOCH = -1;

/** A store, not a module variable: the components have to *re-render* for a
 *  new `src` to reach the DOM. Refetching the shows query does not do that
 *  — React Query shares structurally equal results, so a refetch that finds
 *  the same shows hands back the identical object and nothing re-renders. */
const useArtworkEpochStore = create<ArtworkEpochState>((set) => ({
  rev: { epoch: 0, perShow: {} },
  markOffline: () => set((s) => ({ rev: { ...s.rev, epoch: OFFLINE_EPOCH } })),
  // Back to a positive counter from either starting state.
  bump: () => set((s) => ({ rev: { ...s.rev, epoch: Math.max(s.rev.epoch, 0) + 1 } })),
  bumpShow: (folder) =>
    set((s) => ({
      rev: { ...s.rev, perShow: { ...s.rev.perShow, [folder]: (s.rev.perShow[folder] ?? 0) + 1 } },
    })),
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

/** After a show's cover is replaced: the new image is served at the URL the
 *  old one had, so only this show's `src` gets a new revision. */
export function bumpShowArtwork(folder: string): void {
  useArtworkEpochStore.getState().bumpShow(folder);
}

/** Subscribe to the epoch. Call once per component that renders covers and
 *  pass the value to `showArtworkSrc`. */
export function useArtworkEpoch(): ArtworkRevision {
  return useArtworkEpochStore((state) => state.rev);
}

/** Short stable key for a cover source (FNV-1a), to tell sources apart in
 *  a URL without putting a feed URL in it. */
function sourceKey(value: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < value.length; i++) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(36);
}

/** Cover src for a show given its `artwork_url` field and folder path. */
export function showArtworkSrc(
  artworkUrlField: string | undefined | null,
  folder: string,
  rev: ArtworkRevision = { epoch: 0, perShow: {} },
): string {
  const { epoch } = rev;
  if (!artworkUrlField) return DEFAULT_COVER_URL;
  // The backend route would refuse the connection, and the tile would sit
  // broken until the bump. The bundled default cover is on disk and paints
  // at once, so the launch shows a cover-shaped tile that swaps to the real
  // art instead of a broken image that fills in.
  if (epoch === OFFLINE_EPOCH) return DEFAULT_COVER_URL;
  const url = artworkUrl(folder);
  // `v` is omitted while the epoch is still 0, so a launch that never
  // rendered early covers keeps the same URLs and the cache stays warm.
  // The page keeps images by URL and the route's URL is fixed per show, so
  // the URL has to follow the cover itself: `s` changes with the source
  // (a feed URL, "local" after an upload, back to the feed after a reset),
  // `r` with each upload over an upload, which keeps the same source. The
  // route's `no-cache` header covers the disk cache across restarts.
  const showRev = rev.perShow[folder];
  return url
    + (epoch ? `&v=${epoch}` : "")
    + `&s=${sourceKey(artworkUrlField)}`
    + (showRev ? `&r=${showRev}` : "");
}
