import type { Episode, ShowMeta } from "@/api/types";

/**
 * Whether an episode's audio can be fetched again after it is deleted.
 *
 * Only a live entry of the show's feed can: a video of a YouTube show, or an
 * RSS item that carries an enclosure URL. A local episode (a folder show, the
 * Files bucket, or a file dropped into a feed show's folder, which the server
 * lists with `in_feed` false) has no source to fetch from, so its audio file
 * is the only copy, and an episode gone from the feed cannot be re-downloaded
 * either. YouTube rows carry no `audio_url` (the video id is the source),
 * which is why this cannot be a plain `audio_url` check.
 *
 * Returns false while *meta* is loading: the answer gates destructive copy and
 * download actions, so "unknown" must read as "not re-downloadable".
 */
export function canRedownloadAudio(ep: Episode, meta: ShowMeta | null | undefined): boolean {
  if (!meta || ep.removed || !ep.in_feed) return false;
  if (meta.youtube_url) return true;
  return !!meta.rss_url && !!ep.audio_url;
}
