import type { EpisodeListItem, EpisodeMeta } from "./generated-types";
import { json } from "./client";

export interface ListEpisodesParams {
  showId: string;
  model?: string;
  chunking?: string;
  pub_date_min?: string | null;
  pub_date_max?: string | null;
  title_contains?: string | null;
}

// Keyed by the show id, never the display name: two shows may share a name.
export const listIndexedEpisodes = (p: ListEpisodesParams) => {
  const qs = new URLSearchParams({ show_id: p.showId });
  if (p.model) qs.set("model", p.model);
  if (p.chunking) qs.set("chunking", p.chunking);
  if (p.pub_date_min) qs.set("pub_date_min", p.pub_date_min);
  if (p.pub_date_max) qs.set("pub_date_max", p.pub_date_max);
  if (p.title_contains) qs.set("title_contains", p.title_contains);
  return json<EpisodeListItem[]>(`/api/episodes/list?${qs}`);
};

export const getIndexedEpisode = (
  showId: string,
  stem: string,
  opts: { model?: string; chunking?: string } = {},
) => {
  const qs = new URLSearchParams({ show_id: showId, episode_stem: stem });
  if (opts.model) qs.set("model", opts.model);
  if (opts.chunking) qs.set("chunking", opts.chunking);
  return json<EpisodeMeta>(`/api/episodes/one?${qs}`);
};
