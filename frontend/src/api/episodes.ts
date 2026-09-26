import type { EpisodeListItem, EpisodeMeta } from "./generated-types";
import { json } from "./client";

export interface ListEpisodesParams {
  show: string;
  model?: string;
  chunking?: string;
  pub_date_min?: string | null;
  pub_date_max?: string | null;
  title_contains?: string | null;
}

// The show label rides in the query string, never the path: a label with a
// "/" in it (feed titles) split the path and hit the wrong route.
export const listIndexedEpisodes = (p: ListEpisodesParams) => {
  const qs = new URLSearchParams({ show: p.show });
  if (p.model) qs.set("model", p.model);
  if (p.chunking) qs.set("chunking", p.chunking);
  if (p.pub_date_min) qs.set("pub_date_min", p.pub_date_min);
  if (p.pub_date_max) qs.set("pub_date_max", p.pub_date_max);
  if (p.title_contains) qs.set("title_contains", p.title_contains);
  return json<EpisodeListItem[]>(`/api/episodes/list?${qs}`);
};

export const getIndexedEpisode = (
  show: string,
  stem: string,
  opts: { model?: string; chunking?: string } = {},
) => {
  const qs = new URLSearchParams({ show, episode_stem: stem });
  if (opts.model) qs.set("model", opts.model);
  if (opts.chunking) qs.set("chunking", opts.chunking);
  return json<EpisodeMeta>(`/api/episodes/one?${qs}`);
};
