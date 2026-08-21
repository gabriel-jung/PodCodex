"""podcodex.rag.search_service: one search facility for API, bot, and MCP.

Every surface resolves shows to collections and fans queries across them
through this module. Surfaces keep transport, access control, and response
shaping; this module owns collection picking, per-model query encoding,
cross-collection merging, and result ordering.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from podcodex.core.app_config import load_config
from podcodex.ingest.show import load_show_meta
from podcodex.rag.defaults import ALPHA, DEFAULT_CHUNKING, DEFAULT_MODEL
from podcodex.rag.hit import Hit
from podcodex.rag.retriever import Retriever, get_retriever, merge_results

_PREFS_TTL = 5.0
_prefs_cache: tuple[float, dict[str, tuple[str, str]]] | None = None


@dataclass(frozen=True)
class SearchCollection:
    """One queryable collection: the picked representative of a show."""

    name: str
    model: str
    show: str
    artwork_url: str = ""


def resolve_collections(
    col_info: dict[str, dict],
    *,
    shows: Iterable[str] | None = None,
    show_prefs: dict[str, tuple[str, str]] | None = None,
    override: tuple[str, str] | None = None,
    default: tuple[str, str] = (DEFAULT_MODEL, DEFAULT_CHUNKING),
) -> list[SearchCollection]:
    """Pick one collection per show from ``get_all_collection_info()`` output.

    Precedence per show, each step skipped when no collection matches:
    ``override`` (explicit user request), the show's ``show_prefs`` entry,
    ``default``, the global ``DEFAULT_MODEL``/``DEFAULT_CHUNKING`` combo,
    then the first collection by sorted name so a show indexed only under a
    non-default model stays reachable.

    ``shows`` filters by display name, case-insensitive, because that is all
    a Discord user or an MCP client ever types. Grouping and ``show_prefs``
    key on the show id when the collection carries one, so a renamed show
    keeps its preference; collections written before ids existed fall back to
    the lowercased display name.

    Result is sorted by lowercased show name.
    """
    wanted = {s.strip().lower() for s in shows} if shows is not None else None
    by_show: dict[str, list[tuple[str, dict]]] = {}
    for name in sorted(col_info):
        meta = col_info[name]
        show = meta.get("show") or name
        if wanted is not None and show.lower() not in wanted:
            continue
        key = (meta.get("show_id") or "").strip() or show.lower()
        by_show.setdefault(key, []).append((name, meta))

    def _match(cands: list[tuple[str, dict]], combo: tuple[str, str] | None):
        if combo is None:
            return None
        model, chunker = combo
        for name, meta in cands:
            if meta.get("model") == model and meta.get("chunker") == chunker:
                return name, meta
        return None

    out: list[SearchCollection] = []
    prefs = show_prefs or {}
    for key, cands in by_show.items():
        # Keyed by id after the migration. Before it, `key` already *is* the
        # lowercased label, so no second lookup is needed.
        pref = prefs.get(key)
        if pref is None and (cands[0][1].get("show_id") or "").strip():
            # The row has an id but the show may not have been minted one in
            # show.toml yet, in which case prefs are still label-keyed.
            pref = prefs.get((cands[0][1].get("show") or "").strip().lower())
        picked = (
            _match(cands, override)
            or _match(cands, pref)
            or _match(cands, default)
            # Distinct global tier, not a dedup of `default`: a caller may
            # pass a non-default `default` (e.g. guild settings) and its miss
            # must still fall to the true global defaults before name order.
            or _match(cands, (DEFAULT_MODEL, DEFAULT_CHUNKING))
            or cands[0]
        )
        name, meta = picked
        out.append(
            SearchCollection(
                name=name,
                model=meta.get("model", ""),
                show=meta.get("show") or name,
                artwork_url=(meta.get("artwork_url") or "").strip(),
            )
        )
    out.sort(key=lambda c: c.show.lower())
    return out


def load_show_rag_prefs() -> dict[str, tuple[str, str]]:
    """Per-show RAG prefs from show.toml: ``{show_id: (model, chunker)}``.

    Shows that have not been minted an id yet are keyed by lowercased display
    name instead, which is what ``resolve_collections`` falls back to.

    Only shows that set at least one of rag_model/rag_chunker appear;
    the blank half is filled with the global default. Unreadable folders and
    config errors degrade to fewer or no prefs, never an exception: the
    caller falls through to defaults, same contract as the MCP original.

    Cached for ``_PREFS_TTL`` seconds: this walks every show folder and
    reads its show.toml, and is called on every query on every surface
    (API, bot, MCP), including the bot's event loop. A show.toml pref edit
    still appears within a few seconds.
    """
    global _prefs_cache
    now = time.monotonic()
    if _prefs_cache is not None and now - _prefs_cache[0] < _PREFS_TTL:
        return dict(_prefs_cache[1])

    prefs: dict[str, tuple[str, str]] = {}
    try:
        folders = load_config().show_folders
    except Exception:
        logger.warning("search_service: could not load config for RAG prefs")
        return prefs
    for folder in folders:
        try:
            p = Path(folder)
            if not p.is_dir():
                continue
            meta = load_show_meta(p)
        except Exception:
            continue
        if meta is None:
            continue
        model = (meta.pipeline.rag_model or "").strip()
        chunker = (meta.pipeline.rag_chunker or "").strip()
        if not model and not chunker:
            continue
        key = meta.id or (meta.name or p.name).strip().lower()
        prefs[key] = (
            model or DEFAULT_MODEL,
            chunker or DEFAULT_CHUNKING,
        )
    _prefs_cache = (now, dict(prefs))
    return prefs


def _by_model(
    collections: Sequence[SearchCollection],
) -> dict[str, list[SearchCollection]]:
    grouped: dict[str, list[SearchCollection]] = {}
    for col in collections:
        grouped.setdefault(col.model, []).append(col)
    return grouped


def hybrid_search(
    query: str,
    collections: Sequence[SearchCollection],
    *,
    top_k: int,
    alpha: float = ALPHA,
    strategy: str = "roundrobin",
    score_floor: float = 0.0,
    episode: str | None = None,
    episodes: list[str] | None = None,
    source: str | None = None,
    speaker: str | None = None,
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
    retriever_factory: Callable[[str], Retriever] = get_retriever,
) -> list[tuple[Hit, str]]:
    """Fan a semantic query across collections and merge.

    Groups by embedding model so the query is encoded once per model, not
    once per collection. ``ValueError`` (bad filter, dim mismatch)
    re-raises; any other per-collection failure is logged and skipped so one
    broken table never blanks the whole answer.
    """
    hits_by_col: dict[str, list[Hit]] = {}
    for model, cols in _by_model(collections).items():
        ret = retriever_factory(model)
        qv = ret.encode_query(query)
        for col in cols:
            try:
                hits_by_col[col.name] = ret.retrieve(
                    query,
                    col.name,
                    top_k=top_k,
                    alpha=alpha,
                    episode=episode,
                    episodes=episodes,
                    source=source,
                    speaker=speaker,
                    pub_date_min=pub_date_min,
                    pub_date_max=pub_date_max,
                    query_vector=qv,
                )
            except ValueError:
                raise
            except Exception:
                logger.exception(f"search_service: retrieve failed for {col.name}")
    merged = merge_results(hits_by_col, top_k=top_k, strategy=strategy)
    if score_floor > 0.0:
        merged = [r for r in merged if (r[0].score or 0.0) > score_floor]
    return merged


def exact_search(
    query: str,
    collections: Sequence[SearchCollection],
    *,
    order: str = "positional",
    episode: str | None = None,
    episodes: list[str] | None = None,
    source: str | None = None,
    speaker: str | None = None,
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
    retriever_factory: Callable[[str], Retriever] = get_retriever,
) -> list[tuple[Hit, str]]:
    """Literal search across collections.

    ``order="positional"`` keeps each collection's retriever order (MCP,
    API). ``order="chronological"`` reproduces the bot's reading order:
    phrase hits (exact + accent) sorted by (score desc, episode, start), so
    whole-word matches (1.0) precede superstring matches (0.99) precede
    accent variants, each group chronological; fuzzy hits appended by score
    descending.
    """
    out: list[tuple[Hit, str]] = []
    for col in collections:
        ret = retriever_factory(col.model)
        try:
            hits = ret.exact(
                query,
                col.name,
                episode=episode,
                episodes=episodes,
                source=source,
                speaker=speaker,
                pub_date_min=pub_date_min,
                pub_date_max=pub_date_max,
            )
        except ValueError:
            raise
        except Exception:
            logger.exception(f"search_service: exact failed for {col.name}")
            continue
        out.extend((hit, col.name) for hit in hits)
    if order == "chronological":
        # `score if score is not None`, not `score or ...`: a legitimate 0.0
        # must sort as 0.0, not jump to the top as if the field were missing.
        phrase = sorted(
            (r for r in out if not r[0].fuzzy_match),
            key=lambda x: (
                -(x[0].score if x[0].score is not None else 1.0),
                x[0].episode,
                x[0].start,
            ),
        )
        fuzzy = sorted(
            (r for r in out if r[0].fuzzy_match),
            key=lambda x: -(x[0].score if x[0].score is not None else 0.6),
        )
        out = phrase + fuzzy
    return out


def random_quote(
    collections: Sequence[SearchCollection],
    *,
    episode: str | None = None,
    episodes: list[str] | None = None,
    source: str | None = None,
    speaker: str | None = None,
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
    retriever_factory: Callable[[str], Retriever] = get_retriever,
) -> tuple[Hit, str] | None:
    """One random quote from a randomly picked collection, or None.

    ``ValueError`` from the retriever (bad filter) re-raises; any other
    failure is logged and the function returns None.
    """
    if not collections:
        return None
    col = random.choice(list(collections))
    try:
        chunk = retriever_factory(col.model).random(
            col.name,
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
    except ValueError:
        raise
    except Exception:
        logger.exception(f"search_service: random failed for {col.name}")
        return None
    if chunk is None:
        return None
    return chunk, col.name
