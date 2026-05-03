"""
podcodex.mcp.server — MCP server exposing PodCodex retrieval.

Surfaces the LanceDB index as tools for Claude Desktop / Claude Code (and
any other MCP-capable client). The server performs retrieval only — the
client LLM synthesises answers from the returned chunks, so no API key is
required on the server side.

Tools:
    - ``list_shows``     — catalog of indexed shows.
    - ``list_episodes``  — per-episode metadata with date/title filters.
    - ``get_episode``    — metadata card for a single episode.
    - ``search``         — hybrid semantic + FTS search.
    - ``exact``          — literal phrase match, returns every hit.
    - ``get_context``    — expand a hit with its neighboring chunks.
    - ``speaker_stats``  — aggregate chunk counts / airtime per speaker.

Environment:
    PODCODEX_INDEX — path to the LanceDB directory. Default is
                     ``<data_dir>/index`` (alongside models and logs).
                     Shared with the desktop app and Discord bot.
"""

from __future__ import annotations

from loguru import logger
from mcp.server.fastmcp import FastMCP

from podcodex.core._utils import episode_display, merge_display_turns
from podcodex.rag.defaults import (
    ALPHA,
    CONTEXT_WINDOW,
    DEFAULT_CHUNKING,
    DEFAULT_MODEL,
    TOP_K,
)
from podcodex.rag.index_store import get_index_store
from podcodex.rag.retriever import get_retriever, merge_results
from podcodex.rag.store import collection_name

mcp = FastMCP("podcodex")
# Internal HTTP route is "/" so when the sub-app is mounted at /mcp by
# podcodex.api.app, the full path stays /mcp (rather than /mcp/mcp).
mcp.settings.streamable_http_path = "/"


def _default_collections_meta() -> list[tuple[str, dict]]:
    """(name, meta) pairs for collections indexed under the default model+chunker."""
    info = get_index_store().get_all_collection_info()
    return [
        (name, meta)
        for name, meta in info.items()
        if meta.get("model") == DEFAULT_MODEL
        and meta.get("chunker") == DEFAULT_CHUNKING
    ]


def _resolve_collections(show: str | None) -> list[str]:
    """Collection names for default model+chunker, optionally filtered by show (case-insensitive)."""
    target = (show or "").lower().strip()
    return [
        name
        for name, meta in _default_collections_meta()
        if not target or meta.get("show", "").lower() == target
    ]


# Cache for ``list_shows`` date ranges. Keyed by (collection_name,
# index_mtime) so a reindex invalidates it automatically. Keeps the
# per-collection ``list_episodes_filtered`` scan out of the hot path of
# repeat ``list_shows`` calls (Claude Desktop / browser tabs re-invoke
# this on every discovery refresh). Capped: every write prunes stale
# mtime entries for the same collection so a long-running process
# doesn't accumulate tombstones.
_SHOW_DATE_CACHE: dict[tuple[str, float], tuple[str, str] | None] = {}


def _put_show_date_cache(key: tuple[str, float], value: tuple[str, str] | None) -> None:
    """Store a date range, dropping older mtime entries for the same collection."""
    collection, _ = key
    stale = [k for k in _SHOW_DATE_CACHE if k[0] == collection and k != key]
    for k in stale:
        _SHOW_DATE_CACHE.pop(k, None)
    _SHOW_DATE_CACHE[key] = value


def _trim(chunk: dict) -> dict:
    """Compact chunk shape sent to MCP clients.

    ``episode_title`` is the human-readable label to cite (RSS title if
    the episode has one, otherwise a humanised form of the stem).
    ``episode`` remains the raw identifier needed for later ``get_context``
    lookups. ``pub_date`` carries the RSS publication date so clients can
    answer date-scoped questions (``"épisodes de février 2026"``) without
    an extra ``list_episodes`` round-trip.

    When the chunk carries per-turn ``speakers`` (semantic chunking on a
    diarised transcript), emit the turn list and omit the redundant
    flat ``text`` blob — the LLM can cite the right speaker per quote
    instead of guessing from the chunk-level ``dominant_speaker``.
    Consecutive same-speaker turns are merged for compactness.
    """
    out: dict = {
        "show": chunk.get("show", ""),
        "episode": chunk.get("episode", ""),
        "episode_title": episode_display(chunk),
        "chunk_index": int(chunk.get("chunk_index", -1)),
        "start": float(chunk.get("start", 0.0)),
        "end": float(chunk.get("end", 0.0)),
        "speaker": chunk.get("dominant_speaker", ""),
    }
    turns = merge_display_turns(chunk.get("speakers") or [])
    if turns:
        out["speakers"] = [
            {
                "speaker": t.get("speaker", "Unknown"),
                "start": float(t["start"]) if t.get("start") is not None else 0.0,
                "end": float(t["end"]) if t.get("end") is not None else 0.0,
                "text": t.get("text", ""),
            }
            for t in turns
        ]
    else:
        out["text"] = chunk.get("text", "")
    pub_date = chunk.get("pub_date") or ""
    if pub_date:
        out["pub_date"] = pub_date
    ep_num = chunk.get("episode_number")
    if ep_num is not None:
        out["episode_number"] = int(ep_num)
    if "score" in chunk:
        out["score"] = float(chunk["score"])
    for flag in ("accent_match", "fuzzy_match"):
        if chunk.get(flag):
            out[flag] = True
    return out


# ── Tools ─────────────────────────────────────────────────────────────


@mcp.tool()
def list_shows() -> list[dict]:
    """List the podcast shows available in the user's PodCodex index.

    Call this first when the user asks what is indexed, or to discover
    valid ``show`` values for ``search`` / ``exact`` / ``get_context``.
    Only shows built with the default model (bge-m3) and chunker
    (semantic) are returned — other combinations are invisible to the
    other tools.

    Returns:
        A list of ``{"show", "episodes", "first_pub_date", "last_pub_date"}``
        entries (the date fields are omitted when no episode carries a
        publication date). Empty if no qualifying shows are indexed.
    """
    store = get_index_store()
    try:
        mtime = store.index_mtime()
    except Exception:
        mtime = 0.0
    out: list[dict] = []
    for name, meta in _default_collections_meta():
        show_name = meta.get("show", "")
        entry: dict = {"show": show_name, "episodes": store.episode_count(name)}
        key = (name, mtime)
        if key in _SHOW_DATE_CACHE:
            range_ = _SHOW_DATE_CACHE[key]
        else:
            try:
                items = store.list_episodes_filtered(name)
            except Exception:
                items = []
            dates = sorted(i.get("pub_date", "") for i in items if i.get("pub_date"))
            range_ = (dates[0], dates[-1]) if dates else None
            _put_show_date_cache(key, range_)
        if range_ is not None:
            entry["first_pub_date"], entry["last_pub_date"] = range_
        out.append(entry)
    return out


@mcp.tool()
def list_episodes(
    show: str | None = None,
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
    title_contains: str | None = None,
) -> list[dict]:
    """List episodes in the user's PodCodex index with optional filters.

    Use this to browse what's indexed before running a ``search``, or
    to answer questions like "what episodes came out last month?".
    Each record carries enough metadata (title, date, duration,
    speakers, description) to render a browse view without an extra
    ``get_episode`` round-trip per stem.

    Args:
        show: Restrict to this show (case-insensitive). Omit to list
            across every indexed show.
        pub_date_min: Oldest publication date, inclusive (``YYYY-MM-DD``).
        pub_date_max: Newest publication date, inclusive.
        title_contains: Substring match on episode title or stem
            (case-insensitive).

    Returns:
        Per-episode records, each ``{show, episode, episode_title,
        pub_date, episode_number, chunk_count, duration, speakers,
        description}``. ``speakers`` is a sorted list of the
        dominant-speaker values that appear in any chunk of the episode;
        ``description`` is the RSS description truncated at index time.
        Sorted by episode identifier within a show.
    """
    collections = _resolve_collections(show)
    if not collections:
        return []
    store = get_index_store()
    meta_by_col = {name: meta for name, meta in _default_collections_meta()}
    out: list[dict] = []
    for col in collections:
        items = store.list_episodes_filtered(
            col,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
            title_contains=title_contains,
            with_detail=True,
        )
        show_name = meta_by_col.get(col, {}).get("show", "")
        for item in items:
            out.append(
                {
                    "show": show_name,
                    "episode": item["episode"],
                    "episode_title": episode_display(item),
                    "pub_date": item.get("pub_date", ""),
                    "episode_number": item.get("episode_number"),
                    "chunk_count": int(item.get("chunk_count", 0)),
                    "duration": float(item.get("duration", 0.0)),
                    "speakers": list(item.get("speakers") or []),
                    "description": item.get("description", ""),
                }
            )
    return out


@mcp.tool()
def get_episode(show: str, episode: str) -> dict | None:
    """Return metadata for a single episode in the user's PodCodex index.

    Call this when the user asks for the description, pub date,
    duration, speakers, or episode number of a specific episode —
    cheaper and more direct than running a ``search`` for metadata.

    Args:
        show: Show name, as returned by ``list_shows`` or a prior
            result's ``show`` field.
        episode: Episode identifier (stem), e.g. from ``list_episodes``
            or a prior result's ``episode`` field.

    Returns:
        ``{show, episode, episode_title, pub_date, episode_number,
        description, source, chunk_count, duration, speakers}`` — or
        ``None`` if the episode is not indexed.
    """
    col = collection_name(show, DEFAULT_MODEL, DEFAULT_CHUNKING)
    store = get_index_store()
    rec = store.get_episode(col, episode)
    if rec is None:
        return None
    return {
        "show": show,
        "episode": rec["episode"],
        "episode_title": episode_display(rec),
        "pub_date": rec.get("pub_date", ""),
        "episode_number": rec.get("episode_number"),
        "description": rec.get("description", ""),
        "source": rec.get("source", ""),
        "chunk_count": int(rec.get("chunk_count", 0)),
        "duration": float(rec.get("duration", 0.0)),
        "speakers": list(rec.get("speakers", [])),
    }


@mcp.tool()
def search(
    query: str,
    show: str | None = None,
    top_k: int = TOP_K,
    episode: str | None = None,
    episodes: list[str] | None = None,
    speaker: str | None = None,
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
) -> list[dict]:
    """Search the user's PodCodex podcast transcripts by meaning (hybrid dense + BM25).

    **Use only when** the user asks a question about their own podcasts,
    specific shows, episodes, or guests, or explicitly invokes podcodex.
    Do **not** use for general knowledge, coding, or topics unrelated to
    the user's transcripts. If uncertain, ask the user first.

    Choose ``search`` for meaning ("what did they say about X?"). Choose
    ``exact`` instead when the user wants to verify a specific wording.

    **Cite every fact drawn from the results.** Attribute each claim
    inline as ``[Show • Episode @ MM:SS]`` using the ``show``,
    ``episode``, and ``start`` fields of the chunk it came from. When a
    chunk carries a ``speakers`` array, cite the *turn's* speaker and
    ``start`` — not the chunk-level ``speaker`` (that field is only the
    chunk's dominant voice and may not own the quote). Never blend in
    outside information without labeling it ``(outside the transcripts)``.
    Mark inferences or synthesis as ``(inference)``. Users rely on
    podcodex answers to reflect what their transcripts actually say —
    unlabeled outside knowledge breaks that contract.

    Args:
        query: Natural-language query. Works in any language the
            transcripts use.
        show: Restrict to this show (exact, case-insensitive name match).
            Omit to search every indexed show and merge results
            round-robin across shows.
        top_k: Maximum chunks to return (default 5). Raise cautiously —
            more chunks costs more of your context without helping if
            the top results already answer the question.
        episode: Restrict to a single episode identifier (as returned in
            prior results' ``episode`` field). Omit to search all
            episodes.
        episodes: Alternative to ``episode`` — restrict to a list of
            episode identifiers (from prior results or ``list_episodes``).
            Takes precedence over ``episode`` when both are given.
        speaker: Restrict to a single speaker (exact name match on
            ``dominant_speaker``).
        pub_date_min: Oldest publication date to include, inclusive.
            Accepts ``YYYY-MM-DD`` (RFC 2822 / ISO 8601 also parsed).
        pub_date_max: Newest publication date to include, inclusive.

    Returns:
        Ranked chunks, each containing ``show``, ``episode``,
        ``episode_title``, ``chunk_index``, ``start``, ``end``, ``speaker``
        (chunk-level dominant), ``score``, and — when present —
        ``pub_date`` (ISO 8601) and ``episode_number``. Diarised chunks
        carry a ``speakers`` array of ``{speaker, start, end, text}``
        turns (consecutive same-speaker turns merged); cite from these
        for accurate attribution. Untimed or single-speaker chunks
        instead carry a flat ``text`` field. Pass a chunk's
        ``chunk_index`` to ``get_context`` to read its surrounding scene.
    """
    collections = _resolve_collections(show)
    if not collections:
        return []
    ret = get_retriever()
    # Encode the query once across all collections — BGE-M3 encode is the
    # dominant cost per call.
    qv = ret.encode_query(query)
    hits_by_col: dict[str, list[dict]] = {}
    for col in collections:
        try:
            hits = ret.retrieve(
                query,
                col,
                top_k=top_k,
                alpha=ALPHA,
                episode=episode,
                episodes=episodes,
                speaker=speaker,
                pub_date_min=pub_date_min,
                pub_date_max=pub_date_max,
                query_vector=qv,
            )
        except ValueError:
            raise
        except Exception:
            logger.exception(f"search: collection {col!r} failed; skipping")
            continue
        if hits:
            hits_by_col[col] = hits
    merged = merge_results(hits_by_col, top_k=top_k)
    return [_trim(chunk) for chunk, _col in merged]


@mcp.tool()
def exact(
    query: str,
    show: str | None = None,
    episode: str | None = None,
    episodes: list[str] | None = None,
    speaker: str | None = None,
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
) -> list[dict]:
    """Find every literal occurrence of a phrase in the user's PodCodex transcripts.

    Case-insensitive, accent-tolerant, no result cap. Behaves like
    Ctrl+F across every indexed episode.

    **Use only when** the user wants to verify a quote, count mentions
    of specific wording, or explicitly invokes podcodex. Do **not** use
    for general-knowledge lookups or topics unrelated to the user's
    transcripts. Choose ``search`` instead for meaning-based questions.

    **Cite every quoted or referenced passage** inline as
    ``[Show • Episode @ MM:SS]`` using the chunk's ``show``,
    ``episode``, and ``start`` fields. Never supplement with outside
    information unless labeled ``(outside the transcripts)``. Mark
    inferences as ``(inference)``.

    Args:
        query: Phrase to find. Multi-word phrases are matched as a unit,
            tolerant of accent differences and minor typos.
        show: Restrict to this show (exact, case-insensitive name).
            Omit to search every indexed show.
        episode: Restrict to a single episode identifier.
        episodes: Alternative to ``episode`` — restrict to a list of
            episode identifiers.
        speaker: Restrict to a single ``dominant_speaker`` value.
        pub_date_min: Oldest publication date to include, inclusive
            (``YYYY-MM-DD``).
        pub_date_max: Newest publication date to include, inclusive.

    Returns:
        Every matching chunk, ordered by collection then position. Each
        carries the same fields as ``search`` results (``show``,
        ``episode``, ``episode_title``, ``chunk_index``, ``start``,
        ``end``, ``speaker``, plus a per-turn ``speakers`` array on
        diarised chunks or a flat ``text`` on untimed/single-speaker
        chunks, and — when present — ``pub_date`` and ``episode_number``).
        Matches that are not perfect string hits additionally carry
        ``accent_match`` or ``fuzzy_match`` flags. No relevance ranking —
        results are positional.
    """
    collections = _resolve_collections(show)
    if not collections:
        return []
    ret = get_retriever()
    out: list[dict] = []
    for col in collections:
        try:
            matches = ret.exact(
                query,
                col,
                episode=episode,
                episodes=episodes,
                speaker=speaker,
                pub_date_min=pub_date_min,
                pub_date_max=pub_date_max,
            )
        except ValueError:
            raise
        except Exception:
            logger.exception(f"exact: collection {col!r} failed; skipping")
            continue
        for match in matches:
            out.append(_trim(match))
    return out


@mcp.tool()
def get_context(
    show: str,
    episode: str,
    chunk_index: int,
    window: int = CONTEXT_WINDOW,
) -> list[dict]:
    """Expand a PodCodex search hit with its neighboring chunks.

    Call this after ``search`` or ``exact`` when a single chunk doesn't
    carry enough context — to read the full exchange around a quote,
    follow a narrative beat, or recover speaker attribution for a
    pronoun. The returned window is contiguous and chronological.

    **Cite the expanded passage** just like ``search`` / ``exact``
    results: attribute facts inline as ``[Show • Episode @ MM:SS]``.
    Outside information must be labeled ``(outside the transcripts)``;
    inferences must be labeled ``(inference)``.

    Args:
        show: Show name, as returned by ``list_shows`` or in a prior
            result's ``show`` field.
        episode: Episode identifier from a prior result's ``episode``
            field (do not invent values).
        chunk_index: ``chunk_index`` of the hit to expand (from a prior
            result).
        window: Chunks to include on each side of the center (default 3,
            so 7 total). Raise to widen the scene; setting 0 returns
            only the center chunk.

    Returns:
        Chunks covering ``[chunk_index - window, chunk_index + window]``
        inclusive, sorted by position. Each chunk carries the same fields
        as ``search`` results (``show``, ``episode``, ``episode_title``,
        ``chunk_index``, ``start``, ``end``, ``speaker``, plus a per-turn
        ``speakers`` array on diarised chunks or a flat ``text`` on
        untimed/single-speaker chunks, and — when present — ``pub_date``
        and ``episode_number``). Empty list if the episode or
        ``chunk_index`` is not found.
    """
    col = collection_name(show, DEFAULT_MODEL, DEFAULT_CHUNKING)
    chunks = get_index_store().get_chunk_window(
        col, episode, chunk_index, window=window
    )
    return [_trim(c) for c in chunks]


@mcp.tool()
def speaker_stats(show: str | None = None) -> list[dict]:
    """Aggregate per-speaker airtime across the user's PodCodex index.

    Use this when the user asks who speaks most, wants a speaker ranking,
    or needs a rough breakdown of participation — instead of triangulating
    with several ``search`` calls.

    Counts every chunk whose ``dominant_speaker`` is the speaker in
    question. ``total_duration`` sums ``end - start`` over those chunks,
    which approximates airtime (seconds). Chunks with no identified
    speaker are skipped.

    Args:
        show: Restrict to this show (case-insensitive match). Omit to
            aggregate across every indexed show.

    Returns:
        Sorted descending by ``chunk_count``. Each record:
        ``{speaker, chunk_count, total_duration, episodes}``. Combine
        across collections when multiple shows match.
    """
    collections = _resolve_collections(show)
    if not collections:
        return []
    return get_index_store().speaker_stats_multi(collections)


def main() -> None:
    """Run the MCP server over stdio.

    Used both by the dev ``podcodex-mcp`` console-script and by the
    bundled binary's ``--mcp`` flag. The bundled path goes through
    ``podcodex.api.server._handle_mcp_flag`` which has already wired
    caches and called ``bootstrap_for_mcp_stdio``; the dev path has
    not, so bootstrap here is idempotent for the bundled case (loguru
    sinks are reset before re-adding) and necessary for the dev case.
    """
    from podcodex.bootstrap import bootstrap_for_mcp_stdio

    bootstrap_for_mcp_stdio()
    logger.info("podcodex-mcp starting (stdio)")
    # Embedder loads lazily on first tool call — Claude Desktop's
    # initialize handshake has a 60s timeout that we'd otherwise blow on
    # cold-starts of the PyInstaller --onefile bundle.
    mcp.run()


# Register user-defined prompts at module import so both stdio and HTTP
# transports expose the same slash-menu entries to Claude Desktop.
from podcodex.mcp.prompts import live_reload_lifespan, reregister_all  # noqa: E402

reregister_all(mcp)

# Live-reload: the stdio subprocess Claude Desktop spawns watches the
# prompts JSON file and pushes `prompts/list_changed` when the desktop
# app's Settings UI edits it. Users see prompt changes without quitting
# Claude Desktop. Code changes still need a full restart.
mcp.settings.lifespan = live_reload_lifespan


if __name__ == "__main__":
    main()
