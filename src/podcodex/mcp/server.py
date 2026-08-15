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
    - ``exact_count``    — literal phrase counts per episode, no text.
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

from podcodex.core._utils import (
    episode_display,
    format_hms,
    is_unattributed,
    merge_display_turns,
)
from podcodex.rag.defaults import ALPHA, CONTEXT_WINDOW, TOP_K
from podcodex.rag.hit import Hit
from podcodex.rag.index_store import chunk_map_from_chunks, get_index_store
from podcodex.rag.retriever import get_retriever
from podcodex.rag.search_service import (
    SearchCollection,
    exact_search as _svc_exact,
    hybrid_search as _svc_hybrid,
    load_show_rag_prefs,
    resolve_collections,
)

mcp = FastMCP("podcodex")
# Internal HTTP route is "/" so when the sub-app is mounted at /mcp by
# podcodex.api.app, the full path stays /mcp (rather than /mcp/mcp).
mcp.settings.streamable_http_path = "/"


def _episode_transcript_text(chunks: list[Hit]) -> str:
    """Flatten preloaded episode chunks to ``[MmSS] Speaker: text`` lines.

    Diarised chunks emit one line per merged turn; untimed/single-speaker
    chunks emit one line each.
    """
    lines: list[str] = []
    for c in chunks:
        turns = merge_display_turns(c.speakers or [])
        if turns:
            for t in turns:
                ts = format_hms(float(t.get("start") or c.start))
                sp = t.get("speaker", "")
                lines.append(
                    f"[{ts}] {sp}: {t.get('text', '')}"
                    if not is_unattributed(sp)
                    else f"[{ts}] {t.get('text', '')}"
                )
        else:
            ts = format_hms(c.start)
            sp = c.dominant_speaker
            lines.append(
                f"[{ts}] {sp}: {c.text}"
                if not is_unattributed(sp)
                else f"[{ts}] {c.text}"
            )
    return "\n".join(lines)


def _resolve_collections(show: str | None = None) -> list[SearchCollection]:
    """Collections the MCP tools should query, one per show.

    Each show resolves to the single collection named by its ``show.toml``
    RAG preference (falling back to the default model+chunker). Optionally
    filtered to a single show (case-insensitive name match). Thin wrapper
    over the shared search_service resolver.
    """
    return resolve_collections(
        get_index_store().get_all_collection_info(),
        shows=[show] if show else None,
        show_prefs=load_show_rag_prefs(),
    )


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


def _trim(chunk: Hit) -> dict:
    """Compact chunk shape sent to MCP clients.

    ``episode_title`` is the human-readable label to cite (RSS title if
    the episode has one, otherwise a humanised form of the stem).
    ``episode`` remains the raw identifier needed for later ``get_context``
    lookups. ``pub_date`` carries the RSS publication date so clients can
    answer date-scoped questions (``"épisodes de février 2026"``) without
    an extra ``list_episodes`` round-trip. ``start_hms`` is the citation-ready
    form of ``start`` (``"9m38"`` / ``"1h09m46"``) so clients never re-derive
    it and get the ``HhMMmSS`` boundary wrong.

    When the chunk carries per-turn ``speakers`` (semantic chunking on a
    diarised transcript), emit the turn list and omit the redundant
    flat ``text`` blob — the LLM can cite the right speaker per quote
    instead of guessing from the chunk-level ``dominant_speaker``.
    Consecutive same-speaker turns are merged for compactness.
    """
    out: dict = {
        "show": chunk.show,
        "episode": chunk.episode,
        "episode_title": chunk.display_title,
        "chunk_index": chunk.chunk_index,
        "start": chunk.start,
        "start_hms": format_hms(chunk.start),
        "end": chunk.end,
        "speaker": chunk.dominant_speaker,
    }
    turns = merge_display_turns(chunk.speakers or [])
    if turns:
        out["speakers"] = [
            {
                "speaker": t.get("speaker", "Unknown"),
                "start": float(t["start"]) if t.get("start") is not None else 0.0,
                "start_hms": format_hms(
                    float(t["start"]) if t.get("start") is not None else 0.0
                ),
                "end": float(t["end"]) if t.get("end") is not None else 0.0,
                "text": t.get("text", ""),
            }
            for t in turns
        ]
    else:
        out["text"] = chunk.text
    # effective_pub_date tolerates legacy indexes that only carry the raw
    # rss_pub_date form (previously rendered as no date at all).
    pub_date = chunk.effective_pub_date
    if pub_date:
        out["pub_date"] = pub_date
    if chunk.episode_number is not None:
        out["episode_number"] = chunk.episode_number
    if chunk.score is not None:
        out["score"] = chunk.score
    if chunk.accent_match:
        out["accent_match"] = True
    if chunk.fuzzy_match:
        out["fuzzy_match"] = True
    return out


# ── Tools ─────────────────────────────────────────────────────────────


@mcp.tool()
def list_shows() -> list[dict]:
    """List the podcast shows available in the user's PodCodex index.

    Call this first when the user asks what is indexed, or to discover
    valid ``show`` values for ``search`` / ``exact`` / ``get_context``.
    Each show resolves to one collection: the embedding model and chunker
    set in its PodCodex settings, or the default (bge-m3 + semantic) when
    unset, so every indexed show is reachable.

    Returns:
        A list of ``{"show", "episodes", "first_pub_date", "last_pub_date"}``
        entries (the date fields are omitted when no episode carries a
        publication date). Empty if no shows are indexed.
    """
    store = get_index_store()
    try:
        mtime = store.index_mtime()
    except Exception:
        mtime = 0.0
    out: list[dict] = []
    for c in _resolve_collections(None):
        entry: dict = {"show": c.show, "episodes": store.episode_count(c.name)}
        key = (c.name, mtime)
        if key in _SHOW_DATE_CACHE:
            range_ = _SHOW_DATE_CACHE[key]
        else:
            try:
                items = store.list_episodes_filtered(c.name)
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
    broadcast_number: int | None = None,
    fields: list[str] | None = None,
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
        broadcast_number: Restrict to the episode whose broadcast (airing)
            number matches. Only populated for shows that set a
            ``broadcast_number_pattern`` in their settings; absent
            otherwise. This is the airing number in the title, distinct
            from the per-season ``episode_number``.
        fields: Restrict each record to these keys (e.g.
            ``["episode", "chunk_count", "duration"]``) to skip the heavy
            ``description``. Omit for the full record.

    Returns:
        Per-episode records, each ``{show, episode, episode_title,
        pub_date, episode_number, broadcast_number, chunk_count, duration,
        speakers, description}`` (subset when ``fields`` is given).
        ``speakers`` is a sorted list of the dominant-speaker values that
        appear in any chunk of the episode; ``description`` is the RSS
        description truncated at index time. ``episode_number`` is the
        per-season index (resets each season); ``broadcast_number`` is the
        airing number. Sorted by ``pub_date`` (then stem).
    """
    collections = _resolve_collections(show)
    if not collections:
        return []
    store = get_index_store()
    out: list[dict] = []
    for c in collections:
        items = store.list_episodes_filtered(
            c.name,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
            title_contains=title_contains,
            with_detail=True,
        )
        for item in items:
            if (
                broadcast_number is not None
                and item.get("broadcast_number") != broadcast_number
            ):
                continue
            out.append(
                {
                    "show": c.show,
                    "episode": item["episode"],
                    "episode_title": episode_display(item),
                    "pub_date": item.get("pub_date", ""),
                    "episode_number": item.get("episode_number"),
                    "broadcast_number": item.get("broadcast_number"),
                    "chunk_count": int(item.get("chunk_count", 0)),
                    "duration": float(item.get("duration", 0.0)),
                    "speakers": list(item.get("speakers") or []),
                    "description": item.get("description", ""),
                }
            )
    out.sort(key=lambda r: (r.get("pub_date") or "", r.get("episode", "")))
    if fields:
        allowed = set(fields)
        out = [{k: v for k, v in r.items() if k in allowed} for r in out]
    return out


@mcp.tool()
def get_episode(
    show: str,
    episode: str,
    include_chunk_map: bool = False,
    text_preview: int = 0,
    include_transcript: str = "",
) -> dict | None:
    """Return metadata for a single episode in the user's PodCodex index.

    Call this when the user asks for the description, pub date,
    duration, speakers, or episode number of a specific episode —
    cheaper and more direct than running a ``search`` for metadata.

    For whole-episode reads, this tool also serves the chunk-map (a light
    ``chunk_index -> start`` table, no transcript text) and the raw
    transcript, so a single call can back both timestamp verification and
    a linear read.

    Args:
        show: Show name, as returned by ``list_shows`` or a prior
            result's ``show`` field.
        episode: Episode identifier (stem), e.g. from ``list_episodes``
            or a prior result's ``episode`` field.
        include_chunk_map: When true, add ``chunk_map``: a list of
            ``{chunk_index, start, end, start_hms, speakers}`` for every
            chunk, ordered by position. Few KB even for a long episode.
        text_preview: With ``include_chunk_map``, add a ``text_preview``
            of the first N characters to each chunk-map entry (0 = off).
        include_transcript: ``"text"`` adds a ``transcript`` field of
            ``[MmSS] Speaker: line`` rows (more compact than per-turn
            JSON). Empty (default) omits it.

    Returns:
        ``{show, episode, episode_title, pub_date, episode_number,
        description, source, chunk_count, duration, speakers}``, plus
        ``chunk_map`` and/or ``transcript`` when requested — or ``None``
        if the episode is not indexed.
    """
    collections = _resolve_collections(show)
    if not collections:
        return None
    store = get_index_store()
    rec = store.get_episode(collections[0].name, episode)
    if rec is None:
        return None
    card: dict = {
        "show": show,
        "episode": rec["episode"],
        "episode_title": episode_display(rec),
        "pub_date": rec.get("pub_date", ""),
        "episode_number": rec.get("episode_number"),
        "description": rec.get("description", ""),
        "source": rec.get("source", ""),
        "chunk_count": int(rec.get("chunk_count", 0)),
        "duration": float(rec.get("duration", 0.0)),
        "speakers": rec.get("speakers", []),
    }
    want_transcript = include_transcript == "text"
    if include_chunk_map or want_transcript:
        # One episode load feeds both the chunk-map and the transcript.
        chunks = store.load_chunks_no_embeddings(collections[0].name, episode)
        if include_chunk_map:
            card["chunk_map"] = chunk_map_from_chunks(chunks, text_preview)
        if want_transcript:
            card["transcript"] = _episode_transcript_text(chunks)
    return card


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
    inline as ``[Show • Episode @ <start_hms>]`` using the ``show``,
    ``episode``, and ``start_hms`` fields of the chunk it came from (the
    citation-ready ``9m38`` / ``1h09m46`` form). When a chunk carries a
    ``speakers`` array, cite the *turn's* speaker and ``start_hms`` — not
    the chunk-level ``speaker`` (that field is only the
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
    cols = _resolve_collections(show)
    if not cols:
        return []
    merged = _svc_hybrid(
        query,
        cols,
        top_k=top_k,
        alpha=ALPHA,
        episode=episode,
        episodes=episodes,
        speaker=speaker,
        pub_date_min=pub_date_min,
        pub_date_max=pub_date_max,
    )
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

    **Use only when** the user wants to verify a quote or read the wording
    around a mention, or explicitly invokes podcodex. To *count* mentions
    across episodes without pulling text, use ``exact_count`` instead. Do
    **not** use for general-knowledge lookups or topics unrelated to the
    user's transcripts. Choose ``search`` for meaning-based questions.

    **Cite every quoted or referenced passage** inline as
    ``[Show • Episode @ <start_hms>]`` using the chunk's ``show``,
    ``episode``, and ``start_hms`` fields. Never supplement with outside
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
        speaker: Restrict to a single speaker (exact name match on
            ``dominant_speaker``).
        pub_date_min: Oldest publication date to include, inclusive
            (``YYYY-MM-DD``).
        pub_date_max: Newest publication date to include, inclusive.

    Returns:
        Every matching chunk, ordered by collection then position. Each
        carries the same fields as ``search`` results (``show``,
        ``episode``, ``episode_title``, ``chunk_index``, ``start``,
        ``start_hms``, ``end``, ``speaker``, plus a per-turn ``speakers``
        array on diarised chunks or a flat ``text`` on untimed/single-speaker
        chunks, and — when present — ``pub_date`` and ``episode_number``).
        Matches that are not perfect string hits additionally carry
        ``accent_match`` or ``fuzzy_match`` flags. No relevance ranking;
        results are positional.
    """
    cols = _resolve_collections(show)
    if not cols:
        return []
    return [
        _trim(m)
        for m, _col in _svc_exact(
            query,
            cols,
            order="positional",
            episode=episode,
            episodes=episodes,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
    ]


@mcp.tool()
def exact_count(
    queries: list[str],
    show: str | None = None,
    episode: str | None = None,
    episodes: list[str] | None = None,
    speaker: str | None = None,
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
    group_by: str = "episode",
    first_hit: bool = False,
) -> dict:
    """Count literal phrase occurrences per episode, without returning any text.

    The counting counterpart of ``exact``: for each phrase in ``queries``,
    report how many times it occurs and where, grouped by episode (or show).
    Built for recurrence and cross-contamination checks over many entities at
    once, where the chunk text would be pure overhead. Fuzzy near-typo matches
    are excluded (accent variants are kept) so counts stay trustworthy.

    **Use only when** the user (or a workflow) needs occurrence counts of
    specific wording, or explicitly invokes podcodex. For the actual passages,
    use ``exact``; for meaning-based questions, ``search``.

    Args:
        queries: Phrases to count. Result keys are these strings verbatim.
        show: Restrict to this show (exact, case-insensitive name). Omit to
            count across every indexed show.
        episode: Restrict to a single episode identifier.
        episodes: Alternative to ``episode`` — restrict to a list.
        speaker: Restrict to a single ``dominant_speaker`` value.
        pub_date_min: Oldest publication date to include, inclusive.
        pub_date_max: Newest publication date to include, inclusive.
        group_by: ``"episode"`` (default) or ``"show"``.
        first_hit: Also return the earliest hit per group as
            ``{"chunk_index", "start", "start_hms"}``.

    Returns:
        ``{phrase: {group_key: count}}``, or with ``first_hit``
        ``{phrase: {group_key: {"count", "first"}}}``. Groups with zero hits
        are omitted, so ``phrase in result and group in result[phrase]`` is a
        valid "entity present in this episode" check. Cost is ``O(len(queries))``
        full scans per collection, so a large batch is not free.
    """
    valid = [q for q in (queries or []) if q]
    if not valid:
        raise ValueError("exact_count: provide a non-empty `queries` list")
    collections = _resolve_collections(show)
    if not collections:
        return {}
    merged: dict[str, dict] = {q: {} for q in valid}
    for c in collections:
        try:
            counts = get_retriever(c.model).exact_counts(
                valid,
                c.name,
                group_by=group_by,
                first_hit=first_hit,
                episode=episode,
                episodes=episodes,
                speaker=speaker,
                pub_date_min=pub_date_min,
                pub_date_max=pub_date_max,
            )
        except ValueError:
            raise
        except Exception:
            logger.exception(f"exact_count: collection {c.name!r} failed; skipping")
            continue
        for q, groups in counts.items():
            merged[q].update(groups)
    return merged


@mcp.tool()
def get_context(
    show: str,
    episode: str,
    chunk_index: int | None = None,
    window: int = CONTEXT_WINDOW,
    at_time: str | int | float | None = None,
) -> list[dict]:
    """Expand a PodCodex search hit with its neighboring chunks.

    Call this after ``search`` or ``exact`` when a single chunk doesn't
    carry enough context — to read the full exchange around a quote,
    follow a narrative beat, or recover speaker attribution for a
    pronoun. The returned window is contiguous and chronological.

    **Cite the expanded passage** just like ``search`` / ``exact``
    results: attribute facts inline as ``[Show • Episode @ <start_hms>]``.
    Outside information must be labeled ``(outside the transcripts)``;
    inferences must be labeled ``(inference)``.

    Args:
        show: Show name, as returned by ``list_shows`` or in a prior
            result's ``show`` field.
        episode: Episode identifier from a prior result's ``episode``
            field (do not invent values).
        chunk_index: ``chunk_index`` of the hit to expand (from a prior
            result). Alternative to ``at_time``; if both are given,
            ``chunk_index`` wins. Provide exactly one.
        window: Chunks to include on each side of the center (default 3,
            so 7 total). Raise to widen the scene; setting 0 returns
            only the center chunk.
        at_time: Center the window on the chunk covering this timestamp
            instead of a ``chunk_index``. Accepts seconds (``4186``) or
            the clock forms ``"1h09m46"`` / ``"69m46"``. A time in a gap
            or past the transcript snaps to the nearest chunk.

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
    if chunk_index is None and at_time is None:
        raise ValueError("get_context: provide `chunk_index` or `at_time`")
    collections = _resolve_collections(show)
    if not collections:
        return []
    resolved_time: float | None = None
    if chunk_index is None:
        from podcodex.core._utils import parse_time

        resolved_time = parse_time(at_time)
    chunks = get_index_store().get_chunk_window(
        collections[0].name,
        episode,
        chunk_index=chunk_index,
        window=window,
        at_time=resolved_time,
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
    return get_index_store().speaker_stats_multi([c.name for c in collections])


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
