"""
podcodex.mcp.server — MCP server exposing PodCodex retrieval.

Surfaces the LanceDB index as tools for Claude Desktop / Claude Code (and
any other MCP-capable client). The server performs retrieval only — the
client LLM synthesises answers from the returned chunks, so no API key is
required on the server side.

Tools:
    - ``list_shows``    — catalog of indexed shows.
    - ``search``        — hybrid semantic + FTS search.
    - ``exact``         — literal phrase match, returns every hit.
    - ``get_context``   — expand a hit with its neighboring chunks.
    - ``speaker_stats`` — aggregate chunk counts / airtime per speaker.

Environment:
    PODCODEX_INDEX — path to the LanceDB directory (defaults to
                     ``~/.local/share/podcodex/index``). Shared with the
                     desktop app and Discord bot.
"""

from __future__ import annotations

from loguru import logger
from mcp.server.fastmcp import FastMCP

from podcodex.core._utils import episode_display
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


def _trim(chunk: dict) -> dict:
    """Compact chunk shape sent to MCP clients.

    ``episode_title`` is the human-readable label to cite (RSS title if
    the episode has one, otherwise a humanised form of the stem).
    ``episode`` remains the raw identifier needed for later ``get_context``
    lookups.
    """
    out = {
        "show": chunk.get("show", ""),
        "episode": chunk.get("episode", ""),
        "episode_title": episode_display(chunk),
        "chunk_index": int(chunk.get("chunk_index", -1)),
        "start": float(chunk.get("start", 0.0)),
        "end": float(chunk.get("end", 0.0)),
        "speaker": chunk.get("dominant_speaker", ""),
        "text": chunk.get("text", ""),
    }
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
        A list of ``{"show": str, "episodes": int}`` entries. Empty if
        no qualifying shows are indexed.
    """
    store = get_index_store()
    return [
        {"show": meta.get("show", ""), "episodes": store.episode_count(name)}
        for name, meta in _default_collections_meta()
    ]


@mcp.tool()
def search(
    query: str,
    show: str | None = None,
    top_k: int = TOP_K,
    episode: str | None = None,
    speaker: str | None = None,
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
    ``episode``, and ``start`` fields of the chunk it came from. Never
    blend in outside information without labeling it ``(outside the
    transcripts)``. Mark inferences or synthesis as ``(inference)``.
    Users rely on podcodex answers to reflect what their transcripts
    actually say — unlabeled outside knowledge breaks that contract.

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
        speaker: Restrict to a single speaker (exact name match on
            ``dominant_speaker``).

    Returns:
        Ranked chunks, each containing ``show``, ``episode``,
        ``chunk_index``, ``start``, ``end``, ``speaker``, ``text``, and
        ``score``. Pass a chunk's ``chunk_index`` to ``get_context`` to
        read its surrounding scene.
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
                speaker=speaker,
                query_vector=qv,
            )
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
    speaker: str | None = None,
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
        speaker: Restrict to a single ``dominant_speaker`` value.

    Returns:
        Every matching chunk, ordered by collection then position.
        Matches that are not perfect string hits carry ``accent_match``
        or ``fuzzy_match`` fields. No relevance ranking — results are
        positional.
    """
    collections = _resolve_collections(show)
    if not collections:
        return []
    ret = get_retriever()
    out: list[dict] = []
    for col in collections:
        try:
            matches = ret.find(query, col, episode=episode, speaker=speaker)
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
        inclusive, sorted by position. Empty list if the episode or
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
    """Run the MCP server over stdio."""
    logger.info("podcodex-mcp starting (stdio)")
    # Force embedder load now so a broken install / missing weights surface at
    # startup rather than on the first tool call.
    get_retriever().embedder
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
