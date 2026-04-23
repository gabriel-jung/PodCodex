"""
podcodex.rag.chunker — Chunking strategies for transcript segments.

Two strategies:
    speaker_chunks  — one chunk per (filtered) speaker turn. Fast, no extra deps.
    semantic_chunks — full-episode text chunked by semantic similarity via
                      Chonkie SemanticChunker, then mapped back to timing metadata.

Both accept a transcript dict (``{meta: {show, episode, ...}, segments: [...]}``)
and return a list of chunk dicts ready for embedding and storage.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import NamedTuple

from loguru import logger

from podcodex.ingest.rss import clean_description
from podcodex.rag.index_store import _normalize_pub_date
from podcodex.rag.defaults import CHUNKER_MODEL


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

_SEP = " "
_SENTENCE_TERMINATORS = (".", "!", "?", "…")
_MIN_PUNCTUATION_DENSITY = 0.15  # avg terminators per turn


def _meta_fields(transcript: dict) -> tuple[str, str, str, dict]:
    """Extract (show, episode, source, extras) from transcript metadata.

    ``extras`` carries optional display fields (episode_title, pub_date,
    episode_number, description). RSS description is HTML-stripped and
    truncated so bot commands can surface it without an extra lookup.
    """
    meta = transcript.get("meta", {})
    episode = meta.get("episode", "")
    extras: dict = {}
    rss_title = meta.get("rss_title", "")
    if rss_title and rss_title != episode:
        extras["episode_title"] = rss_title
    pub_date = _normalize_pub_date(meta.get("rss_pub_date"))
    if pub_date:
        extras["pub_date"] = pub_date
    if meta.get("episode_number") is not None:
        extras["episode_number"] = meta["episode_number"]
    rss_description = meta.get("rss_description", "")
    if rss_description:
        cleaned = clean_description(rss_description)
        if cleaned:
            extras["description"] = cleaned
    if not meta.get("timed", True):
        extras["timed"] = False
    return meta.get("show", ""), episode, meta.get("source", ""), extras


def _build_episode_text(segments: list[dict]) -> tuple[str, list[dict]]:
    """Concatenate turns into a single string, tracking character offsets per turn.

    Segments are assumed to be pre-filtered (all must have ``text``).

    Args:
        segments: List of segment dicts, each with at least a ``text`` key.

    Returns:
        A ``(full_text, offset_map)`` tuple where *offset_map* entries are dicts
        with keys ``{start_char, end_char, speaker, start, end, text}``.
    """
    offset_map = []
    parts = []
    cursor = 0
    for seg in segments:
        text = seg["text"]
        offset_map.append(
            {
                "start_char": cursor,
                "end_char": cursor + len(text),
                "speaker": seg.get("speaker") or "UNKNOWN",
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": text,
            }
        )
        parts.append(text)
        cursor += len(text) + len(_SEP)
    return _SEP.join(parts), offset_map


def _padded_for_chunker(
    full_text: str, offset_map: list[dict]
) -> tuple[str, list[int]]:
    """Produce a copy of ``full_text`` with ``.`` inserted after every turn
    that doesn't already end with a sentence terminator.

    Chonkie's SemanticChunker splits on sentence boundaries — a transcript
    without punctuation collapses to a single "sentence" and returns one
    mega-chunk regardless of ``chunk_size``. Padding gives the splitter
    something to work with, but only for chunking: offsets returned by
    Chonkie are translated back to the unpadded ``full_text`` via
    ``_unpadded_pos`` so stored chunk text and metadata mapping stay clean.

    Returns:
        ``(padded_text, inserts)`` where *inserts* is a sorted list of
        positions in ``padded_text`` at which a synthetic ``.`` was added.
    """
    pieces: list[str] = []
    inserts: list[int] = []
    padded_cursor = 0
    for i, entry in enumerate(offset_map):
        turn_text = full_text[entry["start_char"] : entry["end_char"]]
        pieces.append(turn_text)
        padded_cursor += len(turn_text)
        if not turn_text.rstrip().endswith(_SENTENCE_TERMINATORS):
            inserts.append(padded_cursor)
            pieces.append(".")
            padded_cursor += 1
        if i < len(offset_map) - 1:
            pieces.append(_SEP)
            padded_cursor += len(_SEP)
    return "".join(pieces), inserts


def _unpadded_pos(padded_pos: int, inserts: list[int]) -> int:
    """Translate a position in the padded text back to the unpadded text."""
    return padded_pos - bisect_right(inserts, padded_pos - 1)


def _looks_punctuated(segments: list[dict]) -> bool:
    """Heuristic: decide whether a transcript already carries punctuation.

    Counts sentence terminators anywhere in the text (not only at turn
    ends — a single long turn may hold several sentences) and divides by
    the number of turns. Above ``_MIN_PUNCTUATION_DENSITY`` the transcript
    is considered already punctuated and the caller should skip the
    synthetic-period workaround so the original text is handed to
    Chonkie untouched.
    """
    if not segments:
        return False
    terminator_count = sum(
        sum(seg["text"].count(t) for t in _SENTENCE_TERMINATORS) for seg in segments
    )
    return terminator_count / len(segments) >= _MIN_PUNCTUATION_DENSITY


def _map_offsets_to_metadata(
    chunk_start: int, chunk_end: int, offset_map: list[dict]
) -> dict | None:
    """Map chunk character offsets back to timing and speaker metadata.

    Every overlapping turn is returned in ``speakers``, but each turn's
    ``text`` is clipped to the chunk's char bounds so adjacent chunks
    don't duplicate the boundary turn's content. The clip ensures the
    displayed text stays in sync with the chunk's literal slice, so an
    exact-match hit in a boundary turn is visible in the result card.

    Args:
        chunk_start: Start character index of the chunk in the full text.
        chunk_end: End character index of the chunk in the full text.
        offset_map: Per-turn offset entries produced by ``_build_episode_text``.

    Returns:
        A dict with ``{start, end, dominant_speaker, speakers}`` if any
        turn overlaps the chunk, or ``None`` otherwise.
    """
    overlapping = [
        t
        for t in offset_map
        if t["end_char"] > chunk_start and t["start_char"] < chunk_end
    ]
    if not overlapping:
        return None

    speakers = []
    speaker_chars: dict[str, int] = {}
    for t in overlapping:
        clip_start = max(t["start_char"], chunk_start) - t["start_char"]
        clip_end = min(t["end_char"], chunk_end) - t["start_char"]
        clipped = t["text"][clip_start:clip_end].strip()
        spk = t.get("speaker") or "UNKNOWN"
        if clipped:
            speakers.append({**t, "text": clipped})
            speaker_chars[spk] = speaker_chars.get(spk, 0) + len(clipped)

    if not speakers:
        return None

    return {
        "start": overlapping[0]["start"],
        "end": overlapping[-1]["end"],
        "dominant_speaker": max(speaker_chars, key=speaker_chars.__getitem__),
        "speakers": speakers,
    }


class _SplitChunk(NamedTuple):
    """Stand-in for Chonkie chunks; mirrors the attrs consumed downstream."""

    text: str
    start_index: int
    end_index: int
    token_count: int


def _split_oversized(raw_chunks, full_text: str, max_tokens: int, target_tokens: int):
    """Yield chunks from ``raw_chunks``, subdividing any that exceed ``max_tokens``.

    Splits are made at whitespace boundaries closest to proportional cut
    points in the chunk's character span, so each sub-chunk stays near
    ``target_tokens`` without cutting mid-word. Character offsets into the
    original ``full_text`` are preserved so offset-to-metadata mapping still
    recovers accurate speaker/timing info for each sub-chunk.
    """
    for raw in raw_chunks:
        if raw.token_count <= max_tokens:
            yield raw
            continue

        n_parts = max(2, -(-raw.token_count // target_tokens))  # ceil div
        span_start = raw.start_index
        span_end = raw.end_index
        span_len = span_end - span_start
        if span_len <= 0:
            yield raw
            continue

        # Find whitespace break near each proportional cut point.
        cuts: list[int] = [span_start]
        for i in range(1, n_parts):
            target = span_start + (span_len * i) // n_parts
            # Nearest whitespace at-or-before target; fall back to at-or-after.
            j = full_text.rfind(" ", span_start, target)
            if j <= cuts[-1]:
                j = full_text.find(" ", target, span_end)
            if j < 0 or j <= cuts[-1]:
                continue
            cuts.append(j + 1)
        cuts.append(span_end)

        approx_tokens = max(1, raw.token_count // (len(cuts) - 1))
        for k in range(len(cuts) - 1):
            s, e = cuts[k], cuts[k + 1]
            if e <= s:
                continue
            piece = full_text[s:e].strip()
            if not piece:
                continue
            yield _SplitChunk(
                text=piece,
                start_index=s,
                end_index=e,
                token_count=approx_tokens,
            )


# ──────────────────────────────────────────────
# Strategy 1 — Speaker chunks
# ──────────────────────────────────────────────


def speaker_chunks(transcript: dict, min_chars: int = 30) -> list[dict]:
    """
    Return one chunk per speaker turn, filtering out noise.

    Segments are already speaker-merged by the pipeline, so this only
    applies noise filtering (min_chars) and attaches show/episode metadata.

    Args:
        transcript : transcript dict with ``meta`` and ``segments`` keys
        min_chars  : minimum character count to keep a turn (default 30)

    Returns:
        List of {show, episode, source, start, end, speaker, text}
    """
    show, episode, source, extras = _meta_fields(transcript)
    chunks = []
    for seg in transcript.get("segments", []):
        text = seg.get("text", "").strip()
        if len(text) < min_chars:
            continue
        chunk = {
            "show": show,
            "episode": episode,
            "source": source,
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "speaker": seg.get("speaker", "UNKNOWN"),
            "text": text,
            **extras,
        }
        chunks.append(chunk)
    logger.info(f"speaker_chunks: {len(chunks)} chunks from '{episode}'")
    return chunks


# ──────────────────────────────────────────────
# Strategy 2 — Semantic chunks (Chonkie)
# ──────────────────────────────────────────────


def semantic_chunks(
    transcript: dict,
    chunk_size: int = 256,
    threshold: float = 0.5,
    min_chars: int = 30,
    pad_unpunctuated: bool = True,
) -> list[dict]:
    """
    Chunk a transcript using semantic similarity (Chonkie SemanticChunker).

    The full episode text is concatenated and chunked freely across speaker
    boundaries, then mapped back to timing and speaker metadata. Each chunk
    may span multiple speakers; the dominant speaker (by character coverage)
    is reported alongside the full list of overlapping turns.

    Args:
        transcript : transcript dict with ``meta`` and ``segments`` keys
        chunk_size : max tokens per chunk (default 256)
        threshold  : semantic similarity threshold for splitting (default 0.5)
        min_chars  : minimum chars to keep a segment before chunking (default 30)
        pad_unpunctuated : when True (default), auto-detect whether the
            transcript already carries punctuation and skip padding when
            it does. Unpunctuated transcripts (YouTube auto-captions) get
            a synthetic ``.`` appended to each turn lacking a sentence
            terminator, purely as input to Chonkie — Chonkie's sentence
            splitter needs punctuation to produce multiple sentences; on
            an unpunctuated transcript the whole episode would collapse
            to one mega-chunk. Storage, metadata, and display are
            unaffected: stored text is sliced from the original unpadded
            string. Set False to disable entirely.

    Returns:
        List of chunk dicts with keys:
            show, episode, source, start, end, text, token_count,
            dominant_speaker — speaker with most characters in the chunk,
            speakers — list of overlapping turns, each a dict with
                {start_char, end_char, speaker, start, end, text}.
    """
    from chonkie import SemanticChunker

    show, episode, source, extras = _meta_fields(transcript)

    segments = [
        {**seg, "text": text}
        for seg in transcript.get("segments", [])
        if len(text := seg.get("text", "").strip()) >= min_chars
    ]
    if not segments:
        logger.warning(f"semantic_chunks: no segments after filtering for '{episode}'")
        return []

    full_text, offset_map = _build_episode_text(segments)
    apply_padding = pad_unpunctuated and not _looks_punctuated(segments)
    if apply_padding:
        padded_text, inserts = _padded_for_chunker(full_text, offset_map)
    else:
        padded_text, inserts = full_text, []

    chunker = SemanticChunker(
        embedding_model=CHUNKER_MODEL,
        chunk_size=chunk_size,
        threshold=threshold,
        skip_window=1,
    )
    raw_chunks = chunker.chunk(padded_text)

    # Translate to unpadded coords so storage/display stay faithful to the
    # source and ``_map_offsets_to_metadata`` sees its expected coord space.
    unpadded_chunks = []
    for raw in raw_chunks:
        u_start = _unpadded_pos(raw.start_index, inserts)
        u_end = _unpadded_pos(raw.end_index, inserts)
        if u_end <= u_start:
            continue
        unpadded_chunks.append(
            _SplitChunk(
                text=full_text[u_start:u_end],
                start_index=u_start,
                end_index=u_end,
                token_count=raw.token_count,
            )
        )

    # Hard cap above Chonkie's soft target — keeps a single semantically
    # coherent span from producing one mega-chunk that swallows an episode.
    max_tokens = chunk_size * 3
    split_chunks = list(
        _split_oversized(unpadded_chunks, full_text, max_tokens, chunk_size)
    )

    chunks = []
    for raw in split_chunks:
        meta = _map_offsets_to_metadata(raw.start_index, raw.end_index, offset_map)
        if meta is None:
            continue
        chunks.append(
            {
                "show": show,
                "episode": episode,
                "source": source,
                "start": meta["start"],
                "end": meta["end"],
                "dominant_speaker": meta["dominant_speaker"],
                "speakers": meta["speakers"],
                "text": raw.text,
                "token_count": raw.token_count,
                **extras,
            }
        )

    logger.info(
        f"semantic_chunks: {len(chunks)} chunks from '{episode}' "
        f"(chunk_size={chunk_size}, threshold={threshold})"
    )
    return chunks
