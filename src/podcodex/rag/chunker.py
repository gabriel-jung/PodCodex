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

from loguru import logger

from podcodex.rag.defaults import CHUNKER_MODEL


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

_SEP = " "


def _meta_fields(transcript: dict) -> tuple[str, str, str, dict]:
    """Extract (show, episode, source, extras) from transcript metadata.

    ``extras`` contains optional display fields (episode_title, pub_date,
    episode_number) when available — only non-empty values are included.
    """
    meta = transcript.get("meta", {})
    episode = meta.get("episode", "")
    extras: dict = {}
    rss_title = meta.get("rss_title", "")
    if rss_title and rss_title != episode:
        extras["episode_title"] = rss_title
    if meta.get("rss_pub_date"):
        extras["pub_date"] = meta["rss_pub_date"]
    if meta.get("episode_number") is not None:
        extras["episode_number"] = meta["episode_number"]
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


def _map_offsets_to_metadata(
    chunk_start: int, chunk_end: int, offset_map: list[dict]
) -> dict | None:
    """Map chunk character offsets back to timing and speaker metadata.

    Args:
        chunk_start: Start character index of the chunk in the full text.
        chunk_end: End character index of the chunk in the full text.
        offset_map: Per-turn offset entries produced by ``_build_episode_text``.

    Returns:
        A dict with ``{start, end, dominant_speaker, speakers}`` if the chunk
        overlaps at least one turn, or ``None`` otherwise.
    """
    overlapping = [
        t
        for t in offset_map
        if t["end_char"] > chunk_start and t["start_char"] < chunk_end
    ]
    if not overlapping:
        return None

    speaker_chars: dict[str, int] = {}
    for t in overlapping:
        chars = max(
            0, min(t["end_char"], chunk_end) - max(t["start_char"], chunk_start)
        )
        spk = t.get("speaker") or "UNKNOWN"
        speaker_chars[spk] = speaker_chars.get(spk, 0) + chars

    return {
        "start": overlapping[0]["start"],
        "end": overlapping[-1]["end"],
        "dominant_speaker": max(speaker_chars, key=speaker_chars.__getitem__),
        "speakers": overlapping,
    }


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

    chunker = SemanticChunker(
        embedding_model=CHUNKER_MODEL,
        chunk_size=chunk_size,
        threshold=threshold,
        skip_window=1,
    )
    raw_chunks = chunker.chunk(full_text)

    chunks = []
    for raw in raw_chunks:
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
