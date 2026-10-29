"""
podcodex.rag.indexing — Transcript vectorization pipeline.

Chunks a transcript, embeds each chunk with the chosen model, and writes the
result into the global :class:`IndexStore` LanceDB index.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from loguru import logger

from podcodex.rag.chunker import semantic_chunks, speaker_chunks
from podcodex.rag.defaults import CHUNK_SIZE, CHUNK_THRESHOLD, MODELS
from podcodex.rag.embedder import get_embedder
from podcodex.rag.index_store import IndexStore


class NoChunksError(ValueError):
    """The transcript chunked to nothing for this strategy (not a failure)."""


class IndexingError(RuntimeError):
    """One or more (model, chunker) combinations failed to index.

    Raised after every other combination has been written, so a failing
    embedder does not also cost the ones that work.
    """


# ──────────────────────────────────────────────
# Chunking + embedding
# ──────────────────────────────────────────────


def _chunk_transcript(
    transcript: dict,
    chunking: str,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = CHUNK_THRESHOLD,
) -> list[dict]:
    """Chunk a transcript using the given strategy. Returns empty list on failure."""
    if chunking == "speaker":
        if not transcript.get("meta", {}).get("diarized", True):
            logger.warning(
                "Speaker chunking is not useful without diarization — "
                "falling back to semantic."
            )
            return semantic_chunks(
                transcript, chunk_size=chunk_size, threshold=threshold
            )
        return speaker_chunks(transcript)
    else:
        return semantic_chunks(transcript, chunk_size=chunk_size, threshold=threshold)


def vectorize_episode(
    transcript: dict,
    show: str,
    episode: str,
    model_key: str,
    chunking: str,
    local: IndexStore,
    *,
    show_id: str = "",
    chunks: list[dict] | None = None,
    chunk_size: int = CHUNK_SIZE,
    threshold: float = CHUNK_THRESHOLD,
    overwrite: bool = False,
    device: str = "cpu",
) -> tuple[list[dict], int]:
    """
    Vectorize a single (model, chunker) combination into the IndexStore.

    Args:
        transcript : parsed transcript dict (with meta.show / meta.episode set)
        show, episode : display label and episode identifier
        show_id : the show's stable id; the key the collection is registered
            under. Empty only for callers that have no show folder to read.
        model_key, chunking : which model and chunker to use
        local : IndexStore instance (LanceDB)
        chunks : pre-computed chunks for this chunking strategy (avoids re-chunking)
        chunk_size, threshold : semantic chunking params
        overwrite : delete and recreate if already indexed

    Returns:
        ``(chunks, n_embedded)`` — the chunks list (for reuse across models)
        and the count of chunks upserted (0 if skipped/cached).

    Raises:
        ValueError: if no chunks could be produced.
    """
    dim = MODELS[model_key].dim
    # Resolve-or-create: a show indexed before ids existed keeps writing into
    # the collection it already has, instead of growing a second one beside it.
    col = local.ensure_collection_for_show(show_id, show, model_key, chunking, dim)

    if local.episode_is_indexed(col, episode) and not overwrite:
        new_source = transcript.get("meta", {}).get("source", "")
        new_hash = transcript.get("meta", {}).get("source_hash", "")
        new_ref = transcript.get("meta", {}).get("source_ref", "")
        stored = local.load_chunks_no_embeddings(col, episode)
        stored_source = stored[0].source if stored else ""
        stored_hash = stored[0].source_hash if stored else ""
        stored_ref = stored[0].source_ref if stored else ""

        if stored_source and new_source and stored_source != new_source:
            logger.info(
                f"[UPGRADE] '{episode}' source changed: "
                f"{stored_source} → {new_source} ({col})"
            )
            # save_chunks replaces this episode's rows, stale ones included.
        elif new_hash and (
            new_hash != stored_hash or (new_ref and new_ref != stored_ref)
        ):
            # An edit of the same step, or chunks from before the hash was
            # recorded: re-cut them, or the index keeps serving the old text.
            # Same content chosen another way (a pin now followed, or the
            # reverse) re-cuts too: the record decides staleness later.
            logger.info(f"[UPGRADE] '{episode}' source changed ({col})")
        else:
            local_count = local.episode_chunk_count(col, episode)
            logger.info(f"[SKIP] '{episode}' cached ({col}, {local_count} chunks)")
            # Stored hits may be reused as write-path chunks for the next
            # model; convert back to the chunker dict shape.
            return chunks or [c.to_chunk_dict() for c in stored], 0

    if chunks is None:
        n_segs = len(transcript.get("segments", []))
        logger.info(
            f"Chunking '{episode}' with strategy '{chunking}' "
            f"({n_segs} segments, chunk_size={chunk_size}, threshold={threshold})"
        )
        t0 = time.perf_counter()
        chunks = _chunk_transcript(transcript, chunking, chunk_size, threshold)
        logger.info(
            f"Chunked '{episode}': {len(chunks)} chunks in {time.perf_counter() - t0:.1f}s"
        )
    if not chunks:
        raise NoChunksError(f"No chunks produced for strategy '{chunking}'")

    embedder = get_embedder(model_key, device=device)
    logger.info(
        f"Embedding {len(chunks)} chunks for '{episode}' "
        f"with '{model_key}' on {device} → '{col}'"
    )
    t0 = time.perf_counter()
    embeddings = embedder.encode_passages(chunks)
    logger.info(
        f"Embedded {len(chunks)} chunks for '{episode}' "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    _check_embeddings(embeddings, len(chunks), dim, model_key)
    _warn_truncated(chunks, model_key, episode)

    logger.info(f"Upserting {len(chunks)} chunks into '{col}'")
    t0 = time.perf_counter()
    local.save_chunks(col, episode, chunks, embeddings)
    logger.success(
        f"Vectorized {len(chunks)} chunks into '{col}' "
        f"(upsert {time.perf_counter() - t0:.1f}s)"
    )
    return chunks, len(chunks)


# Tokens an embedder reads per passage: BGE-M3 is called with max_length=512,
# and the E5 models cap at 512 positions. Anything past this is not embedded.
_EMBED_WINDOW = 512


def _check_embeddings(embeddings, n: int, dim: int, model_key: str) -> None:
    """Refuse model output that cannot be a valid embedding of *n* chunks.

    A NaN vector scores 0 in cosine search and a zero vector matches nothing,
    so either would be saved as a chunk dense search can never return, with
    no sign anything went wrong. Raised as a combination failure instead.
    """
    shape = getattr(embeddings, "shape", None)
    if shape != (n, dim):
        raise ValueError(
            f"{model_key} returned embeddings of shape {shape}, expected {(n, dim)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{model_key} returned non-finite embeddings")
    if not np.any(embeddings, axis=1).all():
        raise ValueError(f"{model_key} returned an all-zero embedding")


def _warn_truncated(chunks: list[dict], model_key: str, episode: str) -> None:
    """Say when chunks run past the embedder's window (only their head is
    embedded): speaker chunks have no length cap, so a long monologue loses
    most of its text from dense search. Keyword search still sees it all."""
    if model_key.startswith("pplx"):
        return  # long-context model

    def tokens(c: dict) -> int:
        # Speaker chunks carry no count: estimate (subword tokenizers average
        # well over one token per word, so this errs low).
        return c.get("token_count") or int(len(c.get("text", "").split()) * 1.3)

    over = [t for c in chunks if (t := tokens(c)) > _EMBED_WINDOW]
    if over:
        longest = max(over)
        logger.warning(
            f"{len(over)} of {len(chunks)} chunks of '{episode}' exceed the "
            f"{_EMBED_WINDOW}-token window of {model_key} (longest {longest}): "
            "dense search sees only their beginning"
        )


def _drop_episode(
    local: IndexStore,
    show_id: str,
    show: str,
    episode: str,
    model_keys: list[str],
    chunking: str,
) -> None:
    """Remove *episode* from each (model, *chunking*) collection of the show."""
    for model_key in model_keys:
        col = local.resolve_collection(show_id, model_key, chunking, show_label=show)
        if col and local.episode_is_indexed(col, episode):
            local.delete_episode(col, episode)
            logger.info(f"Removed '{episode}' from '{col}': it now yields no chunks")


def vectorize_batch(
    transcript: dict,
    show: str,
    episode: str,
    model_keys: list[str],
    chunkings: list[str],
    local: IndexStore,
    *,
    show_id: str = "",
    chunk_size: int = CHUNK_SIZE,
    threshold: float = CHUNK_THRESHOLD,
    overwrite: bool = False,
    device: str = "cpu",
    on_progress: Callable[[int, int, str], None] | None = None,
) -> int:
    """
    Vectorize all (model, chunker) combinations into the IndexStore.

    Chunks once per strategy, then embeds with each model.

    Args:
        on_progress : callback ``(step, total, label)`` for UI progress updates.

    Returns:
        Total number of chunks present in the index across all combinations,
        counting both newly upserted and cache-hit chunks. 0 when the
        transcript produced no chunks.

    Raises:
        IndexingError: when any combination failed. The others are still
            written first; the episode must not be marked indexed.
    """
    total = len(model_keys) * len(chunkings)
    step = 0
    total_processed = 0
    failures: list[str] = []

    for chunking in chunkings:
        chunks_for_strategy: list[dict] | None = None

        for model_key in model_keys:
            label = f"{MODELS[model_key].label} / {chunking}"
            if on_progress:
                on_progress(step, total, label)

            try:
                chunks_for_strategy, n = vectorize_episode(
                    transcript,
                    show,
                    episode,
                    model_key,
                    chunking,
                    local,
                    show_id=show_id,
                    chunks=chunks_for_strategy,
                    chunk_size=chunk_size,
                    threshold=threshold,
                    overwrite=overwrite,
                    device=device,
                )
                # Count cache hits (n == 0 with chunks present) as processed,
                # so index_job can distinguish "all cached" from "real 0".
                if n > 0:
                    total_processed += n
                elif chunks_for_strategy:
                    total_processed += len(chunks_for_strategy)
            except NoChunksError as e:
                logger.warning(str(e))
                if overwrite:
                    # Overwrite means "these combinations now hold what the
                    # transcript yields": nothing, for this strategy. Left in
                    # place, the old chunks would answer searches forever
                    # (podcodex-reindex prunes by episode, not by strategy).
                    _drop_episode(local, show_id, show, episode, model_keys, chunking)
                step += len(model_keys) - model_keys.index(model_key)
                break
            except Exception as exc:
                logger.exception(f"Failed for {label}")
                failures.append(f"{label}: {exc}")

            step += 1

    if failures:
        raise IndexingError(
            f"Indexing failed for {len(failures)} of {total} combination(s): "
            + "; ".join(failures)
        )
    return total_processed
