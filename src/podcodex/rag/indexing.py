"""
podcodex.rag.indexing — Transcript vectorization pipeline.

Chunks a transcript, embeds each chunk with the chosen model, and writes the
result into the global :class:`IndexStore` LanceDB index.
"""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger

from podcodex.rag.chunker import semantic_chunks, speaker_chunks
from podcodex.rag.defaults import CHUNK_SIZE, CHUNK_THRESHOLD, MODELS
from podcodex.rag.embedder import get_embedder
from podcodex.rag.index_store import IndexStore
from podcodex.rag.store import collection_name


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
        show, episode : identifiers
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
    col = collection_name(show, model_key, chunking)
    dim = MODELS[model_key].dim
    local.ensure_collection(col, show=show, model=model_key, chunker=chunking, dim=dim)

    if local.episode_is_indexed(col, episode) and not overwrite:
        new_source = transcript.get("meta", {}).get("source", "")
        stored = local.load_chunks_no_embeddings(col, episode)
        stored_source = stored[0].get("source", "") if stored else ""

        if stored_source and new_source and stored_source != new_source:
            logger.info(
                f"[UPGRADE] '{episode}' source changed: "
                f"{stored_source} → {new_source} ({col})"
            )
            local.delete_episode(col, episode)
        else:
            local_count = local.episode_chunk_count(col, episode)
            logger.info(f"[SKIP] '{episode}' cached ({col}, {local_count} chunks)")
            return chunks or stored, 0

    if overwrite and local.episode_is_indexed(col, episode):
        local.delete_episode(col, episode)

    if chunks is None:
        chunks = _chunk_transcript(transcript, chunking, chunk_size, threshold)
    if not chunks:
        raise ValueError(f"No chunks produced for strategy '{chunking}'")

    embedder = get_embedder(model_key, device=device)
    embeddings = embedder.encode_passages(chunks)
    local.save_chunks(col, episode, chunks, embeddings)

    logger.success(f"Vectorized {len(chunks)} chunks into '{col}'")
    return chunks, len(chunks)


def vectorize_batch(
    transcript: dict,
    show: str,
    episode: str,
    model_keys: list[str],
    chunkings: list[str],
    local: IndexStore,
    *,
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
        Total number of chunks upserted across all combinations.
    """
    total = len(model_keys) * len(chunkings)
    step = 0
    total_upserted = 0

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
                    chunks=chunks_for_strategy,
                    chunk_size=chunk_size,
                    threshold=threshold,
                    overwrite=overwrite,
                    device=device,
                )
                total_upserted += n
            except ValueError as e:
                logger.warning(str(e))
                step += len(model_keys) - model_keys.index(model_key)
                break
            except Exception:
                logger.exception(f"Failed for {label}")

            step += 1

    return total_upserted
