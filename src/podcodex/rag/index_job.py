"""Subprocess entry point for the RAG index pipeline step.

Invoked by ``podcodex.api.subprocess_runner`` as a top-level target so it
can be pickled + launched via ``multiprocessing.spawn``. Keeps heavy
imports (torch, sentence-transformers, LanceDB) out of the server process.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from podcodex.rag.defaults import CHUNK_SIZE, CHUNK_THRESHOLD, DEFAULT_MODEL


def run(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    audio_path: str,
    output_dir: str | None,
    show: str,
    source: str,
    version_id: str | None,
    model_keys: list[str],
    chunkings: list[str],
    chunk_size: int,
    threshold: float,
    overwrite: bool,
) -> dict[str, Any]:
    """Vectorize one episode. Returns ``{chunks_upserted, source, stem}``.

    Raises when the transcript chunks to nothing: the episode panel asked for
    this episode, so "nothing indexed" is an error there, not a skip.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.core.source import load_source

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    episode = p.audio_path.stem

    progress_cb(0.0, "Resolving source...")
    consumed = load_source(audio_path, output_dir, version_id, step=source)
    upserted, source_label = _index_episode(
        progress_cb,
        cancelled,
        p,
        audio_path=audio_path,
        output_dir=output_dir,
        show=show,
        stem=episode,
        consumed=consumed,
        model_keys=model_keys,
        chunkings=chunkings,
        chunk_size=chunk_size,
        threshold=threshold,
        overwrite=overwrite,
    )
    if upserted == 0:
        raise ValueError(
            f"Indexing produced 0 chunks for '{episode}'. "
            "The transcript may be too short or have unsupported format."
        )
    # `stem` rides along so the server-side completion hook can warm its
    # indexed-stems cache without re-deriving the episode name.
    return {"chunks_upserted": upserted, "source": source_label, "stem": episode}


def run_for_batch(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    audio_path: str,
    stem: str,
    show_name: str,
    model_keys: list[str],
    chunkings: list[str],
    force: bool,
    version_id: str | None = None,
) -> dict[str, Any]:
    """Batch-mode index entry. Returns ``{upserted, indexed, skipped}``.

    The parent (``api/routes/batch.py``) has already skipped episodes that
    are fully indexed unless ``force``, so this does not ask again.
    ``indexed`` is False when the episode has no transcript to index or it
    chunked to nothing: in a batch that is a skip, not an error.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.core.source import NoTranscriptError, load_source

    p = AudioPaths.from_audio(audio_path)
    progress_cb(0.0, "Indexing...")
    try:
        consumed = load_source(audio_path, None, version_id)
    except NoTranscriptError:
        # No transcript of any kind yet: nothing to index in this batch. Any
        # other error (a corrupt file, a missing version) fails the episode.
        return {"upserted": 0, "indexed": False, "skipped": False}
    upserted, _ = _index_episode(
        progress_cb,
        cancelled,
        p,
        audio_path=audio_path,
        output_dir=None,
        show=show_name,
        stem=stem,
        consumed=consumed,
        model_keys=model_keys,
        chunkings=chunkings,
        chunk_size=CHUNK_SIZE,
        threshold=CHUNK_THRESHOLD,
        overwrite=force,
    )
    return {"upserted": upserted, "indexed": upserted > 0, "skipped": False}


def _index_episode(
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    p,
    *,
    audio_path: str,
    output_dir: str | None,
    show: str,
    stem: str,
    consumed,
    model_keys: list[str],
    chunkings: list[str],
    chunk_size: int,
    threshold: float,
    overwrite: bool,
) -> tuple[int, str]:
    """The index sequence both entries share: build, vectorize, compact, mark.

    Returns ``(chunks, source_label)``; 0 chunks means nothing was written
    and the episode is not marked. ``IndexingError`` (a failed combination)
    propagates before ``mark_step``, so a partial index is never flagged
    indexed. A cancel is honoured between combinations: the runner SIGTERMs
    the child a few seconds later anyway, and stopping at a boundary keeps
    the half-written combination out of the provenance.
    """
    from podcodex.core.device import device_str
    from podcodex.core.pipeline_db import mark_step
    from podcodex.core.provenance import build_provenance
    from podcodex.core.source import build_index_transcript
    from podcodex.ingest.show_registry import show_id_for_folder
    from podcodex.rag.index_store import get_index_store
    from podcodex.rag.indexing import vectorize_batch

    transcript = build_index_transcript(
        audio_path,
        show,
        stem,
        segments=consumed.segments,
        source=consumed.step,
        output_dir=output_dir,
    )
    if not transcript.get("segments"):
        return 0, ""
    source_label = transcript["meta"].get("source", "auto")
    local = get_index_store()
    progress_cb(0.05, "Starting vectorization...")

    def on_prog(step: int, total: int, label: str) -> None:
        if cancelled():
            raise RuntimeError("Cancelled")
        progress_cb(
            0.05 + 0.9 * (step / max(total, 1)), f"{label} ({step + 1}/{total})"
        )

    try:
        upserted = vectorize_batch(
            transcript,
            show,
            stem,
            model_keys,
            chunkings,
            local,
            show_id=show_id_for_folder(p.show_dir, show, mint=True),
            chunk_size=chunk_size,
            threshold=threshold,
            overwrite=overwrite,
            device=device_str(),
            on_progress=on_prog,
        )
    finally:
        # Lance keeps every superseded fragment until something reclaims it,
        # and a re-index is a delete plus an add. Compacted here, at the end
        # of the job that wrote them, including a cancelled or partly failed
        # one: the combinations already written are on disk either way, and
        # this subprocess's store (the only one that knows) exits next.
        local.compact()
    if upserted == 0:
        return 0, source_label

    provenance = build_provenance(
        "indexed",
        model=(model_keys or [DEFAULT_MODEL])[0],
        audio_path=audio_path,
        output_dir=output_dir,
        source=consumed,
        params={
            "source": source_label,
            "model_keys": model_keys,
            "chunkings": chunkings,
            "chunk_size": chunk_size,
            "threshold": threshold,
            "overwrite": overwrite,
        },
    )
    mark_step(p.show_dir, p.base.name, indexed=True, provenance={"indexed": provenance})
    return upserted, source_label


def already_indexed(
    audio_path: str,
    stem: str,
    show_name: str,
    model_keys: list[str],
    chunkings: list[str],
) -> bool:
    """True when every wanted (model, chunking) already holds this episode.

    Index reads only, no embedder: the batch runner asks it in the parent
    before spawning a child that would only answer "skipped". Reads the id
    without minting; the job itself mints when it writes.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.ingest.show_registry import show_id_for_folder
    from podcodex.rag.index_store import get_index_store

    wanted = [(m, c) for m in model_keys for c in chunkings]
    if not wanted:
        return False
    local = get_index_store()
    # From this folder's own show.toml, with no fallback to a lookup by name,
    # and strict: a folder with no id cannot own a same-named show's table,
    # which used to make this skip an episode that was never indexed.
    show_id = show_id_for_folder(AudioPaths.from_audio(audio_path).show_dir)
    return all(
        (
            col := local.resolve_collection(
                show_id, m, c, show_label=show_name, strict=True
            )
        )
        and local.episode_is_indexed(col, stem)
        for m, c in wanted
    )
