"""Subprocess entry point for the RAG index pipeline step.

Invoked by ``podcodex.api.subprocess_runner`` as a top-level target so it
can be pickled + launched via ``multiprocessing.spawn``. Keeps heavy
imports (torch, sentence-transformers, LanceDB) out of the server process.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from podcodex.core.cancel import TaskCancelled
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
        source_ref=_source_ref(p.base, source, version_id, consumed),
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
    source: str = "auto",
    source_ref: str | None = None,
) -> dict[str, Any]:
    """Batch-mode index entry. Returns ``{upserted, indexed, skipped}``.

    *source* is ``auto`` (the canonical seglist) unless the episode's index
    was built from a step the user picked, which a batch re-index keeps
    (see :func:`batch_index_source`). *source_ref* overrides how the run is
    recorded: a version the batch picked for the whole pipeline is not a pin
    of the index (``auto``), one kept from an earlier pin is (``pinned``).

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
        consumed = load_source(audio_path, None, version_id, step=source)
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
        source_ref=source_ref or _source_ref(p.base, source, version_id, consumed),
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
    source_ref: str = "auto",
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
    from podcodex.core.versions import version_content_hash
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
    # Recorded on every chunk of this install's index, so a later edit of
    # the source (a new content hash) reads as an outdated index, and a
    # re-run re-chunks instead of skipping the episode as cached. The ref
    # says what the hash is compared against later (see _source_ref).
    source_hash = version_content_hash(p.base, consumed.step, consumed.version_id)
    if source_hash:
        transcript["meta"]["source_hash"] = source_hash
        transcript["meta"]["source_ref"] = source_ref
        transcript["meta"]["source_version"] = consumed.version_id
    source_label = transcript["meta"].get("source", "auto")
    local = get_index_store()
    progress_cb(0.05, "Starting vectorization...")

    def on_prog(step: int, total: int, label: str) -> None:
        if cancelled():
            raise TaskCancelled()
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


def _source_ref(base, source: str | None, version_id: str | None, consumed) -> str:
    """How the indexed source was chosen, which decides what counts as stale.

    ``auto``: the episode's canonical seglist, compared against the current
    one. ``step``: a step the user picked (a translation, the raw
    transcript), compared against that step's current default version.
    ``pinned``: an exact version, which no later edit makes outdated.
    Picking the version that is canonical right now is following the
    canonical seglist, not a pin: the source pickers preselect it, and a pin
    would freeze the index on it through every later edit. Judged on the
    loaded version (step and id: ids from before sub-second stamps repeat
    across steps) once the run has it, so a batch's correct step has landed.
    """
    if version_id:
        from podcodex.core.versions import resolve_canonical_ref

        canonical = resolve_canonical_ref(base)
        loaded = (consumed.step, consumed.version_id)
        return "auto" if canonical == loaded else "pinned"
    return "auto" if source in (None, "", "auto") else "step"


def already_indexed(
    audio_path: str,
    stem: str,
    show_name: str,
    model_keys: list[str],
    chunkings: list[str],
    version_id: str | None = None,
) -> bool:
    """True when every wanted (model, chunking) already holds this episode,
    cut from the source this run would index.

    Index reads only, no embedder: the batch runner asks it in the parent
    before spawning a child that would only answer "skipped". Reads the id
    without minting; the job itself mints when it writes.

    With *version_id* (a source picked for this batch) the chunks must come
    from that version, and chunks recording nothing do not. Without one, chunks of the canonical seglist are
    current while its hash is unchanged, chunks of a picked step while that
    step's default version is, and a pinned version always is (an unpinned
    batch never replaces it). Chunks recording nothing predate the record
    and count as current, unless a version was picked. Same rule as the
    status overlay's index_outdated.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import version_meta_by_id
    from podcodex.ingest.show_registry import show_id_for_folder
    from podcodex.rag.index_store import get_index_store

    wanted = [(m, c) for m in model_keys for c in chunkings]
    if not wanted:
        return False
    local = get_index_store()
    p = AudioPaths.from_audio(audio_path)
    # From this folder's own show.toml, with no fallback to a lookup by name,
    # and strict: a folder with no id cannot own a same-named show's table,
    # which used to make this skip an episode that was never indexed.
    show_id = show_id_for_folder(p.show_dir)
    records = []
    for m, c in wanted:
        col = local.resolve_collection(show_id, m, c, show_label=show_name, strict=True)
        if not col or not local.episode_is_indexed(col, stem):
            return False
        records.append(local.episode_source_record(col, stem))
    if version_id:
        meta = version_meta_by_id(p.base, version_id)
        target = meta.get("content_hash") if meta else None
        # Chunks recording nothing predate the record: they may come from any
        # version, so they do not satisfy an explicit pick.
        return all(
            r is not None and (not target or r["hash"] == target) for r in records
        )
    return not any(_record_is_stale(p.base, r) for r in records)


def _record_is_stale(base, record: dict | None) -> bool:
    """One collection's chunks of an episode are behind their source.

    The rule the status overlay's ``index_outdated`` applies: canonical
    (``auto``) chunks against the canonical seglist, a picked step's chunks
    against that step's default version, pinned chunks never. No record
    (chunks from before it existed) or nothing to compare against: current.
    """
    from podcodex.core.versions import current_source_hash

    if not record or record["ref"] == "pinned":
        return False
    now = current_source_hash(base, record["step"], record["ref"])
    return bool(now) and record["hash"] != now


def stale_index_combos(
    audio_path: str, stem: str, show_name: str
) -> list[tuple[str, str]]:
    """``(model, chunker)`` of every collection holding stale chunks of *stem*.

    Across all the show's collections, not only the ones a batch asks for:
    the status overlay flags the episode while any of them is stale, so the
    batch re-indexes all of them, or the flag could never be cleared by it.
    Collections of a model or chunker this install does not know are left
    out: re-indexing them could only fail.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.ingest.show_registry import show_id_for_folder
    from podcodex.rag.defaults import is_known_collection
    from podcodex.rag.index_store import get_index_store

    local = get_index_store()
    p = AudioPaths.from_audio(audio_path)
    show_id = show_id_for_folder(p.show_dir)
    info = local.get_all_collection_info()
    combos = []
    for col in local.collections_for_show(
        show_id, show_label=show_name, strict=True, info=info
    ):
        meta = info.get(col) or {}
        if is_known_collection(meta):
            if _record_is_stale(p.base, local.episode_source_record(col, stem)):
                combos.append((meta["model"], meta["chunker"]))
    return combos


def batch_index_source(
    audio_path: str,
    stem: str,
    show_name: str,
    model_keys: list[str],
    chunkings: list[str],
) -> tuple[str, str | None, str]:
    """``(source, version_id, source_ref)`` an unpinned batch indexes from.

    The canonical seglist (``"auto", None, "auto"``), unless the episode's
    index was built from something the user picked: a step (kept as that
    step) or a pinned version (kept, while that version still exists).
    Re-indexing either from the canonical source, which a missing collection,
    a stale one or ``force`` triggers, would silently replace the choice.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.core.versions import version_meta_by_id, version_path
    from podcodex.ingest.show_registry import show_id_for_folder
    from podcodex.rag.index_store import get_index_store

    local = get_index_store()
    p = AudioPaths.from_audio(audio_path)
    show_id = show_id_for_folder(p.show_dir)
    for m in model_keys:
        for c in chunkings:
            col = local.resolve_collection(
                show_id, m, c, show_label=show_name, strict=True
            )
            record = local.episode_source_record(col, stem) if col else None
            if not record:
                continue
            if record["ref"] == "pinned" and record.get("version"):
                vid = record["version"]
                meta = version_meta_by_id(p.base, vid)
                live = bool(meta) and version_path(p.base, meta["step"], vid).exists()
                # A deleted pin falls back to the canonical source rather
                # than failing the episode on "Source version not found".
                if live:
                    return "auto", vid, "pinned"
            elif record["ref"] == "step" and record["step"]:
                return record["step"], None, "step"
    return "auto", None, "auto"
