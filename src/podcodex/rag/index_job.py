"""Subprocess entry point for the RAG index pipeline step.

Invoked by ``podcodex.api.subprocess_runner`` as a top-level target so it
can be pickled + launched via ``multiprocessing.spawn``. Keeps heavy
imports (torch, sentence-transformers, LanceDB) out of the server process.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


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
    """Vectorize one episode. Returns ``{chunks_upserted, source}``."""
    from podcodex.api.routes._helpers import (
        build_index_transcript,
        build_provenance,
        get_index_store,
    )
    from podcodex.core._utils import AudioPaths
    from podcodex.core.pipeline_db import mark_step
    from podcodex.core.versions import load_version_by_id
    from podcodex.rag.indexing import vectorize_batch

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    episode = p.audio_path.stem

    progress_cb(0.0, "Resolving source...")

    if version_id:
        resolved = load_version_by_id(p.base, version_id)
        if not resolved:
            raise ValueError(f"version_id {version_id!r} not found for {episode!r}")
        segments, resolved_step = resolved
        transcript = build_index_transcript(
            audio_path,
            show,
            episode,
            segments=segments,
            source=resolved_step,
            output_dir=output_dir,
        )
    else:
        transcript = build_index_transcript(
            audio_path, show, episode, source=source, output_dir=output_dir
        )

    source_label = transcript["meta"].get("source", "auto")
    local = get_index_store()
    progress_cb(0.05, "Starting vectorization...")

    def on_prog(step: int, total: int, label: str) -> None:
        frac = 0.05 + 0.9 * (step / max(total, 1))
        progress_cb(frac, f"{label} ({step + 1}/{total})")

    from podcodex.core.device import device_str

    device = device_str()

    from podcodex.ingest.show_registry import show_id_for_label

    total_upserted = vectorize_batch(
        transcript,
        show,
        episode,
        model_keys,
        chunkings,
        local,
        show_id=show_id_for_label(show),
        chunk_size=chunk_size,
        threshold=threshold,
        overwrite=overwrite,
        device=device,
        on_progress=on_prog,
    )

    if total_upserted == 0:
        raise ValueError(
            f"Indexing produced 0 chunks for '{episode}'. "
            "The transcript may be too short or have unsupported format."
        )

    provenance = build_provenance(
        "indexed",
        model=(model_keys or ["bge-m3"])[0],
        audio_path=audio_path,
        output_dir=output_dir,
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

    # `stem` rides along so the server-side completion hook can warm its
    # indexed-stems cache without re-deriving the episode name.
    return {"chunks_upserted": total_upserted, "source": source_label, "stem": episode}


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
    """Batch-mode index entry — skips already-indexed combinations unless forced.

    Returns ``{upserted: int, indexed: bool}``. ``indexed`` is False when the
    episode had no segments to vectorize or when everything is already
    indexed and ``force`` is False.
    """
    from podcodex.api.routes._helpers import (
        build_index_transcript,
        build_provenance,
        get_index_store,
    )
    from podcodex.core._utils import AudioPaths
    from podcodex.core.pipeline_db import mark_step
    from podcodex.ingest.show_registry import show_id_for_label
    from podcodex.rag.indexing import vectorize_batch

    p = AudioPaths.from_audio(audio_path)
    local = get_index_store()
    show_id = show_id_for_label(show_name)

    if not force:
        wanted = [(m, c) for m in model_keys for c in chunkings]
        if wanted and all(
            (col := local.resolve_collection(show_id, m, c, show_label=show_name))
            and local.episode_is_indexed(col, stem)
            for m, c in wanted
        ):
            return {"upserted": 0, "indexed": False, "skipped": True}

    progress_cb(0.0, "Indexing...")
    if version_id:
        from podcodex.core.versions import load_version_by_id

        resolved = load_version_by_id(p.base, version_id)
        if not resolved:
            return {"upserted": 0, "indexed": False, "skipped": False}
        segments, resolved_step = resolved
        transcript = build_index_transcript(
            audio_path, show_name, stem, segments=segments, source=resolved_step
        )
    else:
        transcript = build_index_transcript(audio_path, show_name, stem)
    if not transcript.get("segments"):
        return {"upserted": 0, "indexed": False, "skipped": False}

    def on_prog(step: int, total: int, label: str) -> None:
        progress_cb(step / max(total, 1), f"{label} ({step + 1}/{total})")

    from podcodex.core.device import device_str

    upserted = vectorize_batch(
        transcript,
        show_name,
        stem,
        model_keys,
        chunkings,
        local,
        show_id=show_id,
        overwrite=force,
        device=device_str(),
        on_progress=on_prog,
    )

    if upserted == 0:
        return {"upserted": 0, "indexed": False, "skipped": False}

    provenance = build_provenance(
        "indexed",
        model=(model_keys or ["bge-m3"])[0],
        audio_path=audio_path,
        params={"model_keys": model_keys, "chunkings": chunkings},
    )
    mark_step(p.show_dir, p.base.name, indexed=True, provenance={"indexed": provenance})

    return {"upserted": upserted, "indexed": True, "skipped": False}
