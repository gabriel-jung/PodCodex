"""Indexing routes — vectorize episodes into the LanceDB IndexStore."""

from __future__ import annotations


from fastapi import APIRouter, Query
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import get_index_store, submit_task
from podcodex.api.schemas import TaskResponse
from podcodex.core._utils import AudioPaths

router = APIRouter()


# ── Config (available models + strategies) ───────────────


@router.get("/config")
async def index_config() -> dict:
    """Return available embedding models and chunking strategies."""
    from podcodex.rag.defaults import (
        CHUNKING_STRATEGIES,
        CHUNK_SIZE,
        CHUNK_THRESHOLD,
        DEFAULT_CHUNKING,
        DEFAULT_MODEL,
        MODELS,
    )

    return {
        "models": {
            key: {"label": spec.label, "description": spec.description}
            for key, spec in MODELS.items()
        },
        "chunking_strategies": CHUNKING_STRATEGIES,
        "defaults": {
            "model": DEFAULT_MODEL,
            "chunking": DEFAULT_CHUNKING,
            "chunk_size": CHUNK_SIZE,
            "threshold": CHUNK_THRESHOLD,
        },
    }


# ── Sources (available files to index) ───────────────────


@router.get("/sources")
async def index_sources(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """List available source files for indexing (transcript, corrected, translations).

    Returns a list of {key, label, detail, exists} dicts, ordered from most to
    least advanced.  The first entry with exists=True is the recommended default.
    Includes provenance detail (model, provider) when available.
    """
    from podcodex.core.translate import list_translations
    from podcodex.core.versions import get_latest_provenance

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    def _version_detail(step: str, lang: str | None = None) -> str:
        """Build a short human-readable detail string from the latest version's provenance."""
        try:
            meta = get_latest_provenance(p.base, step)
            if not meta:
                return ""
            parts: list[str] = []
            if meta.get("model"):
                parts.append(meta["model"])
            params = meta.get("params") or {}
            if params.get("llm_provider"):
                parts.append(str(params["llm_provider"]))
            elif params.get("llm_mode"):
                parts.append(str(params["llm_mode"]))
            if lang:
                parts.append(lang.replace("_", " ").title())
            if meta.get("manual_edit") or meta.get("type") == "validated":
                parts.append("edited")
            return ", ".join(parts)
        except Exception:
            return ""

    sources: list[dict] = []

    # Translations (most advanced) — one entry per language
    for lang in list_translations(audio_path, output_dir=output_dir):
        detail = _version_detail(lang, lang)
        sources.append(
            {
                "key": lang,
                "label": lang.replace("_", " ").title(),
                "detail": detail,
                "exists": True,
            }
        )

    # Corrected
    from podcodex.core.versions import has_version as _has_version

    corrected_exists = _has_version(p.base, "corrected")
    detail = _version_detail("corrected") if corrected_exists else ""
    sources.append(
        {
            "key": "corrected",
            "label": "Corrected",
            "detail": detail,
            "exists": corrected_exists,
        }
    )

    # Transcript
    transcript_exists = _has_version(p.base, "transcript")
    detail = _version_detail("transcript") if transcript_exists else ""
    sources.append(
        {
            "key": "transcript",
            "label": "Transcript",
            "detail": detail,
            "exists": transcript_exists,
        }
    )

    return sources


# ── Status ───────────────────────────────────────────────


@router.get("/status")
async def index_status(
    audio_path: str = Query(...),
    show: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Check indexing status per (model, chunking) combination."""
    from podcodex.rag.defaults import CHUNKING_STRATEGIES, MODELS
    from podcodex.rag.store import collection_name

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    episode = p.audio_path.stem

    local = get_index_store()
    combinations = []
    for model_key in MODELS:
        for chunking in CHUNKING_STRATEGIES:
            col = collection_name(show, model_key, chunking)
            indexed = local.episode_is_indexed(col, episode)
            count = local.episode_chunk_count(col, episode) if indexed else 0
            combinations.append(
                {
                    "model": model_key,
                    "chunking": chunking,
                    "indexed": indexed,
                    "chunk_count": count,
                }
            )
    return {"combinations": combinations, "db_exists": True}


# ── Collections ──────────────────────────────────────────


@router.get("/collections")
async def list_collections(
    show: str = Query(""),
) -> list[dict]:
    """List indexed collections, optionally filtered by show."""
    local = get_index_store()
    result = []
    for col_name in local.list_collections(show=show):
        info = local.get_collection_info(col_name)
        episodes = local.list_episodes(col_name)
        result.append(
            {
                "name": col_name,
                "model": info.get("model", "") if info else "",
                "chunker": info.get("chunker", "") if info else "",
                "episode_count": len(episodes),
            }
        )
    return result


@router.get("/episode-collections")
async def episode_collections(
    audio_path: str = Query(...),
    show: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """List the index entries this episode currently lives in.

    Returns one row per (collection, episode) — including model, chunker,
    the source step (transcript / corrected / <lang>) the chunks were
    derived from, and the chunk count.
    """
    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    episode = p.audio_path.stem

    local = get_index_store()
    out: list[dict] = []
    for col_name in local.list_collections(show=show):
        count = local.episode_chunk_count(col_name, episode)
        if not count:
            continue
        info = local.get_collection_info(col_name) or {}
        out.append(
            {
                "collection": col_name,
                "model": info.get("model", ""),
                "chunker": info.get("chunker", ""),
                "source": local.episode_source(col_name, episode),
                "chunk_count": count,
            }
        )
    return out


@router.delete("/episode")
async def delete_episode_from_index(
    audio_path: str = Query(...),
    show: str = Query(...),
    collection: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Remove this episode's chunks from one collection.

    If no other collections still hold the episode, also flips its
    ``indexed`` flag in pipeline.db so the UI reflects the change.
    """
    from podcodex.core.pipeline_db import mark_step

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    episode = p.audio_path.stem

    local = get_index_store()
    local.delete_episode(collection, episode)

    # If this episode is no longer in any collection of this show, clear flag.
    still_indexed = any(
        local.episode_is_indexed(c, episode) for c in local.list_collections(show=show)
    )
    if not still_indexed:
        mark_step(p.show_dir, episode, indexed=False)

    return {"status": "deleted", "still_indexed": still_indexed}


# ── Vectorize (background task) ──────────────────────────


class IndexRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    show: str
    source: str = "auto"
    version_id: str | None = None
    model_keys: list[str] = ["bge-m3"]
    chunkings: list[str] = ["semantic"]
    chunk_size: int = 256
    threshold: float = 0.5
    overwrite: bool = False

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_positive(cls, v: int) -> int:
        """Validate that chunk_size is at least 1."""
        if v < 1:
            raise ValueError("chunk_size must be at least 1")
        return v

    @field_validator("threshold")
    @classmethod
    def threshold_in_range(cls, v: float) -> float:
        """Validate that threshold is between 0.0 and 1.0 inclusive."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v


@router.post("/start", response_model=TaskResponse)
async def start_index(req: IndexRequest) -> TaskResponse:
    """Vectorize an episode into the IndexStore as a background task."""

    def run_index(progress_cb, req_data):
        """Execute the full vectorization pipeline for one episode.

        Args:
            progress_cb: Callable(fraction, message) used to broadcast progress
                over WebSocket to the frontend.
            req_data: IndexRequest instance carrying all vectorization parameters.

        Returns:
            Dict with ``chunks_upserted`` (int) and ``source`` (str) keys.
        """
        from podcodex.api.routes._helpers import build_index_transcript
        from podcodex.core.versions import load_version
        from podcodex.rag.indexing import vectorize_batch

        p = AudioPaths.from_audio(req_data.audio_path, output_dir=req_data.output_dir)
        episode = p.audio_path.stem

        progress_cb(0.0, "Resolving source...")

        # If a specific version is requested, load directly from version store
        if req_data.version_id and req_data.source != "auto":
            step = req_data.source
            segments = load_version(p.base, step, req_data.version_id)
            transcript = build_index_transcript(
                req_data.audio_path,
                req_data.show,
                episode,
                segments=segments,
                output_dir=req_data.output_dir,
            )
        else:
            transcript = build_index_transcript(
                req_data.audio_path,
                req_data.show,
                episode,
                source=req_data.source,
                output_dir=req_data.output_dir,
            )

        source_label = transcript["meta"].get("source", "auto")

        local = get_index_store()
        progress_cb(0.05, "Starting vectorization...")

        def on_progress(step, total, label):
            """Forward per-batch progress to the task progress callback."""
            frac = 0.05 + 0.9 * (step / max(total, 1))
            progress_cb(frac, f"{label} ({step + 1}/{total})")

        total_upserted = vectorize_batch(
            transcript,
            req_data.show,
            episode,
            req_data.model_keys,
            req_data.chunkings,
            local,
            chunk_size=req_data.chunk_size,
            threshold=req_data.threshold,
            overwrite=req_data.overwrite,
            on_progress=on_progress,
        )

        if total_upserted == 0:
            raise ValueError(
                f"Indexing produced 0 chunks for '{episode}'. "
                "The transcript may be too short or have unsupported format."
            )

        # Touch marker file for status detection
        marker = p.base.parent / ".rag_indexed"
        marker.touch()

        from podcodex.api.routes._helpers import build_provenance
        from podcodex.core.pipeline_db import mark_step

        provenance = build_provenance(
            "indexed",
            model=(req_data.model_keys or ["bge-m3"])[0],
            audio_path=req_data.audio_path,
            output_dir=req_data.output_dir,
            params={
                "source": source_label,
                "model_keys": req_data.model_keys,
                "chunkings": req_data.chunkings,
                "chunk_size": req_data.chunk_size,
                "threshold": req_data.threshold,
                "overwrite": req_data.overwrite,
            },
        )
        mark_step(
            p.show_dir, p.base.name, indexed=True, provenance={"indexed": provenance}
        )

        return {
            "chunks_upserted": total_upserted,
            "source": source_label,
        }

    return submit_task("index", req.audio_path, run_index, req)
