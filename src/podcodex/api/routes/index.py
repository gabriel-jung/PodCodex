"""Indexing routes — vectorize episodes into the LanceDB IndexStore."""

from __future__ import annotations


from fastapi import APIRouter, Query
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import get_index_store, submit_subprocess_task
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
    from podcodex.core.versions import get_latest_provenance, is_edited

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
            if is_edited(meta):
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
            # chunk_count > 0 already implies indexed; the separate
            # episode_is_indexed query was a redundant round-trip.
            count = local.episode_chunk_count(col, episode)
            combinations.append(
                {
                    "model": model_key,
                    "chunking": chunking,
                    "indexed": count > 0,
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
        summary = local.episode_collection_summary(col_name, episode)
        if not summary:
            continue
        info = local.get_collection_info(col_name) or {}
        out.append(
            {
                "collection": col_name,
                "model": info.get("model", ""),
                "chunker": info.get("chunker", ""),
                "source": summary["source"],
                "chunk_count": summary["chunk_count"],
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


# ── Inspect (read chunks + vector stats) ────────────────


@router.get("/inspect")
async def inspect_index(
    audio_path: str = Query(...),
    show: str = Query(...),
    model: str = Query(...),
    chunking: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Return chunks + per-chunk vector health stats for one (model, chunking).

    Powers the index inspector modal: lets the user read which text /
    speaker turns landed in each chunk and confirm the embedding vectors
    aren't dead (norm ≈ 0) or collapsed (mostly zeros).
    """
    from podcodex.rag.store import collection_name

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    episode = p.audio_path.stem
    col = collection_name(show, model, chunking)

    local = get_index_store()
    chunks = local.load_chunks_with_vector_stats(col, episode)
    info = local.get_collection_info(col) or {}

    # Some embedders (pplx in particular) have ~2-3% zeros as natural
    # sparsity, not a problem. Only flag chunks with substantially more.
    ZERO_WARN_THRESHOLD = 0.10
    norms = [c["vector_norm"] for c in chunks]
    summary = {
        "dim": int(info.get("dim", 0)),
        "n_chunks": len(chunks),
        "n_dead_chunks": sum(1 for n in norms if n < 1e-3),
        "n_collapsed_chunks": sum(
            1 for c in chunks if c.get("vector_zero_frac", 0.0) > 0.5
        ),
        "n_with_zeros": sum(
            1 for c in chunks if c.get("vector_zero_frac", 0.0) > ZERO_WARN_THRESHOLD
        ),
        "zero_warn_threshold": ZERO_WARN_THRESHOLD,
    }
    return {
        "collection": col,
        "model": model,
        "chunking": chunking,
        "episode": episode,
        "summary": summary,
        "chunks": chunks,
    }


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
    """Vectorize an episode into the IndexStore as a background task.

    The heavy work (embedding encode, LanceDB writes) runs in a spawned
    subprocess so the FastAPI event loop stays responsive.
    """

    return submit_subprocess_task(
        "index",
        req.audio_path,
        entry_path="podcodex.rag.index_job:run",
        kwargs={
            "audio_path": req.audio_path,
            "output_dir": req.output_dir,
            "show": req.show,
            "source": req.source,
            "version_id": req.version_id,
            "model_keys": req.model_keys,
            "chunkings": req.chunkings,
            "chunk_size": req.chunk_size,
            "threshold": req.threshold,
            "overwrite": req.overwrite,
        },
        req=req,
    )
