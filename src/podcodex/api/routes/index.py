"""Indexing routes — vectorize episodes into LocalStore for search."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from podcodex.api.schemas import TaskResponse
from podcodex.api.tasks import task_manager
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


# ── Status ───────────────────────────────────────────────


@router.get("/status")
async def index_status(
    audio_path: str = Query(...),
    show: str = Query(...),
    output_dir: str | None = Query(None),
) -> dict:
    """Check indexing status per (model, chunking) combination."""
    from podcodex.rag.defaults import CHUNKING_STRATEGIES, MODELS
    from podcodex.rag.localstore import LocalStore
    from podcodex.rag.store import collection_name

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    episode = p.audio_path.stem

    db_path = p.vectors_db
    if not db_path.exists():
        return {"combinations": [], "db_exists": False}

    local = LocalStore(db_path)
    try:
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
    finally:
        local.close()
    return {"combinations": combinations, "db_exists": True}


# ── Collections ──────────────────────────────────────────


@router.get("/collections")
async def list_collections(
    audio_path: str = Query(...),
    output_dir: str | None = Query(None),
) -> list[dict]:
    """List all collections in the show's vectors.db."""
    from podcodex.rag.localstore import LocalStore

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    db_path = p.vectors_db
    if not db_path.exists():
        return []

    local = LocalStore(db_path)
    try:
        result = []
        for col_name in local.list_collections():
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
    finally:
        local.close()
    return result


# ── Vectorize (background task) ──────────────────────────


class IndexRequest(BaseModel):
    audio_path: str
    output_dir: str | None = None
    show: str
    source: str = "auto"
    model_keys: list[str] = ["bge-m3"]
    chunkings: list[str] = ["semantic"]
    chunk_size: int = 256
    threshold: float = 0.5
    overwrite: bool = False


@router.post("/start", response_model=TaskResponse)
async def start_index(req: IndexRequest) -> TaskResponse:
    """Vectorize an episode into LocalStore as a background task."""

    def run_index(progress_cb, req_data):
        from podcodex.cli import _resolve_source, _source_label, vectorize_batch
        from podcodex.rag.localstore import LocalStore

        p = AudioPaths.from_audio(req_data.audio_path, output_dir=req_data.output_dir)
        episode = p.audio_path.stem

        # Find the transcript file
        transcript_path = p.transcript_best
        if not transcript_path.exists():
            raise ValueError("No transcript found — transcribe first")

        # Resolve source (auto picks polished > transcript)
        progress_cb(0.0, "Resolving source...")
        source_path = _resolve_source(transcript_path, req_data.source)
        source_label = _source_label(source_path, transcript_path)

        # Load and prepare transcript dict
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            transcript = {
                "meta": {
                    "show": req_data.show,
                    "episode": episode,
                    "source": source_label,
                },
                "segments": data,
            }
        else:
            transcript = data
            transcript.setdefault("meta", {})
            transcript["meta"].setdefault("show", req_data.show)
            transcript["meta"].setdefault("episode", episode)
            transcript["meta"].setdefault("source", source_label)

        # Open LocalStore at show level
        db_path = p.vectors_db
        local = LocalStore(db_path)

        progress_cb(0.05, "Starting vectorization...")

        def on_progress(step, total, label):
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

        # Touch marker file for status detection
        marker = p.base.parent / ".rag_indexed"
        marker.touch()

        return {
            "chunks_upserted": total_upserted,
            "source": source_label,
        }

    try:
        info = task_manager.submit("index", req.audio_path, run_index, req)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return TaskResponse(task_id=info.task_id)
