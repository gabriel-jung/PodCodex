"""Search routes — hybrid retrieval over indexed episodes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, field_validator

from podcodex.core._utils import AudioPaths

router = APIRouter()


# ── Config ────────────────────────────────────────────────


@router.get("/config")
async def search_config() -> dict:
    """Return available models, chunking strategies, and defaults."""
    from podcodex.rag.defaults import (
        ALPHA,
        CHUNKING_STRATEGIES,
        DEFAULT_CHUNKING,
        DEFAULT_MODEL,
        MODELS,
        TOP_K,
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
            "alpha": ALPHA,
            "top_k": TOP_K,
        },
    }


# ── Query ─────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    audio_path: str | None = None
    folder: str | None = None
    output_dir: str | None = None
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    top_k: int = 5
    alpha: float = 0.5
    episode: str | None = None
    speaker: str | None = None

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v

    @field_validator("alpha")
    @classmethod
    def alpha_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v


class SearchResult(BaseModel):
    text: str
    episode: str
    speaker: str
    start: float
    end: float
    score: float
    source: str
    speakers: list[dict] | None = None


@router.post("/query", response_model=list[SearchResult])
async def search_query(req: SearchRequest) -> list[dict]:
    """Hybrid search over locally indexed episodes."""
    from podcodex.rag.defaults import MODELS
    from podcodex.rag.localstore import LocalStore
    from podcodex.rag.retriever import Retriever
    from podcodex.rag.store import collection_name

    if req.model not in MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}")

    col = collection_name(req.show, req.model, req.chunking)
    db_path = _resolve_vectors_db(req)
    logger.info(
        "Search: show={!r} col={!r} db={} exists={} episode={!r}",
        req.show,
        col,
        db_path,
        db_path.exists(),
        req.episode,
    )
    if not db_path.exists():
        return []

    local = LocalStore(db_path)
    try:
        retriever = Retriever(model=req.model, local=local)
        results = retriever.retrieve(
            req.query,
            col,
            top_k=req.top_k,
            alpha=req.alpha,
            episode=req.episode,
            speaker=req.speaker,
        )
    except Exception:
        logger.opt(exception=True).warning("Search failed for collection {}", col)
        results = []
    finally:
        local.close()

    logger.info("Search: {} result(s)", len(results))
    return [_result_to_dict(r) for r in results]


def _resolve_vectors_db(req) -> Path:
    """Resolve vectors.db path from either audio_path or folder."""
    if getattr(req, "folder", None):
        return Path(req.folder) / "vectors.db"
    if getattr(req, "audio_path", None):
        output_dir = getattr(req, "output_dir", None)
        p = AudioPaths.from_audio(req.audio_path, output_dir=output_dir)
        return p.vectors_db
    raise HTTPException(400, "Either audio_path or folder is required")


def _result_to_dict(r: dict) -> dict:
    """Normalize a retriever result into the SearchResult shape."""
    return {
        "text": r.get("text", ""),
        "episode": r.get("episode_title") or r.get("episode", ""),
        "speaker": r.get("dominant_speaker", r.get("speaker", "")),
        "start": r.get("start", 0.0),
        "end": r.get("end", 0.0),
        "score": r.get("score", 0.0),
        "source": r.get("source", ""),
        "speakers": r.get("speakers"),
    }


# ── Exact (substring) search ────────────────────────────


class ExactRequest(BaseModel):
    query: str
    folder: str | None = None
    audio_path: str | None = None
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    top_k: int = 25
    episode: str | None = None
    speaker: str | None = None

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v


@router.post("/exact", response_model=list[SearchResult])
async def exact_search(req: ExactRequest) -> list[dict]:
    """Case-insensitive substring search over indexed chunks (like Ctrl+F)."""
    from podcodex.rag.localstore import LocalStore
    from podcodex.rag.store import collection_name

    db_path = _resolve_vectors_db(req)
    if not db_path.exists():
        return []

    col = collection_name(req.show, req.model, req.chunking)
    local = LocalStore(db_path)
    try:
        episodes = local.list_episodes(col)
        if not episodes:
            return []

        if req.episode:
            episodes = [e for e in episodes if e == req.episode]

        query_lower = req.query.lower()
        results: list[dict] = []
        for ep in episodes:
            chunks = local.load_chunks_no_embeddings(col, ep)
            for chunk in chunks:
                if query_lower not in chunk.get("text", "").lower():
                    continue
                if (
                    req.speaker
                    and chunk.get("dominant_speaker", chunk.get("speaker"))
                    != req.speaker
                ):
                    continue
                results.append({**chunk, "score": 1.0})
                if len(results) >= req.top_k:
                    break
            if len(results) >= req.top_k:
                break
    finally:
        local.close()
    results.sort(key=lambda c: (c.get("episode", ""), c.get("start", 0.0)))
    return [_result_to_dict(r) for r in results[: req.top_k]]


# ── Random quote ─────────────────────────────────────────


class RandomRequest(BaseModel):
    folder: str | None = None
    audio_path: str | None = None
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    episode: str | None = None
    speaker: str | None = None


@router.post("/random", response_model=SearchResult | None)
async def random_quote(req: RandomRequest) -> dict | None:
    """Pick a random indexed chunk."""
    import random as rng

    from podcodex.rag.localstore import LocalStore
    from podcodex.rag.store import collection_name

    db_path = _resolve_vectors_db(req)
    if not db_path.exists():
        return None

    col = collection_name(req.show, req.model, req.chunking)
    local = LocalStore(db_path)
    try:
        episodes = local.list_episodes(col)
        if not episodes:
            return None

        if req.episode:
            episodes = [e for e in episodes if e == req.episode]
        if not episodes:
            return None

        # Pick a random episode, then a random chunk from it
        ep = rng.choice(episodes)
        chunks = local.load_chunks_no_embeddings(col, ep)
    finally:
        local.close()

    if req.speaker:
        chunks = [
            c
            for c in chunks
            if c.get("dominant_speaker", c.get("speaker")) == req.speaker
        ]

    if not chunks:
        return None

    chunk = rng.choice(chunks)

    # If chunk has multiple speaker turns, pick one
    turns: list[dict] = chunk.get("speakers") or []
    if len(turns) > 1:
        if req.speaker:
            matching = [t for t in turns if t.get("speaker") == req.speaker]
            turns = matching or turns
        turn = rng.choice(turns)
        return _result_to_dict(
            {
                "episode": chunk.get("episode", ep),
                "episode_title": chunk.get("episode_title"),
                "source": chunk.get("source", ""),
                "speaker": turn.get("speaker", ""),
                "text": turn.get("text", ""),
                "start": turn.get("start", chunk.get("start", 0.0)),
                "end": turn.get("end", chunk.get("end", 0.0)),
                "score": 1.0,
            }
        )

    return _result_to_dict({**chunk, "score": 1.0})


# ── Index stats ──────────────────────────────────────────


@router.get("/stats")
async def index_stats(folder: str, show: str = "") -> dict:
    """Return index statistics for a show folder."""
    from podcodex.rag.localstore import LocalStore

    db_path = Path(folder) / "vectors.db"
    if not db_path.exists():
        return {"collections": [], "total_episodes": 0, "total_chunks": 0}

    local = LocalStore(db_path)
    try:
        collections = local.list_collections(show=show)

        stats: list[dict] = []
        total_episodes = 0
        total_chunks = 0
        for col in collections:
            info = local.get_collection_info(col)
            episodes = local.list_episodes(col)
            chunk_count = sum(local.episode_chunk_count(col, ep) for ep in episodes)
            stats.append(
                {
                    "collection": col,
                    "model": info["model"] if info else "",
                    "chunking": info["chunker"] if info else "",
                    "episodes": len(episodes),
                    "chunks": chunk_count,
                }
            )
            total_episodes += len(episodes)
            total_chunks += chunk_count
    finally:
        local.close()
    return {
        "collections": stats,
        "total_episodes": total_episodes,
        "total_chunks": total_chunks,
    }
