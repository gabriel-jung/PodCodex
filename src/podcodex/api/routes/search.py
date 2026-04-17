"""Search routes — hybrid retrieval over the global LanceDB IndexStore."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import AUDIO_EXTS, get_index_store
from podcodex.core._utils import humanize_stem
from podcodex.rag.retriever import get_retriever

router = APIRouter()


def _build_audio_lookup() -> dict[str, dict[str, str]]:
    """Per-request map: show name → {episode_stem: audio_path_str}.

    Each show folder is scanned once (one ``iterdir``) rather than probing
    every ``AUDIO_EXTS`` extension per search hit.
    """
    from podcodex.api.routes.config import _load
    from podcodex.ingest.show import load_show_meta

    out: dict[str, dict[str, str]] = {}
    for folder_path in _load().show_folders:
        p = Path(folder_path)
        if not p.is_dir():
            continue
        meta = load_show_meta(p)
        name = (meta.name if meta else None) or p.name
        stems: dict[str, str] = {}
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                stems[f.stem] = str(f)
        out[name] = stems
    return out


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
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    top_k: int = 5
    alpha: float = 0.5
    episode: str | None = None
    speaker: str | None = None
    source: str | None = None

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        """Validate that top_k is at least 1."""
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v

    @field_validator("alpha")
    @classmethod
    def alpha_in_range(cls, v: float) -> float:
        """Validate that alpha is between 0.0 and 1.0 inclusive."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v


class SearchResult(BaseModel):
    text: str
    episode: str
    episode_stem: str = ""
    audio_path: str = ""
    speaker: str
    start: float
    end: float
    score: float
    source: str
    speakers: list[dict] | None = None
    accent_match: bool = False
    fuzzy_match: bool = False
    match_text: str | None = None


@router.post("/query", response_model=list[SearchResult])
async def search_query(req: SearchRequest) -> list[dict]:
    """Hybrid search over the global LanceDB index."""
    from podcodex.rag.defaults import MODELS
    from podcodex.rag.store import collection_name

    if req.model not in MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}")

    col = collection_name(req.show, req.model, req.chunking)
    logger.info("Search: show={!r} col={!r} episode={!r}", req.show, col, req.episode)
    try:
        retriever = get_retriever(req.model)
        results = retriever.retrieve(
            req.query,
            col,
            top_k=req.top_k,
            alpha=req.alpha,
            episode=req.episode,
            speaker=req.speaker,
            source=req.source,
        )
    except Exception:
        logger.opt(exception=True).warning("Search failed for collection {}", col)
        results = []

    logger.info("Search: {} result(s)", len(results))
    audio_lookup = _build_audio_lookup()
    return [_result_to_dict(r, audio_lookup) for r in results]


def _result_to_dict(
    r: dict, audio_lookup: dict[str, dict[str, str]] | None = None
) -> dict:
    stem = r.get("episode", "")
    title = r.get("episode_title") or humanize_stem(stem)
    show_name = r.get("show", "")
    audio_path = (
        (audio_lookup.get(show_name) or {}).get(stem, "")
        if audio_lookup is not None and stem and show_name
        else ""
    )
    return {
        "text": r.get("text", ""),
        "episode": title,
        "episode_stem": stem,
        "audio_path": audio_path,
        "speaker": r.get("dominant_speaker", r.get("speaker", "")),
        "start": r.get("start", 0.0),
        "end": r.get("end", 0.0),
        "score": r.get("score", 0.0),
        "source": r.get("source", ""),
        "speakers": r.get("speakers"),
        "accent_match": bool(r.get("accent_match", False)),
        "fuzzy_match": bool(r.get("fuzzy_match", False)),
        "match_text": r.get("match_text"),
    }


# ── Exact (token-match) search ───────────────────────────


class ExactRequest(BaseModel):
    query: str
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    episode: str | None = None
    speaker: str | None = None
    source: str | None = None


@router.post("/exact", response_model=list[SearchResult])
async def exact_search(req: ExactRequest) -> list[dict]:
    """Phrase search: returns all exact, accent-variant, and near-typo matches."""
    from podcodex.rag.store import collection_name

    col = collection_name(req.show, req.model, req.chunking)
    retriever = get_retriever(req.model)
    hits = retriever.find(
        req.query,
        col,
        episode=req.episode,
        speaker=req.speaker,
        source=req.source,
    )
    audio_lookup = _build_audio_lookup()
    return [_result_to_dict(h, audio_lookup) for h in hits]


# ── Random quote ─────────────────────────────────────────


class RandomRequest(BaseModel):
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    episode: str | None = None
    speaker: str | None = None
    source: str | None = None


@router.post("/random", response_model=SearchResult | None)
async def random_quote(req: RandomRequest) -> dict | None:
    """Pick a random indexed chunk (optionally filtered)."""
    from podcodex.rag.store import collection_name

    col = collection_name(req.show, req.model, req.chunking)
    retriever = get_retriever(req.model)
    chunk = retriever.random(
        col, episode=req.episode, speaker=req.speaker, source=req.source
    )
    if chunk is None:
        return None
    return _result_to_dict({**chunk, "score": 1.0}, _build_audio_lookup())


# ── Index stats ──────────────────────────────────────────


@router.get("/stats")
async def index_stats(show: str = "") -> dict:
    """Return index statistics, optionally scoped to one show."""
    local = get_index_store()
    collections = local.list_collections(show=show)

    stats: list[dict] = []
    total_episodes = 0
    total_chunks = 0
    for col in collections:
        info = local.get_collection_info(col)
        ep_count = len(local.list_episodes(col))
        chunk_count = local.collection_chunk_count(col)
        sources = local.list_sources(col)
        stats.append(
            {
                "collection": col,
                "model": info["model"] if info else "",
                "chunking": info["chunker"] if info else "",
                "episodes": ep_count,
                "chunks": chunk_count,
                "sources": sources,
            }
        )
        total_episodes += ep_count
        total_chunks += chunk_count
    return {
        "collections": stats,
        "total_episodes": total_episodes,
        "total_chunks": total_chunks,
    }
