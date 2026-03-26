"""Search routes — hybrid retrieval over indexed episodes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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
    audio_path: str
    output_dir: str | None = None
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    top_k: int = 5
    alpha: float = 0.5
    episode: str | None = None
    speaker: str | None = None


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
    """Hybrid search: tries Qdrant first, falls back to local SQLite."""
    from podcodex.rag.defaults import MODELS
    from podcodex.rag.store import collection_name

    if req.model not in MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}")

    col = collection_name(req.show, req.model, req.chunking)

    # Try Qdrant-backed retriever first
    results = _try_qdrant_search(req, col)

    # Fall back to local SQLite search
    if results is None:
        p = AudioPaths.from_audio(req.audio_path, output_dir=req.output_dir)
        results = _local_search(
            db_path=p.vectors_db,
            collection=col,
            query=req.query,
            model=req.model,
            top_k=req.top_k,
            episode=req.episode,
            speaker=req.speaker,
        )

    return [_result_to_dict(r) for r in results]


def _try_qdrant_search(req: SearchRequest, col: str) -> list[dict] | None:
    """Attempt Qdrant search. Returns None if Qdrant is unavailable."""
    from podcodex.rag.store import qdrant_available

    if not qdrant_available():
        return None

    try:
        from podcodex.rag.retriever import Retriever

        retriever = Retriever(model=req.model)
        return retriever.retrieve(
            req.query,
            col,
            top_k=req.top_k,
            alpha=req.alpha,
            episode=req.episode,
            speaker=req.speaker,
        )
    except Exception:
        return None


def _local_search(
    db_path,
    collection: str,
    query: str,
    model: str,
    top_k: int,
    episode: str | None = None,
    speaker: str | None = None,
) -> list[dict]:
    """Dense-only search over LocalStore embeddings (no BM25)."""
    import numpy as np

    from podcodex.rag.embedder import get_embedder
    from podcodex.rag.localstore import LocalStore

    if not db_path.exists():
        return []

    local = LocalStore(db_path)

    # Get all episodes in this collection
    episodes = local.list_episodes(collection)
    if not episodes:
        local.close()
        return []

    if episode:
        episodes = [e for e in episodes if e == episode]

    # Load all chunks with embeddings
    all_chunks = []
    all_embeddings = []
    for ep in episodes:
        chunks = local.load_chunks(collection, ep)
        for chunk in chunks:
            if (
                speaker
                and chunk.get("dominant_speaker", chunk.get("speaker")) != speaker
            ):
                continue
            emb = chunk.pop("embedding", None)
            if emb is not None:
                all_chunks.append(chunk)
                all_embeddings.append(emb)

    local.close()

    if not all_chunks:
        return []

    # Embed the query
    embedder = get_embedder(model, device="cpu")
    query_vec = embedder.encode_query(query)

    # Cosine similarity
    embeddings_matrix = np.stack(all_embeddings)
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = embeddings_matrix / norms

    query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-8)
    scores = normalized @ query_norm

    # Top-k
    k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < 0.01:
            continue
        results.append({**all_chunks[idx], "score": score})

    return results


def _result_to_dict(r: dict) -> dict:
    """Normalize a retriever result into the SearchResult shape."""
    return {
        "text": r.get("text", ""),
        "episode": r.get("episode", ""),
        "speaker": r.get("dominant_speaker", r.get("speaker", "")),
        "start": r.get("start", 0.0),
        "end": r.get("end", 0.0),
        "score": r.get("score", 0.0),
        "source": r.get("source", ""),
        "speakers": r.get("speakers"),
    }
