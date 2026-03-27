"""Search routes — hybrid retrieval over indexed episodes."""

from __future__ import annotations

from pathlib import Path

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
        db_path = _resolve_vectors_db(req)
        results = _local_search(
            db_path=db_path,
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


def _resolve_vectors_db(req: SearchRequest) -> Path:
    """Resolve vectors.db path from either audio_path or folder."""
    if req.folder:
        return Path(req.folder) / "vectors.db"
    if req.audio_path:
        p = AudioPaths.from_audio(req.audio_path, output_dir=req.output_dir)
        return p.vectors_db
    raise HTTPException(400, "Either audio_path or folder is required")


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


# ── Sync to Qdrant ───────────────────────────────────────


class SyncRequest(BaseModel):
    folder: str
    show: str
    overwrite: bool = False
    qdrant_url: str | None = None


@router.post("/sync")
async def sync_to_qdrant(req: SyncRequest) -> dict:
    """Push indexed episodes from LocalStore (SQLite) to Qdrant."""
    from podcodex.api.tasks import task_manager

    db_path = Path(req.folder) / "vectors.db"
    if not db_path.exists():
        raise HTTPException(404, "No vectors.db found — index episodes first")

    def run_sync(progress_cb):
        import numpy as np

        from podcodex.rag.localstore import LocalStore
        from podcodex.rag.store import QdrantStore

        local = LocalStore(db_path=db_path)
        store = QdrantStore(url=req.qdrant_url)

        collections = local.list_collections(show=req.show)
        if not collections:
            progress_cb(1.0, "No collections found")
            local.close()
            return

        total_chunks = 0
        for ci, col in enumerate(collections):
            info = local.get_collection_info(col)
            if info is None:
                continue

            store.create_collection(col, model=info["model"], overwrite=req.overwrite)

            episodes = local.list_episodes(col)
            for ei, ep in enumerate(episodes):
                cached = local.load_chunks(col, ep)
                if not cached:
                    continue

                local_count = len(cached)

                if not req.overwrite:
                    qdrant_count = store.episode_point_count(col, ep)
                    if local_count == qdrant_count:
                        continue
                    if qdrant_count > 0:
                        store.delete_episode_points(col, ep)

                chunks = [
                    {k: v for k, v in c.items() if k != "embedding"} for c in cached
                ]
                embeddings = np.stack([c["embedding"] for c in cached])
                store.upsert(col, chunks, embeddings)
                total_chunks += len(chunks)

                frac = (ci + (ei + 1) / len(episodes)) / len(collections)
                progress_cb(frac, f"Synced {ep} ({len(chunks)} chunks)")

        progress_cb(1.0, f"Done — {total_chunks} chunks pushed to Qdrant")
        local.close()

    task = task_manager.submit("sync", req.folder, run_sync)
    return {"task_id": task.task_id}


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


@router.post("/exact", response_model=list[SearchResult])
async def exact_search(req: ExactRequest) -> list[dict]:
    """Case-insensitive substring search over indexed chunks (like Ctrl+F)."""
    from podcodex.rag.localstore import LocalStore
    from podcodex.rag.store import collection_name

    db_path = _resolve_vectors_db(req)  # type: ignore[arg-type]
    if not db_path.exists():
        return []

    col = collection_name(req.show, req.model, req.chunking)
    local = LocalStore(db_path)
    episodes = local.list_episodes(col)
    if not episodes:
        local.close()
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
                and chunk.get("dominant_speaker", chunk.get("speaker")) != req.speaker
            ):
                continue
            results.append({**chunk, "score": 1.0})
            if len(results) >= req.top_k:
                break
        if len(results) >= req.top_k:
            break

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

    db_path = _resolve_vectors_db(req)  # type: ignore[arg-type]
    if not db_path.exists():
        return None

    col = collection_name(req.show, req.model, req.chunking)
    local = LocalStore(db_path)
    episodes = local.list_episodes(col)
    if not episodes:
        local.close()
        return None

    if req.episode:
        episodes = [e for e in episodes if e == req.episode]
    if not episodes:
        local.close()
        return None

    # Pick a random episode, then a random chunk from it
    ep = rng.choice(episodes)
    chunks = local.load_chunks_no_embeddings(col, ep)
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

    local.close()
    return {
        "collections": stats,
        "total_episodes": total_episodes,
        "total_chunks": total_chunks,
    }
