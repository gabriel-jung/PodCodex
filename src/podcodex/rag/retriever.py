"""
podcodex.rag.retriever — Hybrid retriever for podcast RAG.

Hybrid search = alpha * dense_vector_search + (1 - alpha) * BM25_text_search

    alpha=1.0 — dense vector search only
    alpha=0.0 — BM25 keyword search only
    0 < alpha < 1 — linear blend (default: 0.5)

BM25 is computed client-side via bm25s, independently of the embedding
model. This means ALL models support hybrid search.

Dense search uses numpy brute-force cosine similarity over vectors
loaded from the SQLite LocalStore.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

import bm25s

from podcodex.rag.localstore import LocalStore


@dataclass
class _BM25Cache:
    bm25: bm25s.BM25
    chunks: list[dict]
    chunk_count: int


@dataclass
class _VectorCache:
    matrix: np.ndarray  # (N, dim) L2-normalized float32
    chunks: list[dict]  # metadata for each row (no embeddings)
    chunk_count: int  # for staleness check
    created: float = field(default_factory=time.monotonic)


class Retriever:
    """
    Combines an embedding model + numpy dense search + bm25s text search,
    backed by a SQLite LocalStore.

    Args:
        model  : model key from MODELS registry (default: "bge-m3")
        local  : LocalStore instance (created with default path if None)
        device : torch device for the embedder (default: "cpu")
        ttl    : cache time-to-live in seconds (default: 180)
    """

    def __init__(
        self,
        model: str = "bge-m3",
        local: LocalStore | None = None,
        device: str = "cpu",
        ttl: float = 180.0,
    ):
        from podcodex.rag.defaults import MODELS
        from podcodex.rag.embedder import get_embedder

        spec = MODELS.get(model)
        if spec is None:
            valid = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid: {valid}")

        self._model_key = model
        self._embedder = get_embedder(model, device=device)
        self._local = local or LocalStore()
        self._ttl = ttl
        self._vector_cache: dict[str, _VectorCache] = {}
        self._bm25_cache: dict[str, _BM25Cache] = {}
        self._lock = threading.Lock()
        logger.info(f"Retriever ready (model={model})")

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        alpha: float = 0.5,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        """
        Retrieve the top_k most relevant chunks for the query.

        Args:
            query      : natural language query
            collection : collection name
            top_k      : number of results to return
            alpha      : blend between BM25 (0.0) and dense vector (1.0)
            episode    : if set, restrict search to this episode
            source     : if set, restrict to chunks from this source (e.g. "corrected")
            speaker    : if set, restrict to chunks from this speaker

        Returns:
            List of payload dicts with an added 'score' key.
        """
        if alpha >= 1.0:
            return self._dense_search(
                query,
                collection,
                top_k,
                episode=episode,
                source=source,
                speaker=speaker,
            )
        if alpha <= 0.0:
            return self._bm25_search(
                query,
                collection,
                top_k,
                episode=episode,
                source=source,
                speaker=speaker,
            )
        return self._weighted_search(
            query,
            collection,
            top_k,
            alpha,
            episode=episode,
            source=source,
            speaker=speaker,
        )

    # ── Private search methods ─────────────────

    def _get_vectors(self, collection: str) -> _VectorCache:
        """Return cached (matrix, chunks) for a collection, rebuilding if stale."""
        now = time.monotonic()
        with self._lock:
            cache = self._vector_cache.get(collection)
            if cache is not None:
                age = now - cache.created
                current_count = self._local.collection_chunk_count(collection)
                if age < self._ttl and cache.chunk_count == current_count:
                    return cache

        # Build outside the lock (I/O-heavy)
        matrix, chunks = self._local.load_all_vectors(collection)
        if matrix.size == 0:
            vc = _VectorCache(
                matrix=np.empty((0, 0), dtype=np.float32),
                chunks=[],
                chunk_count=0,
            )
            with self._lock:
                self._vector_cache[collection] = vc
            return vc

        # L2-normalize once at cache time
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        matrix = matrix / norms

        vc = _VectorCache(matrix=matrix, chunks=chunks, chunk_count=len(chunks))
        with self._lock:
            self._vector_cache[collection] = vc
        logger.debug(f"Vector cache built for '{collection}': {len(chunks)} vectors")
        return vc

    @staticmethod
    def _filter_mask(
        chunks: list[dict],
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> np.ndarray:
        """Return a boolean mask for metadata filters."""
        n = len(chunks)
        mask = np.ones(n, dtype=bool)
        if not (episode or source or speaker):
            return mask
        for i, c in enumerate(chunks):
            if episode and c.get("episode") != episode:
                mask[i] = False
            elif source and c.get("source") != source:
                mask[i] = False
            elif speaker and (c.get("dominant_speaker", c.get("speaker")) != speaker):
                mask[i] = False
        return mask

    def _dense_search(
        self,
        query: str,
        collection: str,
        top_k: int,
        *,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        vc = self._get_vectors(collection)
        if vc.matrix.size == 0:
            return []

        query_vec = self._embedder.encode_query(query)
        query_norm = query_vec / max(np.linalg.norm(query_vec), 1e-8)

        scores = vc.matrix @ query_norm
        mask = self._filter_mask(vc.chunks, episode, source, speaker)
        scores[~mask] = -1.0

        k = min(top_k, int(mask.sum()))
        if k == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0.01:
                continue
            results.append({**vc.chunks[idx], "score": score})
        return results

    def _bm25_search(
        self,
        query: str,
        collection: str,
        top_k: int,
        *,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        if episode or source or speaker:
            return self._bm25_search_filtered(
                query,
                collection,
                top_k,
                episode=episode,
                source=source,
                speaker=speaker,
            )

        # Full-collection query — use cache
        cache = self._bm25_cache.get(collection)
        current_count = self._local.collection_chunk_count(collection)

        if cache is not None and cache.chunk_count == current_count:
            logger.debug(f"BM25 cache hit for '{collection}' ({current_count} chunks)")
        else:
            logger.debug(f"BM25 cache miss for '{collection}' — rebuilding")
            chunks = self._local.load_all_chunks(collection)
            if not chunks:
                return []
            bm25_inst = self._build_bm25_index(chunks)
            cache = _BM25Cache(bm25=bm25_inst, chunks=chunks, chunk_count=current_count)
            self._bm25_cache[collection] = cache

        return self._query_bm25(query, cache.bm25, cache.chunks, top_k)

    def _bm25_search_filtered(
        self,
        query: str,
        collection: str,
        top_k: int,
        *,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        """BM25 search with metadata filters (not cached)."""
        chunks = self._local.load_all_chunks(collection, episode=episode)
        if source or speaker:
            chunks = [
                c
                for c in chunks
                if (not source or c.get("source") == source)
                and (
                    not speaker
                    or c.get("dominant_speaker", c.get("speaker")) == speaker
                )
            ]
        if not chunks:
            return []
        bm25_inst = self._build_bm25_index(chunks)
        return self._query_bm25(query, bm25_inst, chunks, top_k)

    @staticmethod
    def _build_bm25_index(chunks: list[dict]) -> bm25s.BM25:
        """Build a BM25 index from chunk texts."""
        texts = [c.get("text", "") for c in chunks]
        corpus_tokens = bm25s.tokenize(texts)
        bm25_inst = bm25s.BM25()
        bm25_inst.index(corpus_tokens)
        return bm25_inst

    @staticmethod
    def _query_bm25(
        query: str, bm25_inst: "bm25s.BM25", chunks: list[dict], top_k: int
    ) -> list[dict]:
        """Query a pre-built BM25 index and return normalized results."""
        query_tokens = bm25s.tokenize(query)
        k = min(top_k, len(chunks))
        indices, scores = bm25_inst.retrieve(query_tokens, k=k)

        hits = [
            {**chunks[int(idx)], "score": float(score)}
            for idx, score in zip(indices[0], scores[0])
            if float(score) > 1e-6
        ]
        return _normalize(hits)

    def invalidate(self, collection: str | None = None) -> None:
        """Clear vector and BM25 caches for a collection, or all."""
        with self._lock:
            if collection is None:
                self._vector_cache.clear()
                self._bm25_cache.clear()
            else:
                self._vector_cache.pop(collection, None)
                self._bm25_cache.pop(collection, None)

    # Keep old name as alias for compatibility
    invalidate_bm25_cache = invalidate

    def _weighted_search(
        self,
        query: str,
        collection: str,
        top_k: int,
        alpha: float,
        *,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        """Linear combination: score = alpha * dense + (1 - alpha) * bm25."""
        k = top_k * 4
        dense_hits = _normalize(
            self._dense_search(
                query, collection, k, episode=episode, source=source, speaker=speaker
            )
        )
        bm25_hits = self._bm25_search(
            query, collection, k, episode=episode, source=source, speaker=speaker
        )  # already normalized

        combined: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for r in dense_hits:
            key = _chunk_key(r)
            combined[key] = alpha * r["score"]
            payloads[key] = r

        for r in bm25_hits:
            key = _chunk_key(r)
            combined[key] = combined.get(key, 0.0) + (1 - alpha) * r["score"]
            payloads.setdefault(key, r)

        sorted_keys = sorted(combined, key=combined.__getitem__, reverse=True)[:top_k]
        return [{**payloads[k], "score": combined[k]} for k in sorted_keys]

    def random(
        self,
        collection: str,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> dict | None:
        """
        Pick a random chunk. If the chunk has multiple speaker turns,
        select one turn at random and return it as a single-speaker segment.
        """
        chunks = self._local.load_all_chunks(collection, episode=episode)
        if source or speaker:
            chunks = [
                c
                for c in chunks
                if (not source or c.get("source") == source)
                and (
                    not speaker
                    or c.get("dominant_speaker", c.get("speaker")) == speaker
                )
            ]
        if not chunks:
            return None

        chunk = random.choice(chunks)

        turns: list[dict] = chunk.get("speakers") or []
        if len(turns) > 1:
            if speaker:
                matching = [t for t in turns if t.get("speaker") == speaker]
                turns = matching or turns
            turn = random.choice(turns)
            return {
                "show": chunk.get("show", ""),
                "episode": chunk.get("episode", ""),
                "source": chunk.get("source", ""),
                "speaker": turn.get("speaker", "Unknown"),
                "text": turn.get("text", ""),
                "start": turn.get("start", chunk.get("start", 0.0)),
                "end": turn.get("end", chunk.get("end", 0.0)),
            }

        return chunk

    def find(
        self,
        query: str,
        collection: str,
        top_k: int = 25,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        """
        True substring search — case-insensitive, no scoring.
        Results sorted by start time, all scored 1.0.
        """
        chunks = self._local.load_all_chunks(collection, episode=episode)
        query_lower = query.lower()
        results = []
        for c in chunks:
            if query_lower not in c.get("text", "").lower():
                continue
            if source and c.get("source") != source:
                continue
            if speaker and c.get("dominant_speaker", c.get("speaker")) != speaker:
                continue
            results.append(c)

        results.sort(key=lambda c: c.get("start", 0.0))
        return [{**c, "score": 1.0} for c in results[:top_k]]


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _chunk_key(chunk: dict) -> str:
    """Deduplication key for merging dense + BM25 results."""
    return f"{chunk.get('show', '')}|{chunk.get('episode', '')}|{chunk.get('start', 0)}"


def _normalize(results: list[dict]) -> list[dict]:
    """
    Rank-based normalization to [1/n ... 1.0].
    Ensures no result ever scores 0 — every retrieved chunk
    gets a meaningful position-based score.
    """
    n = len(results)
    if n == 0:
        return results
    return [{**r, "score": 1.0 - (i / n)} for i, r in enumerate(results)]
