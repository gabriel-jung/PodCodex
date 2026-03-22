"""
podcodex.rag.retriever — Hybrid retriever for podcast RAG.

Hybrid search = alpha * dense_vector_search + (1 - alpha) * BM25_text_search

    alpha=1.0 — dense vector search only
    alpha=0.0 — BM25 keyword search only
    0 < alpha < 1 — linear blend (default: 0.5)

BM25 is computed client-side via bm25s, independently of the embedding
model. This means ALL models support hybrid search.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from loguru import logger

from podcodex.rag.store import QdrantStore, _search_filter

import bm25s


@dataclass
class _BM25Cache:
    bm25: bm25s.BM25
    chunks: list[dict]
    point_count: int


class Retriever:
    """
    Combines an embedding model + Qdrant dense search + bm25s text search.

    Args:
        model      : model key from MODELS registry (default: "bge-m3")
        qdrant_url : Qdrant server URL (defaults to QDRANT_URL env or localhost:6333)
        store      : optional pre-built QdrantStore
        device     : torch device for the embedder (default: "cpu")
    """

    def __init__(
        self,
        model: str = "bge-m3",
        qdrant_url: str | None = None,
        store: QdrantStore | None = None,
        device: str = "cpu",
    ):
        from podcodex.rag.defaults import MODELS
        from podcodex.rag.embedder import get_embedder

        spec = MODELS.get(model)
        if spec is None:
            valid = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid: {valid}")

        self._model_key = model
        self._embedder = get_embedder(model, device=device)
        self._store = store or QdrantStore(url=qdrant_url)
        self._bm25_cache: dict[str, _BM25Cache] = {}
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
            collection : Qdrant collection name
            top_k      : number of results to return
            alpha      : blend between BM25 (0.0) and dense vector (1.0)
            episode    : if set, restrict search to this episode
            source     : if set, restrict to chunks from this source (e.g. "polished")
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
        query_vec = self._embedder.encode_query(query)
        query_filter = _search_filter(episode, source, speaker)
        return self._store.search_points(
            collection, query_vec.tolist(), top_k, query_filter
        )

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
            return self._bm25_search_uncached(
                query,
                collection,
                top_k,
                episode=episode,
                source=source,
                speaker=speaker,
            )

        # Full-collection query — use cache
        cache = self._bm25_cache.get(collection)
        current_count = self._store.collection_point_count(collection)

        if cache is not None and cache.point_count == current_count:
            logger.debug(f"BM25 cache hit for '{collection}' ({current_count} points)")
        else:
            logger.debug(f"BM25 cache miss for '{collection}' — rebuilding")
            chunks, bm25_inst = self._build_bm25_index(collection)
            if not chunks:
                return []
            cache = _BM25Cache(bm25=bm25_inst, chunks=chunks, point_count=current_count)
            self._bm25_cache[collection] = cache

        return self._query_bm25(query, cache.bm25, cache.chunks, top_k)

    def _bm25_search_uncached(
        self,
        query: str,
        collection: str,
        top_k: int,
        *,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        """BM25 search without caching (for filtered queries)."""
        scroll_filter = _search_filter(episode, source, speaker)
        chunks, bm25_inst = self._build_bm25_index(collection, scroll_filter)
        if not chunks:
            return []
        return self._query_bm25(query, bm25_inst, chunks, top_k)

    def _build_bm25_index(
        self, collection: str, scroll_filter=None
    ) -> tuple[list[dict], "bm25s.BM25 | None"]:
        """Scroll all points and build a BM25 index. Returns (chunks, bm25_instance)."""
        chunks = self._store.scroll_payloads(collection, scroll_filter=scroll_filter)
        if not chunks:
            return [], None

        texts = [c.get("text", "") for c in chunks]
        corpus_tokens = bm25s.tokenize(texts)
        bm25_inst = bm25s.BM25()
        bm25_inst.index(corpus_tokens)
        return chunks, bm25_inst

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

    def invalidate_bm25_cache(self, collection: str | None = None) -> None:
        """Clear BM25 cache for a collection, or all collections if None."""
        if collection is None:
            self._bm25_cache.clear()
        else:
            self._bm25_cache.pop(collection, None)

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
        scroll_filter = _search_filter(episode, source, speaker)
        chunk = self._store.random_point(collection, scroll_filter)
        if chunk is None:
            return None

        turns: list[dict] = chunk.get("speakers") or []
        if len(turns) > 1:
            # If speaker filter is set, narrow to matching turns
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
        Requires a full-text payload index on 'text' (created by QdrantStore.create_collection).
        Results sorted by start time, all scored 1.0.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue

        conditions = [FieldCondition(key="text", match=MatchText(text=query))]
        if episode:
            conditions.append(
                FieldCondition(key="episode", match=MatchValue(value=episode))
            )
        if source:
            conditions.append(
                FieldCondition(key="source", match=MatchValue(value=source))
            )
        if speaker:
            conditions.append(
                FieldCondition(key="speaker", match=MatchValue(value=speaker))
            )

        chunks = self._store.scroll_payloads(
            collection, scroll_filter=Filter(must=conditions), limit=top_k
        )
        chunks.sort(key=lambda c: c.get("start", 0.0))
        return [{**c, "score": 1.0} for c in chunks]


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _chunk_key(chunk: dict) -> str:
    """Deduplication key for merging dense + BM25 results."""
    return f"{chunk.get('episode', '')}|{chunk.get('start', 0)}"


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
