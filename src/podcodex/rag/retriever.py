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

from loguru import logger

from podcodex.rag.store import QdrantStore

try:
    import bm25s
except ImportError:  # pragma: no cover
    bm25s = None  # type: ignore[assignment]


class Retriever:
    """
    Combines an embedding model + Qdrant dense search + bm25s text search.

    Args:
        model      : model key from MODELS registry (default: "bge-m3")
        qdrant_url : Qdrant server URL (defaults to QDRANT_URL env or localhost:6333)
        store      : optional pre-built QdrantStore
    """

    def __init__(
        self,
        model: str = "bge-m3",
        qdrant_url: str | None = None,
        store: QdrantStore | None = None,
    ):
        from podcodex.rag.defaults import MODELS
        from podcodex.rag.embedder import get_embedder

        spec = MODELS.get(model)
        if spec is None:
            valid = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid: {valid}")

        self._model_key = model
        self._embedder = get_embedder(model)
        self._store = store or QdrantStore(url=qdrant_url)
        logger.info(f"Retriever ready (model={model})")

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict]:
        """
        Retrieve the top_k most relevant chunks for the query.

        Args:
            query      : natural language query
            collection : Qdrant collection name
            top_k      : number of results to return
            alpha      : blend between BM25 (0.0) and dense vector (1.0)

        Returns:
            List of payload dicts with an added 'score' key.
        """
        if alpha >= 1.0:
            return self._dense_search(query, collection, top_k)
        if alpha <= 0.0:
            return self._bm25_search(query, collection, top_k)
        return self._weighted_search(query, collection, top_k, alpha)

    # ── Private search methods ─────────────────

    def _dense_search(self, query: str, collection: str, top_k: int) -> list[dict]:
        query_vec = self._embedder.encode_query(query)
        results = self._store._client.query_points(
            collection_name=collection,
            query=query_vec.tolist(),
            limit=top_k,
        ).points
        return [_result_to_dict(r) for r in results]

    def _bm25_search(self, query: str, collection: str, top_k: int) -> list[dict]:
        results, _ = self._store._client.scroll(
            collection_name=collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return []

        chunks = [dict(r.payload) for r in results]
        texts = [c.get("text", "") for c in chunks]

        corpus_tokens = bm25s.tokenize(texts)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        query_tokens = bm25s.tokenize(query)
        k = min(top_k, len(chunks))
        indices, scores = retriever.retrieve(query_tokens, k=k)

        # ↓ only change: filter zero-padded results before normalizing
        hits = [
            {**chunks[int(idx)], "score": float(score)}
            for idx, score in zip(indices[0], scores[0])
            if float(score) > 1e-6
        ]

        return _normalize(hits)

    def _weighted_search(
        self, query: str, collection: str, top_k: int, alpha: float
    ) -> list[dict]:
        """Linear combination: score = alpha * dense + (1 - alpha) * bm25."""
        k = top_k * 4
        dense_hits = _normalize(self._dense_search(query, collection, k))
        bm25_hits = self._bm25_search(query, collection, k)  # already normalized

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

    def find(self, query: str, collection: str, top_k: int = 25) -> list[dict]:
        """
        True substring search — case-insensitive, no scoring.
        Requires a full-text payload index on 'text' (see QdrantStore.ensure_text_index).
        Results sorted by start time, all scored 1.0.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchText

        results, _ = self._store._client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="text", match=MatchText(text=query))]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        chunks = [dict(r.payload) for r in results]
        chunks.sort(key=lambda c: c.get("start", 0.0))
        return [{**c, "score": 1.0} for c in chunks]


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _chunk_key(chunk: dict) -> str:
    return f"{chunk.get('episode', '')}|{chunk.get('start', 0)}"


# def _normalize(results: list[dict]) -> list[dict]:
#     """Min-max normalize scores within a result set to [0, 1]."""
#     if not results:
#         return results
#     scores = [r["score"] for r in results]
#     lo, hi = min(scores), max(scores)
#     rng = hi - lo
#     if rng == 0:
#         # All scores identical: if all zero → no match at all → 0.0
#         #                       if all same positive value → tied → 1.0
#         val = 0.0 if lo == 0 else 1.0
#         return [{**r, "score": val} for r in results]
#     return [{**r, "score": (r["score"] - lo) / rng} for r in results]


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


def _result_to_dict(result) -> dict:
    payload = dict(result.payload or {})
    payload["score"] = result.score
    return payload
