"""
podcodex.rag.retriever — Query retriever for RAG strategies.

Loads the correct embedder at init and queries Qdrant using the
search method appropriate for the strategy and the requested alpha:

    alpha=1.0 — dense vector search only
    alpha=0.0 — BM25 keyword search only  (BGE strategies only)
    0 < alpha < 1 — linear fusion: alpha*dense + (1-alpha)*BM25  (BGE only)

For dense-only strategies (e5_semantic, pplx_context), alpha is ignored
and dense search is always used.
"""

from __future__ import annotations

from loguru import logger

from podcodex.rag.store import STRATEGIES, QdrantStore

_DENSE_STRATEGIES = {"pplx_context", "e5_semantic"}
_HYBRID_STRATEGIES = {"bge_speaker", "bge_semantic"}


# ──────────────────────────────────────────────
# Embedder factory
# ──────────────────────────────────────────────


def _load_embedder(strategy: str):
    if strategy == "pplx_context":
        from podcodex.rag.embedder import PplxEmbedder

        return PplxEmbedder()
    elif strategy == "e5_semantic":
        from podcodex.rag.embedder import E5Embedder

        return E5Embedder()
    elif strategy in _HYBRID_STRATEGIES:
        from podcodex.rag.embedder import BGEEmbedder

        return BGEEmbedder()
    else:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose from {STRATEGIES}.")


# ──────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────


class Retriever:
    """
    Combines embedder + Qdrant client for a given vectorizing strategy.

    Args:
        strategy   : one of pplx_context / e5_semantic / bge_speaker / bge_semantic
        qdrant_url : Qdrant server URL (defaults to QDRANT_URL env or localhost:6333)
        store      : optional pre-built QdrantStore
    """

    def __init__(
        self,
        strategy: str,
        qdrant_url: str | None = None,
        store: QdrantStore | None = None,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from {STRATEGIES}."
            )
        self._strategy = strategy
        self._embedder = _load_embedder(strategy)
        self._store = store or QdrantStore(url=qdrant_url)
        logger.info(f"Retriever ready (strategy={strategy})")

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
            alpha      : blend between BM25 (0.0) and dense vector (1.0).
                         Ignored for dense-only strategies (always 1.0).

        Returns:
            List of payload dicts with an added 'score' key.
        """
        if self._strategy in _DENSE_STRATEGIES or alpha >= 1.0:
            return self._dense_search(query, collection, top_k)
        if alpha <= 0.0:
            return self._bm25_search(query, collection, top_k)
        return self._weighted_search(query, collection, top_k, alpha)

    # ── Private search methods ─────────────────

    def _dense_search(self, query: str, collection: str, top_k: int) -> list[dict]:
        query_vec = self._embedder.encode_query(query)
        vec = query_vec["dense"] if isinstance(query_vec, dict) else query_vec
        named = isinstance(query_vec, dict)
        results = self._store._client.query_points(
            collection_name=collection,
            query=vec.tolist(),
            using="dense" if named else None,
            limit=top_k,
        ).points
        return [_result_to_dict(r) for r in results]

    def _bm25_search(self, query: str, collection: str, top_k: int) -> list[dict]:
        from qdrant_client.models import SparseVector

        query_emb = self._embedder.encode_query(query)
        sv = query_emb["sparse"]
        results = self._store._client.query_points(
            collection_name=collection,
            query=SparseVector(indices=sv["indices"], values=sv["values"]),
            using="sparse",
            limit=top_k,
        ).points
        return [_result_to_dict(r) for r in results]

    def _weighted_search(
        self, query: str, collection: str, top_k: int, alpha: float
    ) -> list[dict]:
        """Linear combination: score = alpha*dense + (1-alpha)*bm25."""
        k = top_k * 4
        dense_hits = _normalize(
            _result_to_dict(r)
            for r in self._store._client.query_points(
                collection_name=collection,
                query=self._embedder.encode_query(query)["dense"].tolist(),
                using="dense",
                limit=k,
            ).points
        )
        bm25_hits = _normalize(self._bm25_search(query, collection, k))

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


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _chunk_key(chunk: dict) -> str:
    return f"{chunk.get('episode', '')}|{chunk.get('start', 0)}"


def _normalize(results) -> list[dict]:
    """Min-max normalize scores within a result set."""
    results = list(results)
    if not results:
        return results
    scores = [r["score"] for r in results]
    lo, hi = min(scores), max(scores)
    rng = hi - lo
    if rng == 0:
        return [{**r, "score": 1.0} for r in results]
    return [{**r, "score": (r["score"] - lo) / rng} for r in results]


def _result_to_dict(result) -> dict:
    payload = dict(result.payload or {})
    payload["score"] = result.score
    return payload
