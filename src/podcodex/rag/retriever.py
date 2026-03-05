"""
podcodex.rag.retriever — Query retriever for RAG strategies.

Loads the correct embedder at init and queries Qdrant using the
search method appropriate for the strategy:

    pplx_context  — dense-only (PplxEmbedder, 1024-dim)
    e5_semantic   — dense-only (E5Embedder, 384-dim)
    bge_speaker   — hybrid RRF (BGEEmbedder, dense + sparse)
    bge_semantic  — hybrid RRF (BGEEmbedder, dense + sparse)
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
    """Return the right embedder instance for the given strategy."""
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

    def retrieve(self, query: str, collection: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve the top_k most relevant chunks for the query.

        Returns:
            List of payload dicts with an added 'score' key.
        """
        if self._strategy in _DENSE_STRATEGIES:
            return self._dense_search(query, collection, top_k)
        return self._hybrid_search(query, collection, top_k)

    # ── Private search methods ─────────────────

    def _dense_search(self, query: str, collection: str, top_k: int) -> list[dict]:
        query_vec = self._embedder.encode_query(query)
        results = self._store._client.search(
            collection_name=collection,
            query_vector=query_vec.tolist(),
            limit=top_k,
        )
        return [_result_to_dict(r) for r in results]

    def _hybrid_search(self, query: str, collection: str, top_k: int) -> list[dict]:
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        query_emb = self._embedder.encode_query(query)
        dense_vec = query_emb["dense"]
        sparse_vec = query_emb["sparse"]

        results = self._store._client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(
                    query=dense_vec.tolist(),
                    using="dense",
                    limit=top_k * 4,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_vec["indices"],
                        values=sparse_vec["values"],
                    ),
                    using="sparse",
                    limit=top_k * 4,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
        return [_result_to_dict(r) for r in results.points]


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _result_to_dict(result) -> dict:
    """Convert a Qdrant ScoredPoint to a plain dict with an added 'score' key."""
    payload = dict(result.payload or {})
    payload["score"] = result.score
    return payload
