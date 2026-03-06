"""
podcodex.rag.store — Qdrant storage layer for podcast RAG.

One Qdrant collection per (show, strategy).
Collection naming: "{show}__{strategy}" (e.g. "my_podcast__bge_semantic").

Supported strategies and their vector configs:
    pplx_context — dense 1024-dim (cosine)
    e5_semantic  — dense 384-dim  (cosine)
    bge_speaker  — dense 1024-dim (cosine) + sparse (native hybrid)
    bge_semantic — dense 1024-dim (cosine) + sparse (native hybrid)

The Qdrant URL defaults to QDRANT_URL env var, falling back to localhost:6333.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Union

import numpy as np
from loguru import logger


# ──────────────────────────────────────────────
# Strategy definitions
# ──────────────────────────────────────────────

STRATEGIES = ("pplx_context", "e5_semantic", "bge_speaker", "bge_semantic")

_STRATEGY_CONFIG: dict[str, dict] = {
    "pplx_context": {"type": "dense", "dim": 1024},
    "e5_semantic": {"type": "dense", "dim": 384},
    "bge_speaker": {"type": "hybrid", "dim": 1024},
    "bge_semantic": {"type": "hybrid", "dim": 1024},
}


def _normalize_show(show: str) -> str:
    """Lowercase and collapse non-alphanumeric runs to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", show.lower()).strip("_")


def collection_name(show: str, strategy: str) -> str:
    """Build the canonical collection name for a (show, strategy) pair.

    Normalizes the show name so "My Podcast" and "my_podcast" resolve to
    the same collection: ``my_podcast__bge_speaker``.
    """
    return f"{_normalize_show(show)}__{strategy}"


# ──────────────────────────────────────────────
# QdrantStore
# ──────────────────────────────────────────────


class QdrantStore:
    """
    Thin wrapper around QdrantClient for podcast RAG storage.

    Args:
        url : Qdrant server URL. Defaults to QDRANT_URL env var or localhost:6333.
    """

    DEFAULT_URL = "http://localhost:6333"

    def __init__(self, url: str | None = None, in_memory: bool = False):
        from qdrant_client import QdrantClient

        if in_memory:
            self._client = QdrantClient(":memory:")
            self._url = ":memory:"
        else:
            self._url = url or os.environ.get("QDRANT_URL", self.DEFAULT_URL)
            self._client = QdrantClient(url=self._url)
        logger.info(f"QdrantStore connected to {self._url}")

    # ── Collection management ──────────────────

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def create_collection(
        self, name: str, strategy: str, overwrite: bool = False
    ) -> None:
        """
        Create a Qdrant collection configured for the given strategy.

        Args:
            name      : collection name (use collection_name() helper)
            strategy  : one of STRATEGIES
            overwrite : if True, delete and recreate an existing collection
        """
        if strategy not in _STRATEGY_CONFIG:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Choose from {STRATEGIES}."
            )

        cfg = _STRATEGY_CONFIG[strategy]

        if self.collection_exists(name):
            if overwrite:
                self._client.delete_collection(name)
                logger.info(f"Deleted existing collection '{name}'")
            else:
                logger.info(f"Collection '{name}' already exists — skipping creation")
                return

        from qdrant_client.models import (
            Distance,
            SparseIndexParams,
            SparseVectorParams,
            VectorParams,
        )

        if cfg["type"] == "dense":
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=cfg["dim"], distance=Distance.COSINE),
            )
        else:  # hybrid
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(size=cfg["dim"], distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(full_scan_threshold=5000)
                    )
                },
            )

        logger.success(f"Created collection '{name}' ({strategy})")

    def delete_collection(self, name: str) -> None:
        """Delete a collection by name."""
        self._client.delete_collection(name)
        logger.info(f"Deleted collection '{name}'")

    def list_collections(self, show: str = "") -> list[str]:
        """
        List all collection names, optionally filtered by show prefix.

        Args:
            show : if provided, only return collections for this show
        """
        all_names = [c.name for c in self._client.get_collections().collections]
        if show:
            prefix = f"{_normalize_show(show)}__"
            return [n for n in all_names if n.startswith(prefix)]
        return all_names

    # ── Upsert ────────────────────────────────

    def upsert(
        self,
        collection: str,
        chunks: list[dict],
        embeddings: Union[np.ndarray, dict],
        batch_size: int = 64,
    ) -> None:
        """
        Upsert chunks with their embeddings into a collection.

        Args:
            collection : target collection name
            chunks     : list of chunk dicts (payload stored as-is)
            embeddings : np.ndarray (dense-only) or dict {"dense": ..., "sparse": [...]}
            batch_size : number of points per upsert request
        """
        from qdrant_client.models import PointStruct, SparseVector

        is_hybrid = isinstance(embeddings, dict)
        points = []

        for i, chunk in enumerate(chunks):
            payload = _serialize_payload(chunk)
            point_id = str(uuid.uuid4())

            if is_hybrid:
                dense = embeddings["dense"][i].tolist()
                sv = embeddings["sparse"][i]
                vector = {
                    "dense": dense,
                    "sparse": SparseVector(indices=sv["indices"], values=sv["values"]),
                }
            else:
                vector = embeddings[i].tolist()

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self._client.upsert(collection_name=collection, points=batch)

        logger.success(f"Upserted {len(points)} points into '{collection}'")

    def fetch_episode_chunks(self, collection: str, episode: str) -> list[dict]:
        """
        Fetch all chunks for a given episode, sorted by start time.

        Used by the Discord bot's "Show more" button to retrieve surrounding context.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        results, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="episode", match=MatchValue(value=episode))]
            ),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        chunks = [dict(r.payload) for r in results]
        chunks.sort(key=lambda c: c.get("start", 0.0))
        return chunks


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _serialize_payload(chunk: dict) -> dict:
    """
    Convert a chunk dict to a JSON-safe payload for Qdrant.
    Converts numpy scalars/arrays to Python native types.
    """
    out = {}
    for k, v in chunk.items():
        if isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list):
            out[k] = [_serialize_payload(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out
