"""
podcodex.rag.store — Qdrant storage layer for podcast RAG.

One collection per (show, model) pair.
Collection naming: "{normalized_show}__{model_key}"  e.g. "my_podcast__bge-m3"

Each collection stores dense float32 vectors; vector size comes from the
model's spec in podcodex.rag.defaults.MODELS.

The Qdrant URL defaults to QDRANT_URL env var, falling back to localhost:6333.
"""

from __future__ import annotations

import os
import re
import uuid

import numpy as np
from loguru import logger


def _normalize_show(show: str) -> str:
    """Lowercase and collapse non-alphanumeric runs to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", show.lower()).strip("_")


def collection_name(show: str, model: str, chunker: str = "semantic") -> str:
    """Build the canonical collection name: {normalized_show}__{model}__{chunker}."""
    return f"{_normalize_show(show)}__{model}__{chunker}"


class QdrantStore:
    """
    Thin wrapper around QdrantClient for podcast RAG storage.

    Args:
        url : Qdrant server URL. Defaults to QDRANT_URL env var or localhost:6333.
    """

    DEFAULT_URL = "http://localhost:6333"

    def __init__(self, url: str | None = None):
        from qdrant_client import QdrantClient

        self._url = url or os.environ.get("QDRANT_URL", self.DEFAULT_URL)
        self._client = QdrantClient(url=self._url)
        logger.info(f"QdrantStore connected to {self._url}")

    # ── Collection management ──────────────────

    def collection_exists(self, name: str) -> bool:
        return self._client.collection_exists(name)

    def create_collection(
        self, name: str, model: str = "bge-m3", overwrite: bool = False
    ) -> None:
        from podcodex.rag.defaults import MODELS
        from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

        spec = MODELS.get(model)
        if spec is None:
            valid = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid: {valid}")

        if self.collection_exists(name):
            if overwrite:
                self._client.delete_collection(name)
                logger.info(f"Deleted existing collection '{name}'")
            else:
                logger.info(f"Collection '{name}' already exists — skipping creation")
                return

        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=spec.dim, distance=Distance.COSINE),
        )

        # Full-text index on 'text' enables /find (case-insensitive MatchText)
        self._client.create_payload_index(
            collection_name=name,
            field_name="text",
            field_schema=PayloadSchemaType.TEXT,
        )

        logger.success(f"Created collection '{name}' ({spec.label}, dim={spec.dim})")

    def delete_collection(self, name: str) -> None:
        """Delete a collection by name."""
        self._client.delete_collection(name)
        logger.info(f"Deleted collection '{name}'")

    def list_collections(
        self, show: str = "", model: str = "", chunker: str = ""
    ) -> list[str]:
        """
        List collection names, optionally filtered by show, model, or chunker.

        Collection format: {show}__{model}__{chunker}
        """
        all_names = [c.name for c in self._client.get_collections().collections]
        if show:
            prefix = _normalize_show(show) + "__"
            all_names = [n for n in all_names if n.startswith(prefix)]
        if model:
            all_names = [n for n in all_names if f"__{model}__" in n]
        if chunker:
            all_names = [n for n in all_names if n.endswith(f"__{chunker}")]
        return all_names

    # ── Upsert ────────────────────────────────

    def upsert(
        self,
        collection: str,
        chunks: list[dict],
        embeddings: np.ndarray,
        batch_size: int = 64,
    ) -> None:
        """
        Upsert chunks with their dense embeddings into a collection.

        Args:
            collection  : target collection name
            chunks      : list of chunk dicts (stored as payload)
            embeddings  : np.ndarray of shape (n, dim), float32
            batch_size  : number of points per upsert request
        """
        from qdrant_client.models import PointStruct

        points = []
        for i, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i].tolist(),
                    payload=_serialize_payload(chunk),
                )
            )

        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=collection, points=points[start : start + batch_size]
            )

        logger.success(f"Upserted {len(points)} points into '{collection}'")

    def fetch_episode_chunks(self, collection: str, episode: str) -> list[dict]:
        """
        Fetch all chunks for a given episode, sorted by start time.
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

    def get_episode_stats(self, collection: str) -> list[dict]:
        """
        Aggregate per-episode stats by scrolling all points in a collection.
        Returns list of {episode, chunk_count, duration} sorted by episode.
        Uses paginated scroll to handle arbitrarily large collections.
        """
        stats: dict[str, dict] = {}
        offset = None

        while True:
            batch, next_offset = self._client.scroll(
                collection_name=collection,
                limit=1_000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in batch:
                p = point.payload or {}
                ep = p.get("episode", "(unknown)")
                end = float(p.get("end", 0.0))
                if ep not in stats:
                    stats[ep] = {"episode": ep, "chunk_count": 0, "duration": 0.0}
                stats[ep]["chunk_count"] += 1
                stats[ep]["duration"] = max(stats[ep]["duration"], end)

            if next_offset is None:
                break
            offset = next_offset

        return sorted(stats.values(), key=lambda x: x["episode"])


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _serialize_payload(chunk: dict) -> dict:
    """Convert a chunk dict to a JSON-safe payload for Qdrant."""
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
