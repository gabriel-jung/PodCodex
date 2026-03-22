"""
podcodex.rag.store — Qdrant storage layer for podcast RAG.

One collection per (show, model, chunker) triple.
Collection naming: "{normalized_show}__{model}__{chunker}"
    e.g. "my_podcast__bge-m3__semantic"

Each collection stores dense float32 vectors; vector size comes from the
model's spec in podcodex.rag.defaults.MODELS.

The Qdrant URL defaults to QDRANT_URL env var, falling back to localhost:6333.
"""

from __future__ import annotations

import os
import random
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


def qdrant_available(url: str | None = None, timeout: float = 2.0) -> bool:
    """Return True if the Qdrant server is reachable."""
    import urllib.request
    import urllib.error

    target = url or os.environ.get("QDRANT_URL", QdrantStore.DEFAULT_URL)
    try:
        urllib.request.urlopen(f"{target}/readyz", timeout=timeout)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _episode_filter(episode: str):
    """Build a Qdrant Filter matching a single episode."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(must=[FieldCondition(key="episode", match=MatchValue(value=episode))])


def _search_filter(
    episode: str | None = None,
    source: str | None = None,
    speaker: str | None = None,
):
    """Build a Qdrant Filter for optional episode + source + speaker constraints."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    conditions = []
    if episode:
        conditions.append(
            FieldCondition(key="episode", match=MatchValue(value=episode))
        )
    if source:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))
    if speaker:
        conditions.append(
            FieldCondition(key="speaker", match=MatchValue(value=speaker))
        )
    return Filter(must=conditions) if conditions else None


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
        """Return True if the collection exists in Qdrant."""
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
        # Keyword index on 'episode' speeds up filtered counts, deletes, and searches
        self._client.create_payload_index(
            collection_name=name,
            field_name="episode",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        # Keyword index on 'source' enables filtering by transcript quality
        self._client.create_payload_index(
            collection_name=name,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        # Keyword index on 'speaker' enables filtering by speaker
        self._client.create_payload_index(
            collection_name=name,
            field_name="speaker",
            field_schema=PayloadSchemaType.KEYWORD,
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

    # ── Query ───────────────────────────────────

    def collection_point_count(self, collection: str) -> int:
        """Return total number of points in a collection (cheap, from metadata)."""
        return self._client.get_collection(collection).points_count

    def search_points(
        self, collection: str, query_vector: list[float], top_k: int, query_filter=None
    ) -> list[dict]:
        """Dense vector search. Returns list of {payload fields + score}."""
        results = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points
        return [_result_to_dict(r) for r in results]

    def random_point(self, collection: str, scroll_filter=None) -> dict | None:
        """Return a single random payload from the collection, or None if empty."""
        count = self._client.count(
            collection_name=collection,
            count_filter=scroll_filter,
            exact=True,
        ).count
        if count == 0:
            return None

        # Scroll to a random offset and grab one point
        offset = random.randint(0, count - 1)
        results, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=1,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        return dict(results[0].payload)

    def scroll_payloads(
        self, collection: str, scroll_filter=None, limit: int = 10_000
    ) -> list[dict]:
        """Scroll all payloads matching a filter. Returns list of payload dicts."""
        results, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [dict(r.payload) for r in results]

    # ── Episode-level ──────────────────────────

    def episode_point_count(self, collection: str, episode: str) -> int:
        """Return the number of points for an episode in a collection (cheap count)."""
        result = self._client.count(
            collection_name=collection,
            count_filter=_episode_filter(episode),
            exact=True,
        )
        return result.count

    def delete_episode_points(self, collection: str, episode: str) -> int:
        """Delete all points for an episode. Returns count deleted."""
        from qdrant_client.models import FilterSelector

        count = self.episode_point_count(collection, episode)
        if count == 0:
            return 0

        self._client.delete(
            collection_name=collection,
            points_selector=FilterSelector(filter=_episode_filter(episode)),
        )
        logger.info(f"Deleted {count} points for '{episode}' from '{collection}'")
        return count

    def fetch_episode_chunks(self, collection: str, episode: str) -> list[dict]:
        """Fetch all chunks for a given episode, sorted by start time."""
        chunks = self.scroll_payloads(
            collection, scroll_filter=_episode_filter(episode)
        )
        chunks.sort(key=lambda c: c.get("start", 0.0))
        return chunks

    def list_episode_names(self, collection: str) -> list[str]:
        """Return sorted unique episode names in a collection (paginated scroll)."""
        names: set[str] = set()
        offset = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=collection,
                limit=1_000,
                offset=offset,
                with_payload=["episode"],
                with_vectors=False,
            )
            for point in batch:
                ep = (point.payload or {}).get("episode")
                if ep:
                    names.add(ep)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(names)

    def list_sources(self, collection: str) -> list[str]:
        """Return sorted unique source values in a collection (paginated scroll)."""
        sources: set[str] = set()
        offset = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=collection,
                limit=1_000,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for point in batch:
                src = (point.payload or {}).get("source")
                if src:
                    sources.add(src)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(sources)

    def list_speakers(self, collection: str) -> list[str]:
        """Return sorted unique speaker names in a collection (paginated scroll)."""
        speakers: set[str] = set()
        offset = None
        while True:
            batch, next_offset = self._client.scroll(
                collection_name=collection,
                limit=1_000,
                offset=offset,
                with_payload=["speaker"],
                with_vectors=False,
            )
            for point in batch:
                spk = (point.payload or {}).get("speaker")
                if spk:
                    speakers.add(spk)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(speakers)

    def get_episode_stats(self, collection: str) -> list[dict]:
        """
        Aggregate per-episode stats by scrolling all points in a collection.
        Returns list of {episode, chunk_count, duration, speakers} sorted by episode.
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
                    stats[ep] = {
                        "episode": ep,
                        "chunk_count": 0,
                        "duration": 0.0,
                        "speakers": set(),
                    }
                stats[ep]["chunk_count"] += 1
                stats[ep]["duration"] = max(stats[ep]["duration"], end)
                # Collect speakers from chunk-level or turn-level data
                spk = p.get("speaker") or p.get("dominant_speaker")
                if spk:
                    stats[ep]["speakers"].add(spk)
                for turn in p.get("speakers") or []:
                    s = turn.get("speaker")
                    if s:
                        stats[ep]["speakers"].add(s)

            if next_offset is None:
                break
            offset = next_offset

        # Convert sets to sorted lists for JSON safety
        for ep_stats in stats.values():
            ep_stats["speakers"] = sorted(ep_stats["speakers"])

        return sorted(stats.values(), key=lambda x: x["episode"])


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────


def _result_to_dict(result) -> dict:
    """Convert a Qdrant ScoredPoint to a plain dict with a 'score' key."""
    payload = dict(result.payload or {})
    payload["score"] = result.score
    return payload


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
