"""
podcodex.rag.retriever — Hybrid retriever over a LanceDB IndexStore.

Blends dense ANN cosine similarity with Tantivy FTS BM25 on the ``text``
column.

    alpha = 1.0  — dense only
    alpha = 0.0  — FTS only
    0 < alpha < 1 — linear blend of rank-normalized scores (default 0.5)
"""

from __future__ import annotations

import random

import numpy as np
from loguru import logger

from podcodex.rag.index_store import IndexStore


class Retriever:
    """Retriever over a :class:`IndexStore`.

    Args:
        model: Embedding model key from ``defaults.MODELS`` (default
            ``"bge-m3"``).
        local: IndexStore instance. A store at the default location is
            opened if ``None``.
        device: Torch device for the embedder (default ``"cpu"``).
    """

    def __init__(
        self,
        model: str = "bge-m3",
        local: IndexStore | None = None,
        device: str = "cpu",
    ):
        from podcodex.rag.defaults import MODELS

        if model not in MODELS:
            valid = ", ".join(MODELS.keys())
            raise ValueError(f"Unknown model '{model}'. Valid: {valid}")

        self._model_key = model
        self._device = device
        self._embedder = None  # loaded lazily on first retrieve()
        self._local = local or IndexStore()

    @property
    def embedder(self):
        if self._embedder is None:
            from podcodex.rag.embedder import get_embedder

            self._embedder = get_embedder(self._model_key, device=self._device)
            logger.info(f"Retriever ready (model={self._model_key})")
        return self._embedder

    # ── Public API ───────────────────────────────────────────────────────

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
        """Return the top_k most relevant chunks for a query.

        Args:
            query: Natural-language query.
            collection: Collection name.
            top_k: Number of results to return.
            alpha: Blend between FTS (0.0) and dense (1.0). Out-of-range
                values are clamped to ``[0, 1]``.
            episode, source, speaker: Optional equality filters.

        Returns:
            List of chunk dicts with a ``score`` key added.
        """
        if alpha >= 1.0:
            return self._dense(query, collection, top_k, episode, source, speaker)
        if alpha <= 0.0:
            return self._fts(query, collection, top_k, episode, source, speaker)
        return self._weighted(query, collection, top_k, alpha, episode, source, speaker)

    def find(
        self,
        query: str,
        collection: str,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        """Three-tier phrase search: exact (1.0), accent variant (0.8), near-typo (0.6).

        FTS ~2 pre-filter for speed; Python phrase checks for tier classification.
        Returns all matches — no top_k cap.
        """
        exact, accent_only, fuzzy_only = self._local.search_literal(
            collection,
            query,
            episode=episode,
            source=source,
            speaker=speaker,
        )
        return (
            exact
            + [{**c, "accent_match": True} for c in accent_only]
            + [{**c, "fuzzy_match": True} for c in fuzzy_only]
        )

    def random(
        self,
        collection: str,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> dict | None:
        """Return a single random chunk (with optional per-speaker refinement).

        When the selected chunk has multiple speaker turns and a speaker
        filter is set, a single turn from that speaker is returned as a
        flat chunk dict.

        Args:
            collection: Collection name.
            episode, source, speaker: Optional equality filters.
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

    # ── Internals ────────────────────────────────────────────────────────

    def _dense(
        self,
        query: str,
        collection: str,
        top_k: int,
        episode: str | None,
        source: str | None,
        speaker: str | None,
    ) -> list[dict]:
        qv = self.embedder.encode_query(query).astype(np.float32)
        hits = self._local.search_vector(
            collection,
            qv,
            top_k,
            episode=episode,
            source=source,
            speaker=speaker,
        )
        return [h for h in hits if h["score"] >= 0.01]

    def _fts(
        self,
        query: str,
        collection: str,
        top_k: int,
        episode: str | None,
        source: str | None,
        speaker: str | None,
    ) -> list[dict]:
        hits = self._local.search_fts(
            collection,
            query,
            top_k,
            episode=episode,
            source=source,
            speaker=speaker,
        )
        return _rank_normalize([h for h in hits if h["score"] > 1e-6])

    def _weighted(
        self,
        query: str,
        collection: str,
        top_k: int,
        alpha: float,
        episode: str | None,
        source: str | None,
        speaker: str | None,
    ) -> list[dict]:
        """Linear blend of rank-normalized dense and FTS scores."""
        k = top_k * 4
        dense_hits = _rank_normalize(
            self._dense(query, collection, k, episode, source, speaker)
        )
        fts_hits = self._fts(query, collection, k, episode, source, speaker)

        combined: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        for r in dense_hits:
            key = _chunk_key(r)
            combined[key] = alpha * r["score"]
            payloads[key] = r
        for r in fts_hits:
            key = _chunk_key(r)
            combined[key] = combined.get(key, 0.0) + (1 - alpha) * r["score"]
            payloads.setdefault(key, r)

        sorted_keys = sorted(combined, key=combined.__getitem__, reverse=True)[:top_k]
        return [{**payloads[k], "score": combined[k]} for k in sorted_keys]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _chunk_key(chunk: dict) -> str:
    """Deduplication key for merging dense + FTS hits."""
    return f"{chunk.get('show', '')}|{chunk.get('episode', '')}|{chunk.get('start', 0)}"


def _rank_normalize(results: list[dict]) -> list[dict]:
    """Assign each hit a rank-based score in ``[1/n, 1]`` (top = 1.0)."""
    n = len(results)
    if n == 0:
        return results
    return [{**r, "score": 1.0 - (i / n)} for i, r in enumerate(results)]
