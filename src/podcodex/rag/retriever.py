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
from collections import defaultdict
from functools import lru_cache

import numpy as np
from loguru import logger

from podcodex.rag.defaults import DEFAULT_MODEL
from podcodex.rag.hit import Hit
from podcodex.rag.index_store import IndexStore, get_index_store


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

    def encode_query(self, query: str) -> np.ndarray:
        """Return the float32 query vector (kept separate so callers can hoist
        the encode across multiple collections)."""
        return self.embedder.encode_query(query).astype(np.float32)

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        alpha: float = 0.5,
        episode: str | None = None,
        episodes: list[str] | None = None,
        source: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
        query_vector: np.ndarray | None = None,
    ) -> list[Hit]:
        """Return the top_k most relevant chunks for a query.

        Args:
            query: Natural-language query.
            collection: Collection name.
            top_k: Number of results to return.
            alpha: Blend between FTS (0.0) and dense (1.0). Out-of-range
                values are clamped to ``[0, 1]``.
            episode, source, speaker: Optional equality filters.
            episodes: Alternative to ``episode`` — restrict to a list of stems.
            pub_date_min, pub_date_max: Inclusive date bounds (``YYYY-MM-DD``).
            query_vector: Precomputed query embedding. If provided, skips the
                embedder call — lets callers that fan out over N collections
                encode once.

        Returns:
            Hits with ``score`` set.
        """
        if alpha >= 1.0:
            return self._dense(
                query,
                collection,
                top_k,
                episode,
                episodes,
                source,
                speaker,
                pub_date_min,
                pub_date_max,
                query_vector,
            )
        if alpha <= 0.0:
            return self._fts(
                query,
                collection,
                top_k,
                episode,
                episodes,
                source,
                speaker,
                pub_date_min,
                pub_date_max,
            )
        return self._weighted(
            query,
            collection,
            top_k,
            alpha,
            episode,
            episodes,
            source,
            speaker,
            pub_date_min,
            pub_date_max,
            query_vector,
        )

    def exact(
        self,
        query: str,
        collection: str,
        episode: str | None = None,
        episodes: list[str] | None = None,
        source: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> list[Hit]:
        """Three-tier phrase search: exact (1.0), accent variant (0.8), near-typo (0.6).

        FTS ~2 pre-filter for speed; Python phrase checks for tier classification.
        Returns all matches — no top_k cap.

        When ``speaker`` is set, the filter is applied at the *turn* level
        rather than the chunk's ``dominant_speaker``: a chunk is kept only
        if the target speaker has a turn that contains the matched phrase.
        Fuzzy-tier matches are dropped in that mode (too approximate to
        attribute to a single turn).
        """
        # Don't narrow by dominant_speaker at the DB level — the turn-level
        # filter below would miss chunks where the target speaker is not
        # dominant but actually utters the phrase.
        exact, accent_only, fuzzy_only = self._local.search_literal(
            collection,
            query,
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=None,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
        if speaker:
            from podcodex.rag.index_store import fold_text

            q_lower = query.lower()
            q_folded = fold_text(query)

            def _keep(chunks: list[Hit], needle: str, folded: bool) -> list[Hit]:
                out: list[Hit] = []
                for c in chunks:
                    for t in c.speakers or []:
                        if t.speaker != speaker:
                            continue
                        hay = fold_text(t.text) if folded else t.text.lower()
                        if needle in hay:
                            out.append(c)
                            break
                return out

            exact = _keep(exact, q_lower, folded=False)
            accent_only = _keep(accent_only, q_folded, folded=True)
            fuzzy_only = []
        # Tier hits are freshly built by search_literal: flag in place.
        for c in accent_only:
            c.accent_match = True
        for c in fuzzy_only:
            c.fuzzy_match = True
        return exact + accent_only + fuzzy_only

    def exact_counts(
        self,
        queries: list[str],
        collection: str,
        *,
        group_by: str = "episode",
        first_hit: bool = False,
        episode: str | None = None,
        episodes: list[str] | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> dict[str, dict[str, dict | int]]:
        """Count literal matches per query, grouped by episode or show.

        No chunk text is marshalled. ``group_by`` is ``"episode"`` (default)
        or ``"show"``. Empty groups are omitted, so a query with zero hits
        yields ``{}``. With ``first_hit`` each group value becomes
        ``{"count": n, "first": {"chunk_index", "start", "start_hms"}}`` where
        ``first`` is the earliest hit by ``start``.

        Fuzzy near-typo hits are excluded (accent-variant hits are kept):
        counts feed recurrence and anti-contamination decisions, where a
        single-edit collision (``"chunk 0"`` matching ``"chunk 1"``) is noise
        that would defeat the check.

        Cost is ``O(len(queries))`` full three-tier scans of the collection:
        distinct phrases share no matching work, so a large ``queries`` batch
        is not free.
        """
        from podcodex.core._utils import format_hms

        key = "episode" if group_by == "episode" else "show"
        out: dict[str, dict[str, dict | int]] = {}
        for q in queries:
            hits = self.exact(
                q,
                collection,
                episode=episode,
                episodes=episodes,
                speaker=speaker,
                pub_date_min=pub_date_min,
                pub_date_max=pub_date_max,
            )
            groups: dict[str, dict | int] = {}
            for h in hits:
                if h.fuzzy_match:
                    continue
                g = getattr(h, key)
                if not g:
                    continue
                if first_hit:
                    start = h.start
                    entry = {
                        "chunk_index": h.chunk_index,
                        "start": start,
                        "start_hms": format_hms(start),
                    }
                    cur = groups.get(g)
                    if cur is None:
                        groups[g] = {"count": 1, "first": entry}
                    else:
                        cur["count"] += 1
                        if start < cur["first"]["start"]:
                            cur["first"] = entry
                else:
                    groups[g] = int(groups.get(g, 0)) + 1
            out[q] = groups
        return out

    def random(
        self,
        collection: str,
        episode: str | None = None,
        episodes: list[str] | None = None,
        source: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> Hit | None:
        """Return a single random chunk (with optional per-speaker refinement).

        When the selected chunk has multiple speaker turns and a speaker
        filter is set, the hit is narrowed to a single turn from that
        speaker: ``speaker``/``text``/``start``/``end`` come from the turn
        and ``speakers`` holds only it.

        Args:
            collection: Collection name.
            episode, source, speaker: Optional equality filters.
            episodes: Alternative to ``episode`` — restrict to a list of stems.
            pub_date_min, pub_date_max: Inclusive date bounds (``YYYY-MM-DD``).
        """
        # Count + offset instead of loading (and validating) the whole
        # collection to keep one row. Filters are pushed into the SQL clause;
        # stored rows never carry a flat `speaker`, so the dominant_speaker
        # clause matches the previous Python-side coalesce.
        filters = dict(
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
        # Count and fetch are two reads, so an out-of-process index change
        # between them (rsync onto a running bot, a reindex in the desktop app)
        # can shrink the table and leave the offset past the end. Re-count and
        # retry rather than reporting "no excerpts" on a populated collection.
        chunk = None
        for _ in range(3):
            n = self._local.count_chunks(collection, **filters)
            if n == 0:
                return None
            chunk = self._local.chunk_at(collection, random.randrange(n), **filters)
            if chunk is not None:
                break
        if chunk is None:
            return None
        turns = chunk.speakers or []
        if len(turns) > 1:
            if speaker:
                matching = [t for t in turns if t.speaker == speaker]
                turns = matching or turns
            turn = random.choice(turns)
            # Absence stays absence: an unnamed turn keeps its empty speaker
            # so the display layer (display_speaker) renders it, rather than
            # baking a raw "Unknown" sentinel into the data.
            return chunk.model_copy(
                update={
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "start": turn.start or chunk.start,
                    "end": turn.end or chunk.end,
                    "speakers": [turn],
                }
            )
        return chunk

    # ── Internals ────────────────────────────────────────────────────────

    def _dense(
        self,
        query: str,
        collection: str,
        top_k: int,
        episode: str | None,
        episodes: list[str] | None,
        source: str | None,
        speaker: str | None,
        pub_date_min: str | None,
        pub_date_max: str | None,
        query_vector: np.ndarray | None = None,
    ) -> list[Hit]:
        qv = (
            query_vector
            if query_vector is not None
            else self.embedder.encode_query(query).astype(np.float32)
        )
        hits = self._local.search_vector(
            collection,
            qv,
            top_k,
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
        return [h for h in hits if h.score >= 0.01]

    def _fts(
        self,
        query: str,
        collection: str,
        top_k: int,
        episode: str | None,
        episodes: list[str] | None,
        source: str | None,
        speaker: str | None,
        pub_date_min: str | None,
        pub_date_max: str | None,
    ) -> list[Hit]:
        hits = self._local.search_fts(
            collection,
            query,
            top_k,
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
        return _rank_normalize([h for h in hits if h.score > 1e-6])

    def _weighted(
        self,
        query: str,
        collection: str,
        top_k: int,
        alpha: float,
        episode: str | None,
        episodes: list[str] | None,
        source: str | None,
        speaker: str | None,
        pub_date_min: str | None,
        pub_date_max: str | None,
        query_vector: np.ndarray | None = None,
    ) -> list[Hit]:
        """Linear blend of rank-normalized dense and FTS scores."""
        k = top_k * 4
        dense_hits = _rank_normalize(
            self._dense(
                query,
                collection,
                k,
                episode,
                episodes,
                source,
                speaker,
                pub_date_min,
                pub_date_max,
                query_vector,
            )
        )
        fts_hits = self._fts(
            query,
            collection,
            k,
            episode,
            episodes,
            source,
            speaker,
            pub_date_min,
            pub_date_max,
        )

        combined: dict[str, float] = {}
        payloads: dict[str, Hit] = {}
        for r in dense_hits:
            key = _chunk_key(r)
            combined[key] = alpha * r.score
            payloads[key] = r
        for r in fts_hits:
            key = _chunk_key(r)
            combined[key] = combined.get(key, 0.0) + (1 - alpha) * r.score
            payloads.setdefault(key, r)

        sorted_keys = sorted(combined, key=combined.__getitem__, reverse=True)[:top_k]
        out = []
        for k in sorted_keys:
            payloads[k].score = combined[k]
            out.append(payloads[k])
        return out


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _chunk_key(chunk: Hit) -> str:
    """Deduplication key for merging dense + FTS hits."""
    return f"{chunk.show}|{chunk.episode}|{chunk.start}"


def _rank_normalize(results: list[Hit]) -> list[Hit]:
    """Assign each hit a rank-based score in ``[1/n, 1]`` (top = 1.0).

    Mutates in place: every caller passes freshly built, unshared hits.
    """
    n = len(results)
    for i, r in enumerate(results):
        r.score = 1.0 - (i / n)
    return results


@lru_cache(maxsize=4)
def _get_retriever_cached(model: str) -> Retriever:
    return Retriever(model=model, local=get_index_store())


def get_retriever(model: str = DEFAULT_MODEL) -> Retriever:
    """Process-wide cached Retriever for a given model.

    Shared by the desktop API, MCP server, and anything else that wants a
    hybrid retriever against the default IndexStore. Bot instances that need
    a custom index path keep their own Retriever cache.

    The model is resolved before the cache lookup so ``get_retriever()`` and
    ``get_retriever(DEFAULT_MODEL)`` hit the same entry (a bare ``lru_cache``
    keys on the literal call args and would build two embedders).
    """
    return _get_retriever_cached(model)


# Tests and reindex flows clear the embedder cache through this name.
get_retriever.cache_clear = _get_retriever_cached.cache_clear  # type: ignore[attr-defined]


def merge_results(
    hits_by_collection: dict[str, list[Hit]],
    top_k: int,
    strategy: str = "roundrobin",
) -> list[tuple[Hit, str]]:
    """Merge per-collection hits into a ranked list of ``(chunk, collection)``.

    Strategies:
      - ``"score"``      — global sort by score, slice to top_k. Prone to one
                           dominant collection flooding the output.
      - ``"roundrobin"`` — interleave one result per collection in score order.
                           Ensures diversity across collections (default).
    """
    if strategy == "score":
        all_hits = [
            (chunk, col)
            for col, chunks in hits_by_collection.items()
            for chunk in chunks
        ]
        all_hits.sort(key=lambda x: x[0].score or 0.0, reverse=True)
        return all_hits[:top_k]

    sorted_cols: dict[str, list[Hit]] = {
        col: sorted(chunks, key=lambda c: c.score or 0.0, reverse=True)
        for col, chunks in hits_by_collection.items()
    }
    result: list[tuple[Hit, str]] = []
    queues = list(sorted_cols.items())
    idx = defaultdict(int)

    while len(result) < top_k:
        advanced = False
        for col, chunks in queues:
            if len(result) >= top_k:
                break
            i = idx[col]
            if i < len(chunks):
                result.append((chunks[i], col))
                idx[col] += 1
                advanced = True
        if not advanced:
            break  # all collections exhausted

    return result
