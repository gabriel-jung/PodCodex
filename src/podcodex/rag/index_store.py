"""
podcodex.rag.index_store — LanceDB-backed retrieval index.

A single embedded LanceDB database holds one table per collection
(``{show}__{model}__{chunker}``). Each row is fully self-contained: it
carries every piece of metadata the bot, API, or MCP server needs to
render a search result, so the index directory can be rsynced to a VPS
and served without any sibling database.

Collection-level metadata (show, model, chunker, dim) is tracked in a
sidecar ``_collections`` table.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from loguru import logger

_COLLECTIONS_TABLE = "_collections"

DEFAULT_INDEX_PATH: Path = Path(
    os.environ.get(
        "PODCODEX_INDEX",
        str(Path.home() / ".local" / "share" / "podcodex" / "index"),
    )
)


def _chunk_schema(dim: int) -> pa.Schema:
    """Arrow schema for a collection table with the given vector dimensionality."""
    return pa.schema(
        [
            pa.field("chunk_index", pa.int32()),
            pa.field("show", pa.string()),
            pa.field("episode", pa.string()),
            pa.field("source", pa.string()),
            pa.field("dominant_speaker", pa.string()),
            pa.field("start", pa.float64()),
            pa.field("end", pa.float64()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("meta", pa.string()),
        ]
    )


_COLLECTIONS_SCHEMA = pa.schema(
    [
        pa.field("name", pa.string()),
        pa.field("show", pa.string()),
        pa.field("model", pa.string()),
        pa.field("chunker", pa.string()),
        pa.field("dim", pa.int32()),
        pa.field("created_at", pa.string()),
    ]
)


_RESERVED_META_KEYS = {
    "chunk_index",
    "show",
    "episode",
    "source",
    "dominant_speaker",
    "start",
    "end",
    "text",
    "vector",
    "meta",
}


class IndexStore:
    """LanceDB-backed store for chunk text, metadata, and embeddings.

    Args:
        path: Directory holding the LanceDB database. Defaults to
            ``DEFAULT_INDEX_PATH`` (``PODCODEX_INDEX`` env or
            ``~/.local/share/podcodex/index``).
    """

    def __init__(self, path: Path | str | None = None):
        import lancedb

        if path is None:
            path = DEFAULT_INDEX_PATH
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._path))
        self._fts_ready: set[str] = set()
        logger.debug(f"IndexStore opened: {self._path}")

    # ── Internal: collections metadata table ─────────────────────────────

    def _table_names(self) -> list[str]:
        """Return LanceDB table names as a plain list."""
        # Some LanceDB versions wrap the names in a response object with a
        # ``.tables`` attribute; fall back to iterating the result directly.
        result = self._db.list_tables()
        return list(getattr(result, "tables", result))

    def _collections_table(self):
        if _COLLECTIONS_TABLE in self._table_names():
            return self._db.open_table(_COLLECTIONS_TABLE)
        return self._db.create_table(_COLLECTIONS_TABLE, schema=_COLLECTIONS_SCHEMA)

    def _table(self, name: str):
        """Open an existing collection table; raise if missing."""
        if name not in self._table_names():
            raise KeyError(f"Collection '{name}' does not exist")
        return self._db.open_table(name)

    # ── Collection management ────────────────────────────────────────────

    def collection_exists(self, name: str) -> bool:
        """Return True if a collection with the given name exists."""
        return name in self._table_names()

    def ensure_collection(
        self, name: str, show: str, model: str, chunker: str, dim: int
    ) -> None:
        """Create the collection table + metadata row if missing (idempotent).

        Args:
            name: Canonical collection name (see ``store.collection_name``).
            show: Human-readable show name.
            model: Embedding model key.
            chunker: Chunking strategy key.
            dim: Embedding vector dimensionality.
        """
        tables = self._table_names()
        if name not in tables:
            self._db.create_table(name, schema=_chunk_schema(dim))

        meta = self._collections_table()
        existing = meta.search().where(f"name = '{name}'").limit(1).to_list()
        if not existing:
            meta.add(
                [
                    {
                        "name": name,
                        "show": show,
                        "model": model,
                        "chunker": chunker,
                        "dim": dim,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            )

    def list_collections(
        self, show: str = "", model: str = "", chunker: str = ""
    ) -> list[str]:
        """List collection names, optionally filtered by show/model/chunker.

        Args:
            show: Exact match filter (empty = no filter).
            model: Embedding model key filter.
            chunker: Chunking strategy key filter.

        Returns:
            Alphabetically sorted list of matching collection names.
        """
        clauses = []
        if show:
            clauses.append(f"show = '{_escape(show)}'")
        if model:
            clauses.append(f"model = '{_escape(model)}'")
        if chunker:
            clauses.append(f"chunker = '{_escape(chunker)}'")
        q = self._collections_table().search().select(["name"])
        if clauses:
            q = q.where(" AND ".join(clauses))
        return sorted(r["name"] for r in q.limit(10_000).to_list())

    def delete_collection(self, name: str) -> None:
        """Drop a collection table and its metadata row.

        Args:
            name: Collection name to delete.
        """
        if name in self._table_names():
            self._db.drop_table(name)
        meta = self._collections_table()
        meta.delete(f"name = '{_escape(name)}'")
        logger.debug(f"Deleted collection '{name}'")

    def get_collection_info(self, name: str) -> dict | None:
        """Return ``{show, model, chunker, dim}`` or ``None`` if unknown.

        Args:
            name: Collection name.
        """
        rows = (
            self._collections_table()
            .search()
            .where(f"name = '{_escape(name)}'")
            .limit(1)
            .to_list()
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "show": r["show"],
            "model": r["model"],
            "chunker": r["chunker"],
            "dim": int(r["dim"]),
        }

    # ── Episode-level ────────────────────────────────────────────────────

    def episode_is_indexed(self, collection: str, episode: str) -> bool:
        """Return True if at least one chunk exists for this episode.

        Args:
            collection: Collection name.
            episode: Episode identifier.
        """
        if not self.collection_exists(collection):
            return False
        t = self._table(collection)
        rows = t.search().where(f"episode = '{_escape(episode)}'").limit(1).to_list()
        return bool(rows)

    def episode_chunk_count(self, collection: str, episode: str) -> int:
        """Return the number of chunks for an episode in a collection.

        Args:
            collection: Collection name.
            episode: Episode identifier.
        """
        if not self.collection_exists(collection):
            return 0
        t = self._table(collection)
        return int(t.count_rows(filter=f"episode = '{_escape(episode)}'"))

    def episode_source(self, collection: str, episode: str) -> str:
        """Return the source step (transcript/corrected/<lang>) for an
        episode's chunks in this collection. Empty string if not indexed.
        Episodes are always indexed from a single source per collection.
        """
        if not self.collection_exists(collection):
            return ""
        t = self._table(collection)
        rows = (
            t.search()
            .where(f"episode = '{_escape(episode)}'")
            .select(["source"])
            .limit(1)
            .to_list()
        )
        return str(rows[0].get("source", "")) if rows else ""

    def delete_episode(self, collection: str, episode: str) -> None:
        """Delete every chunk for an episode in a collection.

        Args:
            collection: Collection name.
            episode: Episode identifier.
        """
        if not self.collection_exists(collection):
            return
        t = self._table(collection)
        t.delete(f"episode = '{_escape(episode)}'")
        logger.debug(f"Deleted episode '{episode}' from '{collection}'")

    def list_episodes(self, collection: str) -> list[str]:
        """Return a sorted list of distinct episodes in the collection.

        Args:
            collection: Collection name.
        """
        return self._distinct(collection, "episode")

    # ── Write ────────────────────────────────────────────────────────────

    def save_chunks(
        self,
        collection: str,
        episode: str,
        chunks: list[dict],
        embeddings: np.ndarray,
    ) -> None:
        """Insert chunks and their embeddings for an episode.

        The caller is responsible for calling :meth:`delete_episode` first
        when overwriting — this method does not upsert.

        Args:
            collection: Target collection name (must already exist).
            episode: Episode identifier.
            chunks: List of chunk dicts, each with at least ``"text"``.
                Keys ``show``, ``source``, ``dominant_speaker``, ``start``,
                ``end`` are unpacked into filter columns; everything else
                is preserved inside the JSON ``meta`` column.
            embeddings: Float32 array of shape ``(n, dim)`` aligned with
                *chunks*.

        Raises:
            ValueError: If ``len(chunks) != len(embeddings)``.
            KeyError: If the collection does not exist.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        if not chunks:
            return
        t = self._table(collection)
        rows: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            speaker = chunk.get("dominant_speaker") or chunk.get("speaker") or ""
            meta = {
                k: v
                for k, v in chunk.items()
                if k not in _RESERVED_META_KEYS and k != "speaker"
            }
            rows.append(
                {
                    "chunk_index": int(i),
                    "show": str(chunk.get("show", "")),
                    "episode": str(chunk.get("episode", episode)),
                    "source": str(chunk.get("source", "")),
                    "dominant_speaker": str(speaker),
                    "start": float(chunk.get("start", 0.0)),
                    "end": float(chunk.get("end", 0.0)),
                    "text": str(chunk.get("text", "")),
                    "vector": embeddings[i].astype(np.float32).tolist(),
                    "meta": json.dumps(meta),
                }
            )
        t.add(rows)
        logger.debug(f"Saved {len(rows)} chunks for '{episode}' in '{collection}'")

    # ── Read helpers ─────────────────────────────────────────────────────

    def load_chunks_no_embeddings(self, collection: str, episode: str) -> list[dict]:
        """Return chunks for an episode without their vectors.

        Used by the indexer for stale-source detection.

        Args:
            collection: Collection name.
            episode: Episode identifier.

        Returns:
            List of chunk dicts sorted by ``chunk_index``.
        """
        if not self.collection_exists(collection):
            return []
        t = self._table(collection)
        rows = (
            t.search()
            .where(f"episode = '{_escape(episode)}'")
            .select(
                [
                    "chunk_index",
                    "show",
                    "episode",
                    "source",
                    "dominant_speaker",
                    "start",
                    "end",
                    "text",
                    "meta",
                ]
            )
            .limit(100_000)
            .to_list()
        )
        out = [_row_to_chunk(r) for r in rows]
        out.sort(key=lambda c: c.get("chunk_index", 0))
        return out

    def load_all_chunks(
        self, collection: str, episode: str | None = None
    ) -> list[dict]:
        """Load all chunks (without vectors) in a collection.

        Args:
            collection: Collection name.
            episode: If set, restrict to this one episode.

        Returns:
            List of chunk dicts ordered by ``(episode, chunk_index)``.
        """
        if not self.collection_exists(collection):
            return []
        t = self._table(collection)
        q = t.search()
        if episode:
            q = q.where(f"episode = '{_escape(episode)}'")
        rows = (
            q.select(
                [
                    "chunk_index",
                    "show",
                    "episode",
                    "source",
                    "dominant_speaker",
                    "start",
                    "end",
                    "text",
                    "meta",
                ]
            )
            .limit(1_000_000)
            .to_list()
        )
        out = [_row_to_chunk(r) for r in rows]
        out.sort(key=lambda c: (c.get("episode", ""), c.get("chunk_index", 0)))
        return out

    # ── Native search primitives ─────────────────────────────────────────

    def search_vector(
        self,
        collection: str,
        query_vec: np.ndarray,
        top_k: int,
        *,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        """ANN vector search with optional pre-filters.

        Args:
            collection: Collection name.
            query_vec: Query embedding (float32 1-D).
            top_k: Maximum results.
            episode, source, speaker: SQL-style equality filters.

        Returns:
            List of chunk dicts with cosine ``score`` (``1 - distance``)
            attached.
        """
        if not self.collection_exists(collection):
            return []
        t = self._table(collection)
        q = t.search(
            query_vec.astype(np.float32).tolist(),
            query_type="vector",
            vector_column_name="vector",
        )
        clause = _build_where(episode=episode, source=source, speaker=speaker)
        if clause:
            q = q.where(clause)
        rows = q.limit(top_k).to_list()
        out: list[dict] = []
        for r in rows:
            score = max(0.0, 1.0 - float(r.get("_distance", 1.0)))
            out.append({**_row_to_chunk(r), "score": score})
        return out

    def search_fts(
        self,
        collection: str,
        query: str,
        top_k: int,
        *,
        episode: str | None = None,
        source: str | None = None,
        speaker: str | None = None,
    ) -> list[dict]:
        """Full-text search (Tantivy, tokenized).

        Creates the FTS index lazily on first call per collection.

        Args:
            collection: Collection name.
            query: Free-text query.
            top_k: Maximum results.
            episode, source, speaker: SQL-style equality filters.

        Returns:
            List of chunk dicts with BM25 ``score`` attached.
        """
        if not self.collection_exists(collection):
            return []
        t = self._table(collection)
        self._ensure_fts(collection, t)
        q = t.search(query, query_type="fts")
        clause = _build_where(episode=episode, source=source, speaker=speaker)
        if clause:
            q = q.where(clause)
        try:
            rows = q.limit(top_k).to_list()
        except Exception:
            logger.opt(exception=True).debug("FTS query failed — treating as empty")
            return []
        return [
            {**_row_to_chunk(r), "score": float(r.get("_score", 0.0))} for r in rows
        ]

    def _ensure_fts(self, collection: str, table) -> None:
        """Ensure a Tantivy FTS index exists on the ``text`` column (memoized)."""
        if collection in self._fts_ready:
            return
        try:
            existing = {idx.name for idx in table.list_indices()}
        except Exception:
            existing = set()
        if not any("text" in n.lower() for n in existing):
            try:
                table.create_fts_index("text", replace=False)
            except Exception:
                logger.opt(exception=True).debug("FTS index creation skipped")
        self._fts_ready.add(collection)

    # ── Stats ────────────────────────────────────────────────────────────

    def collection_chunk_count(self, collection: str) -> int:
        """Return the total number of chunks in a collection.

        Args:
            collection: Collection name.
        """
        if not self.collection_exists(collection):
            return 0
        return int(self._table(collection).count_rows())

    def list_sources(self, collection: str) -> list[str]:
        """Return sorted distinct ``source`` values in a collection."""
        return self._distinct(collection, "source")

    def list_speakers(self, collection: str) -> list[str]:
        """Return sorted distinct ``dominant_speaker`` values in a collection."""
        return self._distinct(collection, "dominant_speaker")

    def get_episode_stats(self, collection: str) -> list[dict]:
        """Return per-episode ``{episode, chunk_count, duration, speakers}``."""
        if not self.collection_exists(collection):
            return []
        rows = (
            self._table(collection)
            .search()
            .select(["episode", "dominant_speaker", "end"])
            .limit(1_000_000)
            .to_list()
        )
        groups: dict[str, dict] = {}
        for r in rows:
            ep = r.get("episode")
            if ep is None:
                continue
            g = groups.setdefault(
                ep, {"chunk_count": 0, "duration": 0.0, "speakers": set()}
            )
            g["chunk_count"] += 1
            end = float(r.get("end") or 0.0)
            if end > g["duration"]:
                g["duration"] = end
            sp = r.get("dominant_speaker")
            if sp:
                g["speakers"].add(sp)
        return [
            {
                "episode": ep,
                "chunk_count": g["chunk_count"],
                "duration": g["duration"],
                "speakers": sorted(g["speakers"]),
            }
            for ep, g in sorted(groups.items())
        ]

    def _distinct(self, collection: str, column: str) -> list[str]:
        """Return sorted distinct non-empty values of one column."""
        if not self.collection_exists(collection):
            return []
        rows = (
            self._table(collection).search().select([column]).limit(1_000_000).to_list()
        )
        return sorted({r[column] for r in rows if r.get(column)})


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _escape(s: str) -> str:
    """Escape single quotes for LanceDB SQL WHERE clauses."""
    return s.replace("'", "''")


def _build_where(
    *,
    episode: str | None = None,
    source: str | None = None,
    speaker: str | None = None,
) -> str:
    """Build a LanceDB WHERE clause from equality filters."""
    parts: list[str] = []
    if episode:
        parts.append(f"episode = '{_escape(episode)}'")
    if source:
        parts.append(f"source = '{_escape(source)}'")
    if speaker:
        parts.append(f"dominant_speaker = '{_escape(speaker)}'")
    return " AND ".join(parts)


_FILTER_COLUMNS = (
    "chunk_index",
    "show",
    "episode",
    "source",
    "dominant_speaker",
    "start",
    "end",
    "text",
)


def _row_to_chunk(row: dict) -> dict:
    """Re-inflate a LanceDB row into a chunk dict.

    Drops the vector column and LanceDB internals (``_distance``, ``_score``,
    ``_relevance_score``); merges the ``meta`` JSON payload back in.
    """
    meta_raw = row.get("meta") or "{}"
    try:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
    except Exception:
        meta = {}
    chunk: dict[str, Any] = dict(meta)
    for key in _FILTER_COLUMNS:
        if key in row and row[key] is not None:
            chunk[key] = row[key]
    return chunk
