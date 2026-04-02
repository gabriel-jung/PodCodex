"""
podcodex.rag.localstore — SQLite persistence layer for podcast RAG.

Stores chunk texts, metadata, and embeddings locally. Serves as the
single source of truth for indexed content — search is performed
directly over this store using numpy brute-force.

Schema (3 tables):
    collections  — one row per (show, model, chunker) triple
    chunks       — text + meta per chunk, ordered by chunk_index
    embeddings   — float32 blob, one row per chunk (separate to allow
                   metadata queries without pulling large blobs)
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

DEFAULT_DB_PATH: Path = Path(
    os.environ.get(
        "PODCODEX_DB",
        str(Path.home() / ".local" / "share" / "podcodex" / "vectors.db"),
    )
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    name       TEXT PRIMARY KEY,
    show       TEXT NOT NULL,
    model      TEXT NOT NULL,
    chunker    TEXT NOT NULL DEFAULT 'semantic',
    dim        INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    collection  TEXT    NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    episode     TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    meta_json   TEXT    NOT NULL,
    UNIQUE(collection, episode, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_collection_episode
    ON chunks(collection, episode);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id  INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vector    BLOB NOT NULL
);
"""


class LocalStore:
    """SQLite-backed store for chunk texts, metadata, and float32 embeddings.

    Args:
        db_path : Path to the SQLite database file.
                  Use ":memory:" for in-process tests.
                  Defaults to DEFAULT_DB_PATH (PODCODEX_DB env var or
                  ~/.local/share/podcodex/vectors.db).
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.debug(f"LocalStore opened: {self._path}")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "LocalStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Collection management ──────────────────────────────────────────

    def collection_exists(self, name: str) -> bool:
        """Return True if a collection with the given name exists."""
        cur = self._conn.execute("SELECT 1 FROM collections WHERE name=?", (name,))
        return cur.fetchone() is not None

    def ensure_collection(
        self, name: str, show: str, model: str, chunker: str, dim: int
    ) -> None:
        """Create collection row if it does not already exist (idempotent)."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO collections(name, show, model, chunker, dim, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, show, model, chunker, dim, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def list_collections(
        self, show: str = "", model: str = "", chunker: str = ""
    ) -> list[str]:
        """List collection names, optionally filtered by show, model, and/or chunker.

        Note: filters match the raw values stored in the collections table
        (exact match).
        """
        q = "SELECT name FROM collections WHERE 1=1"
        params: list = []
        if show:
            q += " AND show=?"
            params.append(show)
        if model:
            q += " AND model=?"
            params.append(model)
        if chunker:
            q += " AND chunker=?"
            params.append(chunker)
        q += " ORDER BY name"
        return [r[0] for r in self._conn.execute(q, params).fetchall()]

    def delete_collection(self, name: str) -> None:
        """Delete collection; ON DELETE CASCADE removes chunks + embeddings."""
        self._conn.execute("DELETE FROM collections WHERE name=?", (name,))
        self._conn.commit()
        logger.debug(f"Deleted collection '{name}'")

    # ── Episode-level ──────────────────────────────────────────────────

    def episode_is_indexed(self, collection: str, episode: str) -> bool:
        """Return True if at least one chunk exists for this episode in the collection."""
        cur = self._conn.execute(
            "SELECT 1 FROM chunks WHERE collection=? AND episode=? LIMIT 1",
            (collection, episode),
        )
        return cur.fetchone() is not None

    def episode_chunk_count(self, collection: str, episode: str) -> int:
        """Return the number of chunks for an episode in a collection."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE collection=? AND episode=?",
            (collection, episode),
        )
        return cur.fetchone()[0]

    def delete_episode(self, collection: str, episode: str) -> None:
        """Delete all chunks (and their embeddings via CASCADE) for an episode."""
        self._conn.execute(
            "DELETE FROM chunks WHERE collection=? AND episode=?",
            (collection, episode),
        )
        self._conn.commit()
        logger.debug(f"Deleted episode '{episode}' from '{collection}'")

    def get_collection_info(self, name: str) -> dict | None:
        """Return {show, model, chunker, dim} for a collection, or None if not found."""
        row = self._conn.execute(
            "SELECT show, model, chunker, dim FROM collections WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            return None
        return {"show": row[0], "model": row[1], "chunker": row[2], "dim": row[3]}

    def list_episodes(self, collection: str) -> list[str]:
        """Return sorted list of distinct episode names in the collection."""
        rows = self._conn.execute(
            "SELECT DISTINCT episode FROM chunks WHERE collection=? ORDER BY episode",
            (collection,),
        ).fetchall()
        return [r[0] for r in rows]

    # ── Write ──────────────────────────────────────────────────────────

    def save_chunks(
        self,
        collection: str,
        episode: str,
        chunks: list[dict],
        embeddings: np.ndarray,
    ) -> None:
        """Save chunks and embeddings atomically.

        Raises:
            ValueError: if len(chunks) != len(embeddings).
            sqlite3.IntegrityError: if (collection, episode, chunk_index)
                already exists. Call delete_episode() first to overwrite.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        with self._conn:
            for i, chunk in enumerate(chunks):
                text = chunk.get("text", "")
                meta = {k: v for k, v in chunk.items() if k != "text"}
                cur = self._conn.execute(
                    """
                    INSERT INTO chunks(collection, episode, chunk_index, text, meta_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (collection, episode, i, text, json.dumps(meta)),
                )
                chunk_id = cur.lastrowid
                self._conn.execute(
                    "INSERT INTO embeddings(chunk_id, vector) VALUES (?, ?)",
                    (chunk_id, embeddings[i].astype(np.float32).tobytes()),
                )
        logger.debug(f"Saved {len(chunks)} chunks for '{episode}' in '{collection}'")

    # ── Read ───────────────────────────────────────────────────────────

    def load_chunks(self, collection: str, episode: str) -> list[dict]:
        """Load chunks ordered by chunk_index; adds 'embedding': np.ndarray."""
        rows = self._conn.execute(
            """
            SELECT c.chunk_index, c.text, c.meta_json, e.vector
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.id
            WHERE c.collection=? AND c.episode=?
            ORDER BY c.chunk_index
            """,
            (collection, episode),
        ).fetchall()
        result = []
        for _, text, meta_json, blob in rows:
            chunk = json.loads(meta_json)
            chunk["text"] = text
            dim = len(blob) // 4  # float32 = 4 bytes each
            chunk["embedding"] = (
                np.frombuffer(blob, dtype=np.float32).copy().reshape(dim)
            )
            result.append(chunk)
        logger.debug(f"Loaded {len(result)} chunks for '{episode}' in '{collection}'")
        return result

    def load_chunks_no_embeddings(self, collection: str, episode: str) -> list[dict]:
        """Load chunks without pulling embedding blobs."""
        rows = self._conn.execute(
            """
            SELECT chunk_index, text, meta_json
            FROM chunks
            WHERE collection=? AND episode=?
            ORDER BY chunk_index
            """,
            (collection, episode),
        ).fetchall()
        result = []
        for _, text, meta_json in rows:
            chunk = json.loads(meta_json)
            chunk["text"] = text
            result.append(chunk)
        return result

    def enrich_chunk_meta(self, collection: str, episode: str, extras: dict) -> int:
        """Merge *extras* into meta_json for all chunks of an episode.

        Returns the number of updated rows.
        """
        rows = self._conn.execute(
            "SELECT id, meta_json FROM chunks WHERE collection=? AND episode=?",
            (collection, episode),
        ).fetchall()
        if not rows:
            return 0
        with self._conn:
            for row_id, meta_json in rows:
                meta = json.loads(meta_json)
                meta.update(extras)
                self._conn.execute(
                    "UPDATE chunks SET meta_json=? WHERE id=?",
                    (json.dumps(meta), row_id),
                )
        return len(rows)

    # ── Bulk read (for search) ────────────────────────────────────────

    def collection_chunk_count(self, collection: str) -> int:
        """Return total number of chunks in a collection."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE collection=?", (collection,)
        )
        return cur.fetchone()[0]

    def load_all_chunks(
        self, collection: str, episode: str | None = None
    ) -> list[dict]:
        """Load all chunks (without embeddings) for a collection.

        Args:
            collection : collection name
            episode    : if set, restrict to this episode
        """
        if episode:
            rows = self._conn.execute(
                """
                SELECT text, meta_json FROM chunks
                WHERE collection=? AND episode=?
                ORDER BY episode, chunk_index
                """,
                (collection, episode),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT text, meta_json FROM chunks
                WHERE collection=?
                ORDER BY episode, chunk_index
                """,
                (collection,),
            ).fetchall()
        result = []
        for text, meta_json in rows:
            chunk = json.loads(meta_json)
            chunk["text"] = text
            result.append(chunk)
        return result

    def load_all_vectors(self, collection: str) -> tuple[np.ndarray, list[dict]]:
        """Load all embeddings + metadata for a collection in one query.

        Returns:
            (matrix, chunks) where matrix is shape (N, dim) float32
            and chunks is a list of metadata dicts (with 'text' field).
            Returns (empty 0×0 array, []) if collection has no chunks.
        """
        rows = self._conn.execute(
            """
            SELECT c.text, c.meta_json, e.vector
            FROM chunks c
            JOIN embeddings e ON e.chunk_id = c.id
            WHERE c.collection=?
            ORDER BY c.episode, c.chunk_index
            """,
            (collection,),
        ).fetchall()
        if not rows:
            return np.empty((0, 0), dtype=np.float32), []

        chunks = []
        blobs = []
        for text, meta_json, blob in rows:
            chunk = json.loads(meta_json)
            chunk["text"] = text
            chunks.append(chunk)
            blobs.append(blob)

        dim = len(blobs[0]) // 4  # float32 = 4 bytes
        matrix = np.frombuffer(b"".join(blobs), dtype=np.float32).copy()
        matrix = matrix.reshape(len(rows), dim)
        return matrix, chunks

    def list_sources(self, collection: str) -> list[str]:
        """Return sorted list of distinct source values in a collection."""
        rows = self._conn.execute(
            "SELECT DISTINCT meta_json FROM chunks WHERE collection=?",
            (collection,),
        ).fetchall()
        sources: set[str] = set()
        for (meta_json,) in rows:
            meta = json.loads(meta_json)
            src = meta.get("source")
            if src:
                sources.add(src)
        return sorted(sources)

    def list_speakers(self, collection: str) -> list[str]:
        """Return sorted list of distinct speaker/dominant_speaker values."""
        rows = self._conn.execute(
            "SELECT DISTINCT meta_json FROM chunks WHERE collection=?",
            (collection,),
        ).fetchall()
        speakers: set[str] = set()
        for (meta_json,) in rows:
            meta = json.loads(meta_json)
            spk = meta.get("dominant_speaker") or meta.get("speaker")
            if spk:
                speakers.add(spk)
        return sorted(speakers)

    def get_episode_stats(self, collection: str) -> list[dict]:
        """Return per-episode stats: chunk count, duration, speakers."""
        episodes = self._conn.execute(
            "SELECT DISTINCT episode FROM chunks WHERE collection=? ORDER BY episode",
            (collection,),
        ).fetchall()
        result = []
        for (ep,) in episodes:
            chunk_rows = self._conn.execute(
                "SELECT meta_json FROM chunks WHERE collection=? AND episode=?",
                (collection, ep),
            ).fetchall()
            speakers: set[str] = set()
            max_end = 0.0
            for (meta_json,) in chunk_rows:
                meta = json.loads(meta_json)
                spk = meta.get("dominant_speaker") or meta.get("speaker")
                if spk:
                    speakers.add(spk)
                end = meta.get("end", 0.0)
                if end > max_end:
                    max_end = end
            result.append(
                {
                    "episode": ep,
                    "chunk_count": len(chunk_rows),
                    "duration": max_end,
                    "speakers": sorted(speakers),
                }
            )
        return result
