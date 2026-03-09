"""
podcodex.rag.localstore — SQLite persistence layer for podcast RAG.

Stores chunk texts, metadata, and embeddings locally so Qdrant is the
serving layer but not the source of truth. Enables re-indexing without
re-embedding.

Schema (3 tables):
    collections  — one row per (show, model) pair
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

    # ── Collection management ──────────────────────────────────────────

    def collection_exists(self, name: str) -> bool:
        cur = self._conn.execute("SELECT 1 FROM collections WHERE name=?", (name,))
        return cur.fetchone() is not None

    def ensure_collection(self, name: str, show: str, model: str, dim: int) -> None:
        """Create collection row if it does not already exist (idempotent)."""
        self._conn.execute(
            """
            INSERT OR IGNORE INTO collections(name, show, model, dim, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, show, model, dim, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def list_collections(self, show: str = "", model: str = "") -> list[str]:
        """List collection names, optionally filtered by show and/or model."""
        q = "SELECT name FROM collections WHERE 1=1"
        params: list = []
        if show:
            q += " AND show=?"
            params.append(show)
        if model:
            q += " AND model=?"
            params.append(model)
        q += " ORDER BY name"
        return [r[0] for r in self._conn.execute(q, params).fetchall()]

    def delete_collection(self, name: str) -> None:
        """Delete collection; ON DELETE CASCADE removes chunks + embeddings."""
        self._conn.execute("DELETE FROM collections WHERE name=?", (name,))
        self._conn.commit()

    # ── Episode-level ──────────────────────────────────────────────────

    def episode_is_indexed(self, collection: str, episode: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM chunks WHERE collection=? AND episode=? LIMIT 1",
            (collection, episode),
        )
        return cur.fetchone() is not None

    def delete_episode(self, collection: str, episode: str) -> None:
        self._conn.execute(
            "DELETE FROM chunks WHERE collection=? AND episode=?",
            (collection, episode),
        )
        self._conn.commit()

    def get_collection_info(self, name: str) -> dict | None:
        """Return {show, model, dim} for a collection, or None if not found."""
        row = self._conn.execute(
            "SELECT show, model, dim FROM collections WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            return None
        return {"show": row[0], "model": row[1], "dim": row[2]}

    def list_episodes(self, collection: str) -> list[str]:
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
