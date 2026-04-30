"""
podcodex.core.pipeline_db — Per-show SQLite database for pipeline status.

Tracks which pipeline steps have been completed for each episode.
One ``pipeline.db`` per show folder.

Usage::

    db = get_pipeline_db(show_folder)
    db.mark("ep_stem", transcribed=True)

    for row in db.all_episodes():
        print(row["stem"], row["transcribed"])
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from loguru import logger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    stem                  TEXT PRIMARY KEY,
    audio_path            TEXT,
    transcribed           INTEGER DEFAULT 0,
    corrected             INTEGER DEFAULT 0,
    indexed               INTEGER DEFAULT 0,
    synthesized           INTEGER DEFAULT 0,
    translations          TEXT DEFAULT '[]',
    provenance            TEXT DEFAULT '{}',
    updated_at            REAL
);

CREATE TABLE IF NOT EXISTS versions (
    id              TEXT NOT NULL,
    stem            TEXT NOT NULL,
    step            TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    type            TEXT NOT NULL,
    model           TEXT,
    params          TEXT DEFAULT '{}',
    manual_edit     INTEGER DEFAULT 0,
    content_hash    TEXT NOT NULL,
    segment_count   INTEGER NOT NULL,
    input_hash      TEXT,
    PRIMARY KEY (id, stem, step)
);

CREATE INDEX IF NOT EXISTS idx_versions_stem_step
    ON versions(stem, step);
"""

_MIGRATIONS: list[tuple[str, list[str]]] = [
    # (check_sql, [apply_stmts]) — each migration runs once if check returns no rows.
    (
        "SELECT 1 FROM pragma_table_info('episodes') WHERE name='provenance'",
        ["ALTER TABLE episodes ADD COLUMN provenance TEXT DEFAULT '{}'"],
    ),
    # Rename polished → corrected
    (
        "SELECT 1 FROM pragma_table_info('episodes') WHERE name='corrected'",
        ["ALTER TABLE episodes RENAME COLUMN polished TO corrected"],
    ),
    (
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='versions'",
        [
            """
            CREATE TABLE IF NOT EXISTS versions (
                id              TEXT NOT NULL,
                stem            TEXT NOT NULL,
                step            TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                type            TEXT NOT NULL,
                model           TEXT,
                params          TEXT DEFAULT '{}',
                manual_edit     INTEGER DEFAULT 0,
                content_hash    TEXT NOT NULL,
                segment_count   INTEGER NOT NULL,
                input_hash      TEXT,
                PRIMARY KEY (id, stem, step)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_versions_stem_step ON versions(stem, step)",
        ],
    ),
]

# Columns that can be set via mark().
_VALID_COLUMNS = frozenset(
    {
        "audio_path",
        "transcribed",
        "corrected",
        "indexed",
        "synthesized",
        "translations",
        "provenance",
    }
)

DB_FILENAME = "pipeline.db"


class PipelineDB:
    """Per-show SQLite database for pipeline episode status.

    Args:
        db_path: Path to the SQLite file.  Use ``":memory:"`` for tests.
    """

    def __init__(self, db_path: Path | str):
        self._path = str(db_path)
        self._lock = threading.Lock()
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.executescript(_SCHEMA)
        self._run_migrations()
        self._conn.commit()

    def _run_migrations(self) -> None:
        """Apply any pending schema migrations."""
        for check_sql, apply_stmts in _MIGRATIONS:
            if self._conn.execute(check_sql).fetchone():
                continue
            with self._conn:  # atomic: BEGIN/COMMIT or ROLLBACK on exception
                for stmt in apply_stmts:
                    self._conn.execute(stmt)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ── Read ──────────────────────────────────────────────

    def all_episodes(self) -> list[dict]:
        """Return status for every episode in one query."""
        rows = self._conn.execute("SELECT * FROM episodes ORDER BY stem").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_episode(self, stem: str) -> dict | None:
        """Return status for a single episode, or None."""
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE stem = ?", (stem,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def episode_count(self) -> int:
        """Return the number of episodes in the DB."""
        return self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    # ── Write ─────────────────────────────────────────────

    def mark(self, stem: str, **fields: object) -> None:
        """UPSERT specific status columns for an episode.

        Only columns listed in ``_VALID_COLUMNS`` are accepted.
        The ``translations`` field should be a list; it will be JSON-encoded.

        Example::

            db.mark("ep_stem", transcribed=True)
            db.mark("ep_stem", translations=["english", "french"])
        """
        bad = set(fields) - _VALID_COLUMNS
        if bad:
            raise ValueError(f"Unknown columns: {bad}")
        if not fields:
            return

        # JSON-encode translations if provided as a list.
        if "translations" in fields and isinstance(fields["translations"], list):
            fields["translations"] = json.dumps(fields["translations"])

        with self._lock:
            # Provenance is a dict keyed by step — merge with existing.
            if "provenance" in fields and isinstance(fields["provenance"], dict):
                existing = self._get_provenance(stem)
                existing.update(fields["provenance"])
                fields["provenance"] = json.dumps(existing)

            cols = list(fields.keys())
            vals = [fields[c] for c in cols]

            set_clause = ", ".join(f"{c} = excluded.{c}" for c in cols)
            placeholders = ", ".join("?" for _ in cols)
            col_names = ", ".join(cols)

            sql = f"""
                INSERT INTO episodes (stem, {col_names}, updated_at)
                VALUES (?, {placeholders}, ?)
                ON CONFLICT(stem) DO UPDATE SET {set_clause}, updated_at = excluded.updated_at
            """
            vals_full = [stem, *vals, time.time()]
            self._conn.execute(sql, vals_full)
            self._conn.commit()

    def mark_indexed_bulk(self, updates: dict[str, bool]) -> None:
        """Set the ``indexed`` flag for many stems in a single transaction.

        Used by the per-show LanceDB reconciliation path where dozens to
        hundreds of rows may need correcting at once; one commit per row
        would block the FastAPI event loop on the SQLite write lock.
        """
        if not updates:
            return
        now = time.time()
        rows = [(stem, int(v), now, int(v), now) for stem, v in updates.items()]
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO episodes (stem, indexed, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(stem) DO UPDATE SET indexed = ?, updated_at = ?
                """,
                rows,
            )

    def ensure_episode(self, stem: str, audio_path: str | None = None) -> None:
        """Create an episode row if it does not exist (idempotent)."""
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO episodes (stem, audio_path, updated_at)
                VALUES (?, ?, ?)
                """,
                (stem, audio_path, time.time()),
            )
            self._conn.commit()

    def remove_episode(self, stem: str) -> None:
        """Delete an episode row."""
        with self._lock:
            self._conn.execute("DELETE FROM episodes WHERE stem = ?", (stem,))
            self._conn.commit()

    # ── Versions ─────────────────────────────────────────

    def insert_version(self, stem: str, step: str, meta: dict) -> None:
        """Insert a version metadata row."""
        params = meta.get("params", {})
        if isinstance(params, dict):
            params = json.dumps(params)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO versions
                    (id, stem, step, timestamp, type, model, params,
                     manual_edit, content_hash, segment_count, input_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta["id"],
                    stem,
                    step,
                    meta["timestamp"],
                    meta["type"],
                    meta.get("model"),
                    params,
                    int(meta.get("manual_edit", False)),
                    meta["content_hash"],
                    meta["segment_count"],
                    meta.get("input_hash"),
                ),
            )
            self._conn.commit()

    def list_versions(self, stem: str, step: str) -> list[dict]:
        """List all versions for an episode step (newest first)."""
        rows = self._conn.execute(
            """SELECT * FROM versions
               WHERE stem = ? AND step = ?
               ORDER BY timestamp DESC""",
            (stem, step),
        ).fetchall()
        return [self._version_to_dict(r) for r in rows]

    def get_latest_version(self, stem: str, step: str) -> dict | None:
        """Return the most recent version for a step, or None."""
        row = self._conn.execute(
            """SELECT * FROM versions
               WHERE stem = ? AND step = ?
               ORDER BY timestamp DESC LIMIT 1""",
            (stem, step),
        ).fetchone()
        return self._version_to_dict(row) if row else None

    def latest_versions_for_steps(
        self, steps: list[str]
    ) -> dict[tuple[str, str], dict]:
        """Bulk: latest version per ``(stem, step)`` via one window-function query."""
        if not steps:
            return {}
        placeholders = ", ".join("?" for _ in steps)
        rows = self._conn.execute(
            f"""SELECT * FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (PARTITION BY stem, step ORDER BY timestamp DESC) AS rn
                    FROM versions
                    WHERE step IN ({placeholders})
                )
               WHERE rn = 1""",
            steps,
        ).fetchall()
        out: dict[tuple[str, str], dict] = {}
        for r in rows:
            d = self._version_to_dict(r)
            out[(d["stem"], d["step"])] = d
        return out

    def list_all_versions(self, stem: str) -> list[dict]:
        """List all versions across all steps for an episode (newest first)."""
        rows = self._conn.execute(
            """SELECT * FROM versions
               WHERE stem = ?
               ORDER BY timestamp DESC""",
            (stem,),
        ).fetchall()
        return [self._version_to_dict(r) for r in rows]

    def list_steps(self, stem: str) -> list[str]:
        """Return distinct step names for an episode (sorted)."""
        rows = self._conn.execute(
            "SELECT DISTINCT step FROM versions WHERE stem = ? ORDER BY step",
            (stem,),
        ).fetchall()
        return [r[0] for r in rows]

    def version_count(self, stem: str, step: str) -> int:
        """Return the number of versions for a step."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM versions WHERE stem = ? AND step = ?",
            (stem, step),
        ).fetchone()[0]

    def delete_versions(self, stem: str, step: str, ids: list[str]) -> int:
        """Delete specific versions by ID. Returns count deleted."""
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        with self._lock:
            cur = self._conn.execute(
                f"DELETE FROM versions WHERE stem = ? AND step = ? AND id IN ({placeholders})",
                [stem, step, *ids],
            )
            self._conn.commit()
            return cur.rowcount

    def delete_all_versions(self, stem: str) -> int:
        """Delete all versions for an episode. Returns count deleted."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM versions WHERE stem = ?", (stem,))
            self._conn.commit()
            return cur.rowcount

    @staticmethod
    def _version_to_dict(row: sqlite3.Row) -> dict:
        """Convert a versions Row to a plain dict."""
        d = dict(row)
        p = d.get("params", "{}")
        d["params"] = json.loads(p) if isinstance(p, str) else (p or {})
        d["manual_edit"] = bool(d.get("manual_edit", 0))
        return d

    def latest_segment_counts(self, step: str = "transcript") -> dict[str, int]:
        """Return {stem: segment_count} for the latest version of each episode.

        Uses a single query with a window function to avoid N+1 lookups.
        """
        rows = self._conn.execute(
            """SELECT stem, segment_count
               FROM (
                   SELECT stem, segment_count,
                          ROW_NUMBER() OVER (PARTITION BY stem ORDER BY timestamp DESC) AS rn
                   FROM versions WHERE step = ?
               ) WHERE rn = 1""",
            (step,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Bulk ──────────────────────────────────────────────

    def populate_from_scan(self, episodes: list) -> None:
        """Bulk-insert episode status from a list of EpisodeInfo objects.

        Existing rows are updated (UPSERT).  Used for initial migration
        when a show has no pipeline.db yet.
        """
        now = time.time()
        rows = []
        for ep in episodes:
            translations = getattr(ep, "translations", [])
            rows.append(
                (
                    ep.stem,
                    str(ep.audio_path) if ep.audio_path else None,
                    int(ep.transcribed),
                    int(ep.corrected),
                    int(ep.indexed),
                    int(ep.synthesized),
                    json.dumps(translations),
                    "{}",
                    now,
                )
            )
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO episodes (
                    stem, audio_path, transcribed, corrected, indexed, synthesized,
                    translations, provenance, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stem) DO UPDATE SET
                    audio_path = excluded.audio_path,
                    transcribed = excluded.transcribed,
                    corrected = excluded.corrected,
                    indexed = excluded.indexed,
                    synthesized = excluded.synthesized,
                    translations = excluded.translations,
                    provenance = excluded.provenance,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
            self._conn.commit()

    # ── Helpers ───────────────────────────────────────────

    def _get_provenance(self, stem: str) -> dict:
        """Read existing provenance JSON for a stem, or return {}."""
        row = self._conn.execute(
            "SELECT provenance FROM episodes WHERE stem = ?", (stem,)
        ).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        """Convert a Row to a plain dict with proper types."""
        d = dict(row)
        # Decode translations JSON.
        t = d.get("translations", "[]")
        d["translations"] = json.loads(t) if isinstance(t, str) else t
        # Decode provenance JSON.
        p = d.get("provenance", "{}")
        d["provenance"] = json.loads(p) if isinstance(p, str) else (p or {})
        # Booleans.
        for key in ("transcribed", "corrected", "indexed", "synthesized"):
            d[key] = bool(d.get(key, 0))
        return d


# ── Module-level instance cache ───────────────────────────

_dbs: dict[Path, PipelineDB] = {}
_dbs_lock = threading.Lock()


def get_pipeline_db(show_folder: Path | str) -> PipelineDB:
    """Return a cached PipelineDB instance for the given show folder."""
    show_folder = Path(show_folder)
    with _dbs_lock:
        if show_folder not in _dbs:
            db_path = show_folder / DB_FILENAME
            _dbs[show_folder] = PipelineDB(db_path)
        return _dbs[show_folder]


def close_pipeline_db(show_folder: Path | str) -> None:
    """Close and remove a cached PipelineDB instance."""
    show_folder = Path(show_folder)
    with _dbs_lock:
        db = _dbs.pop(show_folder, None)
        if db:
            db.close()


def mark_step(show_dir: Path, stem: str, **fields: object) -> None:
    """Safely update pipeline status — logs and swallows errors.

    Called by pipeline save functions after writing files.  If the DB
    write fails for any reason the pipeline still succeeds.
    """
    try:
        get_pipeline_db(show_dir).mark(stem, **fields)
    except Exception:
        logger.opt(exception=True).warning(
            f"pipeline_db: failed to mark {stem} in {show_dir}"
        )
