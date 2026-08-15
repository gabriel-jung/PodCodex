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
    verified_step         TEXT,
    verified_version_id   TEXT,
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
    # Verified pointer: ``(verified_step, verified_version_id)`` marks the
    # user-reviewed canonical version for the episode. Singleton per episode.
    (
        "SELECT 1 FROM pragma_table_info('episodes') WHERE name='verified_step'",
        ["ALTER TABLE episodes ADD COLUMN verified_step TEXT"],
    ),
    (
        "SELECT 1 FROM pragma_table_info('episodes') WHERE name='verified_version_id'",
        ["ALTER TABLE episodes ADD COLUMN verified_version_id TEXT"],
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
        # DELETE, not WAL: WAL needs shared memory in the same directory and
        # is unsafe on synced/network folders. See "sync-safe pipeline DB".
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
        """Close the underlying SQLite connection.

        Taken under the per-DB lock so an in-flight query on another
        thread finishes before the connection goes away. A thread holding
        a stale instance after close gets one sqlite ProgrammingError and
        re-resolves through get_pipeline_db on its next request.
        """
        with self._lock:
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

    def aggregate_status(self) -> dict[str, int]:
        """Return per-stage completion counts across all episodes.

        Includes edited subset for transcribe/correct/translate. Edited mirrors
        `core.versions.is_edited` (type='validated' or manual_edit=True on the
        latest provenance entry for that step). Translation counts as edited
        when at least one translation language has edited provenance.
        """
        # Lazy import: versions.py also lazy-imports pipeline_db inside its
        # functions, so a top-level import here would still be fine — but
        # keeping it lazy keeps the dependency direction obvious.
        from podcodex.core.versions import is_edited

        rows = self._conn.execute(
            "SELECT transcribed, corrected, indexed, synthesized, "
            "translations, provenance FROM episodes"
        ).fetchall()

        total = transcribed = corrected = translated = synthesized = indexed = 0
        transcribed_edited = corrected_edited = translated_edited = 0

        for r in rows:
            total += 1
            if r["transcribed"]:
                transcribed += 1
            if r["corrected"]:
                corrected += 1
            if r["indexed"]:
                indexed += 1
            if r["synthesized"]:
                synthesized += 1

            translations: list[str] = []
            raw_t = r["translations"]
            if raw_t:
                try:
                    parsed = json.loads(raw_t)
                    if isinstance(parsed, list):
                        translations = [t for t in parsed if isinstance(t, str)]
                except (ValueError, TypeError):
                    pass
            has_translation = len(translations) > 0
            if has_translation:
                translated += 1

            prov: dict = {}
            raw_p = r["provenance"]
            if raw_p:
                try:
                    parsed = json.loads(raw_p)
                    if isinstance(parsed, dict):
                        prov = parsed
                except (ValueError, TypeError):
                    pass

            if r["transcribed"] and is_edited(prov.get("transcript")):
                transcribed_edited += 1
            if r["corrected"] and is_edited(prov.get("corrected")):
                corrected_edited += 1
            if has_translation and any(
                is_edited(prov.get(lang)) for lang in translations
            ):
                translated_edited += 1

        return {
            "total": total,
            "transcribed": transcribed,
            "transcribed_edited": transcribed_edited,
            "corrected": corrected,
            "corrected_edited": corrected_edited,
            "translated": translated,
            "translated_edited": translated_edited,
            "synthesized": synthesized,
            "indexed": indexed,
        }

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

    def demote_step_if_no_versions(
        self, stem: str, step: str, flag: str | None
    ) -> bool:
        """Clear a step's status only if the step still has no versions.

        The check and the write happen inside one ``BEGIN IMMEDIATE``, which
        is what makes this safe: pipeline steps run in spawned subprocesses
        (see ``api/subprocess_runner``) that write to this same file, so the
        in-process lock alone cannot stop a version landing between a
        "any versions left?" read and the demotion that follows it. Losing
        that race strands a False flag next to a live version.

        Args:
            stem: Episode stem.
            step: Version step that was just emptied.
            flag: Boolean column to clear, or None to drop *step* from the
                  ``translations`` list instead.

        Returns:
            True when the step has no versions left, whether or not that
            required a write (the row may be absent, or the language may
            already be off the list). False only when a version exists, i.e.
            one landed between the caller's delete and this check. Callers
            use it to decide whether the step's other leftovers, such as
            recorded LLM batch failures, should be cleared too.
        """
        if flag is not None and flag not in _VALID_COLUMNS:
            raise ValueError(f"Unknown column: {flag}")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT 1 FROM versions WHERE stem = ? AND step = ? LIMIT 1",
                    (stem, step),
                ).fetchone()
                if row is not None:
                    self._conn.rollback()
                    return False
                if flag is not None:
                    self._conn.execute(
                        f"UPDATE episodes SET {flag} = 0, updated_at = ? WHERE stem = ?",
                        (time.time(), stem),
                    )
                else:
                    current = self._conn.execute(
                        "SELECT translations FROM episodes WHERE stem = ?", (stem,)
                    ).fetchone()
                    if current is None:
                        self._conn.rollback()
                        return True
                    langs = json.loads(current["translations"] or "[]")
                    if step not in langs:
                        self._conn.rollback()
                        return True
                    langs.remove(step)
                    self._conn.execute(
                        "UPDATE episodes SET translations = ?, updated_at = ? "
                        "WHERE stem = ?",
                        (json.dumps(langs), time.time(), stem),
                    )
                self._conn.commit()
                return True
            except Exception:
                self._conn.rollback()
                raise

    # ── Verified pointer ──────────────────────────────────

    def set_verified(self, stem: str, step: str, version_id: str) -> None:
        """Mark a specific version as the episode's verified source.

        Singleton: replaces any previous verified pointer for this stem.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO episodes (stem, verified_step, verified_version_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(stem) DO UPDATE SET
                    verified_step = excluded.verified_step,
                    verified_version_id = excluded.verified_version_id,
                    updated_at = excluded.updated_at
                """,
                (stem, step, version_id, time.time()),
            )
            self._conn.commit()

    def clear_verified(self, stem: str) -> None:
        """Clear the verified pointer for an episode."""
        with self._lock:
            self._conn.execute(
                """
                UPDATE episodes
                SET verified_step = NULL, verified_version_id = NULL, updated_at = ?
                WHERE stem = ?
                """,
                (time.time(), stem),
            )
            self._conn.commit()

    def get_verified(self, stem: str) -> dict | None:
        """Return ``{"step", "version_id"}`` for the verified pointer, or None."""
        row = self._conn.execute(
            "SELECT verified_step, verified_version_id FROM episodes WHERE stem = ?",
            (stem,),
        ).fetchone()
        if not row or not row["verified_step"] or not row["verified_version_id"]:
            return None
        return {"step": row["verified_step"], "version_id": row["verified_version_id"]}

    def stems_with_verified(self) -> set[str]:
        """Return the set of stems that have a verified pointer set."""
        rows = self._conn.execute(
            "SELECT stem FROM episodes WHERE verified_version_id IS NOT NULL"
        ).fetchall()
        return {r[0] for r in rows}

    def verified_pointers(self) -> dict[str, dict]:
        """Bulk: ``{stem: {step, version_id}}`` for every episode with a pointer."""
        rows = self._conn.execute(
            "SELECT stem, verified_step, verified_version_id "
            "FROM episodes WHERE verified_version_id IS NOT NULL"
        ).fetchall()
        return {
            r["stem"]: {
                "step": r["verified_step"],
                "version_id": r["verified_version_id"],
            }
            for r in rows
        }

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

    def get_version(self, version_id: str) -> dict | None:
        """Return the version row for ``version_id``, or None."""
        row = self._conn.execute(
            "SELECT * FROM versions WHERE id = ? LIMIT 1",
            (version_id,),
        ).fetchone()
        return self._version_to_dict(row) if row else None

    def versions_for_steps(self, steps: list[str]) -> dict[tuple[str, str], list[dict]]:
        """Bulk: all versions per ``(stem, step)``, newest first, one query.

        The per-stem lists match ``list_versions`` ordering so callers can
        apply the same default-pick rules (e.g. edited-first) without N
        round-trips.
        """
        if not steps:
            return {}
        placeholders = ", ".join("?" for _ in steps)
        rows = self._conn.execute(
            f"""SELECT * FROM versions
                WHERE step IN ({placeholders})
                ORDER BY timestamp DESC""",
            steps,
        ).fetchall()
        out: dict[tuple[str, str], list[dict]] = {}
        for r in rows:
            d = self._version_to_dict(r)
            out.setdefault((d["stem"], d["step"]), []).append(d)
        return out

    def stems_with_step(self, step: str) -> set[str]:
        """Return the set of stems that have at least one version for ``step``."""
        rows = self._conn.execute(
            "SELECT DISTINCT stem FROM versions WHERE step = ?", (step,)
        ).fetchall()
        return {r[0] for r in rows}

    def version_ids_by_stem(self, step: str) -> dict[str, set[str]]:
        """Bulk: ``{stem: {version_id, ...}}`` for one step."""
        rows = self._conn.execute(
            "SELECT stem, id FROM versions WHERE step = ?", (step,)
        ).fetchall()
        out: dict[str, set[str]] = {}
        for r in rows:
            out.setdefault(r["stem"], set()).add(r["id"])
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
        # Collapse the two verified-pointer columns into a single dict on the
        # row so consumers don't have to remember the column split.
        if d.get("verified_step") and d.get("verified_version_id"):
            d["verified"] = {
                "step": d["verified_step"],
                "version_id": d["verified_version_id"],
            }
        else:
            d["verified"] = None
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


def reset_pipeline_db(show_folder: Path | str) -> None:
    """Close the cached instance AND delete the DB file, atomically.

    Holding ``_dbs_lock`` across close + unlink stops a concurrent
    ``get_pipeline_db`` (threadpool status poll) from re-opening the file
    between the two steps and then writing every subsequent update to a
    deleted inode until process restart.
    """
    show_folder = Path(show_folder)
    with _dbs_lock:
        db = _dbs.pop(show_folder, None)
        if db:
            db.close()
        (show_folder / DB_FILENAME).unlink(missing_ok=True)


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
