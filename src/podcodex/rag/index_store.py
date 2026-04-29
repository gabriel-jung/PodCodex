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
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from loguru import logger

_COLLECTIONS_TABLE = "_collections"


# Hyphens/dashes, apostrophes, and non-standard spaces all normalize to a
# regular space — combined into one regex to avoid three sequential passes.
_NORMALIZE_RE = re.compile(
    r"[\u002D\u2010-\u2015\u2212"  # hyphens / dashes
    r"\u0027\u2018\u2019\u201A\u201B\u2032\u02BC"  # apostrophes (straight + curly)
    r"\u00A0\u202F\u2009\u200A\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u200B\u3000]"  # non-standard spaces
)
# Ligatures not decomposed by NFD — map to ASCII equivalents before encoding.
_LIGATURES = str.maketrans(
    {
        "œ": "oe",
        "Œ": "oe",
        "æ": "ae",
        "Æ": "ae",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def fold_text(s: str) -> str:
    """Normalize text for accent-insensitive matching.

    Steps (order matters):
    1. Hyphens/dashes, apostrophes, Unicode spaces → regular space (one pass)
    2. Ligatures (œ, æ, ﬁ…) → ASCII equivalents (NFD doesn't decompose them)
    3. Remaining punctuation (,.!?;: etc.) → space
    4. NFD + ASCII encode to strip diacritics, then lowercase
    """
    s = _NORMALIZE_RE.sub(" ", s)
    s = s.translate(_LIGATURES)
    s = _PUNCT_RE.sub(" ", s)
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def _find_original_span(text: str, folded_q: str) -> str | None:
    """Find the substring of text whose fold_text equals folded_q.

    Needed when ligature expansion (e.g. ﬁ→fi) shifts character offsets so
    the folded index no longer maps directly to the original text position.
    """
    fq_len = len(folded_q)
    for i in range(len(text)):
        acc = ""
        for j in range(i, len(text)):
            acc = fold_text(text[i : j + 1])
            if acc == folded_q:
                return text[i : j + 1]
            if len(acc) > fq_len:
                break
    return None


def _accent_span(text: str, folded_t: str, folded_q: str) -> str | None:
    """Locate the original-text slice that accent-folds to ``folded_q``.

    Fast path when no ligatures expand (folded length == original length);
    fallback scan handles ligature-induced offset drift.
    """
    if len(folded_t) == len(text):
        idx = folded_t.find(folded_q)
        return text[idx : idx + len(folded_q)] if idx >= 0 else None
    return _find_original_span(text, folded_q)


def _accent_score(query: str, span: str) -> float:
    """Similarity between query and its matched original-text span."""
    return SequenceMatcher(None, query.lower(), span.lower()).ratio()


_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


def _levenshtein(a: str, b: str, max_dist: int = 999) -> int:
    """Exact Levenshtein edit distance with optional early exit at max_dist."""
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for c in a:
        new_row = [row[0] + 1]
        for j, d in enumerate(b):
            new_row.append(min(new_row[-1] + 1, row[j + 1] + 1, row[j] + (c != d)))
        if min(new_row) > max_dist:
            return max_dist + 1
        row = new_row
    return row[-1]


_WORD_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _fuzzy_match(query: str, text: str, max_dist: int) -> tuple[float, str] | None:
    """Return (similarity, matched original span) if any word in text is within
    max_dist edits of the folded single-word query, else None."""
    q = fold_text(query)
    best: tuple[float, str] | None = None
    for m in _WORD_TOKEN_RE.finditer(text):
        raw = m.group(0)
        w = fold_text(raw)
        if not w:
            continue
        d = _levenshtein(q, w, max_dist)
        if d <= max_dist:
            ratio = 1.0 - d / max(len(q), len(w), 1)
            if best is None or ratio > best[0]:
                best = (ratio, raw)
    return best


def _chunk_key(chunk: dict) -> str:
    return f"{chunk.get('episode', '')}|{chunk.get('start', 0)}"


def _approx_substring(
    pattern: str, text: str, max_dist: int
) -> tuple[int, int, int] | None:
    """Approximate substring match via Sellers' algorithm.

    Finds the substring of ``text`` with minimum Levenshtein distance to
    ``pattern``. Wagner-Fischer DP with row-0 seeded to zeros (free start
    in text) and the final row min taken over all columns (free end).
    Tracks the start column through the DP so we can return the matched
    span's ``[start, end)`` indices, not just its distance.

    Order is preserved by construction (no reordering is possible in a
    substring search) — unlike token-window fuzzy matching, "lumière
    très" cannot match "êtres de lumière" at low distance.

    Args:
        pattern: Folded query phrase.
        text:    Folded chunk text to search.
        max_dist: Distance ceiling; early-exits when every row cell
            exceeds it, returning ``None``.

    Returns:
        ``(distance, start, end)`` of the best substring, or ``None``
        when distance exceeds ``max_dist``.
    """
    if not pattern:
        return (0, 0, 0)
    n = len(text)
    prev_cost = [0] * (n + 1)
    prev_start = list(range(n + 1))  # free start: each column starts at itself
    for i, pc in enumerate(pattern, 1):
        curr_cost = [i] + [0] * n
        curr_start = [0] + [0] * n
        row_min = i
        for j in range(1, n + 1):
            sub = prev_cost[j - 1] + (0 if pc == text[j - 1] else 1)
            ins = curr_cost[j - 1] + 1
            dele = prev_cost[j] + 1
            best = sub
            start = prev_start[j - 1]
            if ins < best:
                best, start = ins, curr_start[j - 1]
            if dele < best:
                best, start = dele, prev_start[j]
            curr_cost[j] = best
            curr_start[j] = start
            if best < row_min:
                row_min = best
        if row_min > max_dist:
            return None
        prev_cost, prev_start = curr_cost, curr_start
    best_d = min(prev_cost)
    if best_d > max_dist:
        return None
    end = prev_cost.index(best_d)
    return (best_d, prev_start[end], end)


_REPO_FALLBACKS: tuple[Path, ...] = (Path("deploy") / "index", Path("index"))


def _canonical_index_path() -> Path:
    from podcodex.core.app_paths import data_dir

    return data_dir() / "index"


def _dir_has_data(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError as exc:
        logger.warning(f"Cannot read {path}: {exc}")
        return False


def _resolve_default_index_path() -> tuple[Path, str]:
    """Pick the default index path when no explicit one is passed.

    Resolution order:
      1. ``PODCODEX_INDEX`` env var (explicit override).
      2. ``<data_dir>/index`` if populated — canonical location, lives
         under the same app-data root as models/logs/GPU backend.
      3. ``./deploy/index`` or ``./index`` (repo-local fallback, if populated).
      4. ``<data_dir>/index`` (created empty).

    Returns ``(path, reason)`` for logging.
    """
    override = os.environ.get("PODCODEX_INDEX", "").strip()
    if override:
        return Path(override), "PODCODEX_INDEX env"
    canonical = _canonical_index_path()
    if _dir_has_data(canonical):
        return canonical, "data dir"
    for candidate in _REPO_FALLBACKS:
        resolved = candidate.resolve()
        if _dir_has_data(resolved):
            return resolved, f"repo-local fallback ({candidate})"
    return canonical, "data dir (new, empty)"


def _chunk_schema(dim: int) -> pa.Schema:
    """Arrow schema for a collection table with the given vector dimensionality."""
    return pa.schema(
        [
            pa.field("chunk_index", pa.int32()),
            pa.field("show", pa.string()),
            pa.field("episode", pa.string()),
            pa.field("source", pa.string()),
            pa.field("pub_date", pa.string()),
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

_SHOW_PASSWORDS_TABLE = "_show_passwords"
_SHOW_PASSWORDS_SCHEMA = pa.schema(
    [
        pa.field("show", pa.string()),
        pa.field("password_hash", pa.string()),  # "sha256:<hex>"
    ]
)


_RESERVED_META_KEYS = {
    "chunk_index",
    "show",
    "episode",
    "source",
    "pub_date",
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
        path: Directory holding the LanceDB database. When ``None``, resolved
            via :func:`_resolve_default_index_path`.
    """

    # Global resolver (set by the API/bot at startup) mapping a show name to
    # its on-disk folder. Used by the ``episode_title`` backfill to heal
    # chunks where the RSS title never made it into the transcript meta.
    _show_folder_resolver: "callable | None" = None

    @classmethod
    def set_show_folder_resolver(cls, fn) -> None:
        """Register a callable ``(show_name) -> Path | None`` for the backfill."""
        cls._show_folder_resolver = fn

    def __init__(self, path: Path | str | None = None):
        import lancedb

        if path is None:
            resolved, reason = _resolve_default_index_path()
            self._path = resolved
            logger.info(f"IndexStore opened: {self._path} ({reason})")
        else:
            self._path = Path(path)
            logger.info(f"IndexStore opened: {self._path} (explicit path)")
        self._path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._path))
        self._fts_ready: set[str] = set()
        self._pub_date_ready: set[str] = set()
        self._episode_title_ready: set[str] = set()

    @property
    def path(self) -> Path:
        """Filesystem root of the LanceDB index directory."""
        return self._path

    # ── External-change detection & reconnection ─────────────────────────
    #
    # The index directory can be replaced (rsync) or extended (desktop app
    # adds a show, API sets a password) while the bot holds a live LanceDB
    # connection. `index_mtime` lets callers cheaply detect such changes;
    # `reconnect` rebuilds the connection so subsequent reads see the new
    # data without a process restart.

    def index_mtime(self) -> float:
        """Return the newest mtime across top-level entries in the index dir.

        A rising value means something changed on disk (new table, updated
        transaction log, rsync). Callers use this as a staleness signal.
        """
        try:
            entries = list(self._path.iterdir())
        except OSError:
            return 0.0
        if not entries:
            try:
                return self._path.stat().st_mtime
            except OSError:
                return 0.0
        latest = 0.0
        for e in entries:
            try:
                m = e.stat().st_mtime
            except OSError:
                continue
            if m > latest:
                latest = m
        return latest

    def reconnect(self) -> None:
        """Reopen the LanceDB connection so it observes on-disk changes.

        Call after an rsync or any out-of-process write to the index dir.
        Cached FTS-readiness is dropped because indices may have been
        rebuilt externally.
        """
        import lancedb

        self._db = lancedb.connect(str(self._path))
        self._fts_ready.clear()
        self._pub_date_ready.clear()
        self._episode_title_ready.clear()
        logger.info(f"IndexStore reconnected: {self._path}")

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
        t = self._db.open_table(name)
        self._ensure_pub_date_column(name, t)
        self._ensure_episode_title_backfill(name, t)
        return t

    def _ensure_pub_date_column(self, collection: str, table) -> None:
        """Add ``pub_date`` top-level column to pre-migration tables.

        Older tables stored the RSS publication date only in the ``meta``
        JSON blob. Range filters need a dedicated scalar column; this
        helper adds it + backfills.

        Skip decision uses a filesystem sentinel rather than the schema,
        so a process that crashed after ``add_columns`` but before the
        backfill finished retries on the next open.
        """
        if collection in self._pub_date_ready:
            return
        sentinel = self._path / f"{collection}.pub_date_col_v1"
        if sentinel.exists():
            self._pub_date_ready.add(collection)
            return
        try:
            schema_names = set(table.schema.names)
        except Exception:
            schema_names = set()
        if "pub_date" not in schema_names:
            try:
                table.add_columns({"pub_date": "CAST('' AS STRING)"})
            except Exception:
                logger.opt(exception=True).warning(
                    f"Could not add pub_date column to '{collection}'"
                )
                return
        try:
            rows = (
                table.search().select(["episode", "meta"]).limit(10_000_000).to_list()
            )
            per_ep: dict[str, str] = {}
            for r in rows:
                ep = r.get("episode")
                if not ep or ep in per_ep:
                    continue
                try:
                    meta = json.loads(r.get("meta") or "{}")
                except Exception:
                    continue
                norm = _normalize_pub_date(
                    meta.get("pub_date") or meta.get("rss_pub_date")
                )
                if norm:
                    per_ep[ep] = norm
            for ep, dt in per_ep.items():
                table.update(
                    where=f"episode = '{_escape(ep)}'",
                    values={"pub_date": dt},
                )
            sentinel.touch()
            logger.info(
                f"pub_date column backfilled for '{collection}' "
                f"({len(per_ep)} episodes)"
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"pub_date backfill skipped for '{collection}'"
            )
        self._pub_date_ready.add(collection)

    def _ensure_episode_title_backfill(self, collection: str, table) -> None:
        """Backfill ``episode_title`` in chunk meta from ``.episode_meta.json``.

        Some episodes were indexed before RSS metadata was merged into the
        transcript (e.g. a corrupt or empty ``.episode_meta.json``), leaving
        ``meta.episode_title`` unset. ``episode_display`` then falls back to a
        humanised stem, which leaks a normalised slug into search results and
        MCP replies. This one-shot pass walks the stems with a missing title,
        reloads each episode's meta file (``load_episode_meta`` now self-heals
        from the show-level feed cache), and promotes the title into every
        chunk's JSON meta blob. Skip-gated by a filesystem sentinel — if the
        resolver is not registered yet (bot/CLI contexts) we simply return.
        """
        if collection in self._episode_title_ready:
            return
        sentinel = self._path / f"{collection}.episode_title_v1"
        if sentinel.exists():
            self._episode_title_ready.add(collection)
            return
        resolver = type(self)._show_folder_resolver
        if resolver is None:
            return  # try again next time once the API has registered one

        info = self.get_collection_info(collection)
        show_name = (info or {}).get("show", "")
        if not show_name:
            return
        show_folder = resolver(show_name)
        if not show_folder:
            return
        try:
            from podcodex.ingest.rss import (
                episode_stem as rss_episode_stem,
                load_feed_cache,
            )
        except Exception:
            return

        try:
            row_count = int(table.count_rows())
        except Exception:
            row_count = -1
        if row_count == 0:
            # Nothing to heal; drop the sentinel so a freshly-indexed table
            # isn't rescanned on every later open.
            try:
                sentinel.touch()
            except OSError:
                pass
            self._episode_title_ready.add(collection)
            return

        try:
            rows = (
                table.search()
                .select(["episode", "chunk_index", "meta"])
                .limit(10_000_000)
                .to_list()
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"episode_title backfill scan skipped for '{collection}'"
            )
            return

        # Group all chunks by stem in one pass so we can detect missing titles
        # and rewrite meta blobs without a second full-table scan per stem.
        chunks_by_stem: dict[str, list[tuple[int, dict]]] = {}
        for r in rows:
            ep = r.get("episode")
            if not ep:
                continue
            try:
                meta = json.loads(r.get("meta") or "{}")
            except Exception:
                meta = {}
            chunks_by_stem.setdefault(ep, []).append(
                (int(r.get("chunk_index", -1)), meta)
            )

        # Feed cache is the only authoritative fallback for a healed title —
        # read it once here rather than once per stem inside the loop.
        show_folder_path = Path(show_folder)
        cached_eps = load_feed_cache(show_folder_path) or []
        stem_to_title: dict[str, str] = {}
        for ep in cached_eps:
            try:
                stem = rss_episode_stem(ep, show_folder_path)
            except Exception:
                continue
            if ep.title and ep.title != stem:
                stem_to_title[stem] = ep.title

        healed = 0
        for stem, chunks in chunks_by_stem.items():
            if any(m.get("episode_title") for _, m in chunks):
                continue
            title = stem_to_title.get(stem)
            if not title:
                continue
            for ci, meta in chunks:
                meta["episode_title"] = title
                try:
                    table.update(
                        where=(f"episode = '{_escape(stem)}' AND chunk_index = {ci}"),
                        values={"meta": json.dumps(meta, ensure_ascii=False)},
                    )
                except Exception:
                    logger.opt(exception=True).debug(
                        f"episode_title update failed for {collection}/{stem}"
                    )
            healed += 1

        try:
            sentinel.touch()
        except OSError:
            pass
        if healed:
            logger.info(
                f"episode_title backfilled for '{collection}' ({healed} episodes)"
            )
        self._episode_title_ready.add(collection)

    def _passwords_table(self):
        if _SHOW_PASSWORDS_TABLE in self._table_names():
            return self._db.open_table(_SHOW_PASSWORDS_TABLE)
        return self._db.create_table(
            _SHOW_PASSWORDS_TABLE, schema=_SHOW_PASSWORDS_SCHEMA
        )

    # ── Show password management ─────────────────────────────────────────

    def get_show_passwords(self) -> dict[str, str]:
        """Return ``{show_name: password_hash}`` for all password-protected shows."""
        if _SHOW_PASSWORDS_TABLE not in self._table_names():
            return {}
        rows = self._passwords_table().search().limit(10_000).to_list()
        return {r["show"]: r["password_hash"] for r in rows if r.get("show")}

    def set_show_password(self, show: str, password_hash: str) -> None:
        """Set or replace the password hash for a show."""
        t = self._passwords_table()
        t.delete(f"show = '{_escape(show)}'")
        t.add([{"show": show, "password_hash": password_hash}])
        logger.debug(f"Password set for show {show!r}")

    def delete_show_password(self, show: str) -> None:
        """Remove password protection for a show (makes it public)."""
        if _SHOW_PASSWORDS_TABLE not in self._table_names():
            return
        self._passwords_table().delete(f"show = '{_escape(show)}'")
        logger.debug(f"Password removed for show {show!r}")

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
        existing = meta.search().where(f"name = '{_escape(name)}'").limit(1).to_list()
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
        self._fts_ready.discard(name)
        logger.debug(f"Deleted collection '{name}'")

    def get_collection_info(self, name: str) -> dict | None:
        """Return ``{show, model, chunker, dim}`` or ``None`` if unknown."""
        return self.get_all_collection_info().get(name)

    def get_all_collection_info(self) -> dict[str, dict]:
        """Return ``{collection_name: {show, model, chunker, dim}}`` in one scan."""
        rows = self._collections_table().search().limit(10_000).to_list()
        return {
            r["name"]: {
                "show": r["show"],
                "model": r["model"],
                "chunker": r["chunker"],
                "dim": int(r["dim"]),
            }
            for r in rows
            if r.get("name")
        }

    def count_rows(self, collection: str) -> int:
        """Total chunk count for a collection. 0 if the collection is empty/unknown."""
        if not self.collection_exists(collection):
            return 0
        return int(self._table(collection).count_rows())

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
        self._fts_ready.discard(collection)
        logger.debug(f"Deleted episode '{episode}' from '{collection}'")

    def list_episodes(self, collection: str) -> list[str]:
        """Return a sorted list of distinct episodes in the collection.

        Args:
            collection: Collection name.
        """
        return self._distinct(collection, "episode")

    def episode_count(self, collection: str) -> int:
        """Number of distinct episodes in the collection (no sort).

        Cheaper than ``len(list_episodes(...))`` — skips the sort step,
        suitable for dashboards / tool responses where order does not
        matter.
        """
        if not self.collection_exists(collection):
            return 0
        t = self._table(collection)
        rows = t.search().select(["episode"]).limit(1_000_000).to_list()
        return len({r["episode"] for r in rows if r.get("episode")})

    def list_episode_titles(self, collection: str) -> dict[str, str]:
        """Return ``{episode_stem: rss_title}`` for episodes that have one.

        Episodes with no RSS title are omitted; the caller should fall back to
        a humanized stem for those.
        """
        return {
            s["episode"]: s["episode_title"]
            for s in self.get_episode_stats(collection)
            if s["episode_title"]
        }

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
                    "pub_date": _normalize_pub_date(chunk.get("pub_date")) or "",
                    "dominant_speaker": str(speaker),
                    "start": float(chunk.get("start", 0.0)),
                    "end": float(chunk.get("end", 0.0)),
                    "text": str(chunk.get("text", "")),
                    "vector": embeddings[i].astype(np.float32).tolist(),
                    "meta": json.dumps(meta),
                }
            )
        t.add(rows)
        self._fts_ready.discard(collection)
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

    def load_chunks_with_vector_stats(
        self, collection: str, episode: str
    ) -> list[dict]:
        """Return chunks for an episode with per-vector health stats attached.

        Powers the index inspector. Loads the vector column, computes
        L2 norm, zero-fraction, min/max per chunk, then drops the raw
        vector before returning so the response stays small.
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
                    "vector",
                    "meta",
                ]
            )
            .limit(100_000)
            .to_list()
        )
        out: list[dict] = []
        for r in rows:
            v = np.asarray(r.get("vector") or [], dtype=np.float32)
            chunk = _row_to_chunk(r)
            if v.size:
                chunk["vector_norm"] = float(np.linalg.norm(v))
                chunk["vector_zero_frac"] = float((v == 0.0).mean())
                chunk["vector_min"] = float(v.min())
                chunk["vector_max"] = float(v.max())
                chunk["vector_mean"] = float(v.mean())
                chunk["vector_std"] = float(v.std())
            else:
                chunk["vector_norm"] = 0.0
                chunk["vector_zero_frac"] = 1.0
                chunk["vector_min"] = 0.0
                chunk["vector_max"] = 0.0
                chunk["vector_mean"] = 0.0
                chunk["vector_std"] = 0.0
            out.append(chunk)
        out.sort(key=lambda c: c.get("chunk_index", 0))
        return out

    def get_chunk_window(
        self,
        collection: str,
        episode: str,
        chunk_index: int,
        window: int = 3,
    ) -> list[dict]:
        """Return chunks around a center chunk_index, ordered by position.

        Used to expand a retrieved hit with its neighbors so callers
        (MCP ``get_context``) get enough surrounding dialogue for
        grounded answers.

        Args:
            collection: Collection name.
            episode: Episode identifier.
            chunk_index: The center chunk's position in the episode.
            window: Number of chunks to include on each side. Clamped to
                ``[0, ...]``. ``0`` returns only the center chunk.

        Returns:
            Slice of the episode's chunks covering
            ``[chunk_index - window, chunk_index + window]``, inclusive,
            sorted by ``chunk_index``. Empty if the episode or center
            chunk is not found.
        """
        window = max(0, window)
        chunks = self.load_chunks_no_embeddings(collection, episode)
        if not chunks:
            return []
        center = next(
            (i for i, c in enumerate(chunks) if c.get("chunk_index") == chunk_index),
            -1,
        )
        if center < 0:
            return []
        lo = max(0, center - window)
        hi = min(len(chunks), center + window + 1)
        return chunks[lo:hi]

    def load_all_chunks(
        self,
        collection: str,
        episode: str | None = None,
        *,
        episodes: list[str] | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> list[dict]:
        """Load all chunks (without vectors) in a collection.

        Args:
            collection: Collection name.
            episode: If set, restrict to this one episode.
            episodes: Alternative to ``episode`` — restrict to a list of stems.
            pub_date_min, pub_date_max: Inclusive date bounds (``YYYY-MM-DD``).

        Returns:
            List of chunk dicts ordered by ``(episode, chunk_index)``.
        """
        if not self.collection_exists(collection):
            return []
        t = self._table(collection)
        q = t.search()
        clause = _build_where(
            episode=episode,
            episodes=episodes,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
        if clause:
            q = q.where(clause)
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
        episodes: list[str] | None = None,
        source: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> list[dict]:
        """ANN vector search with optional pre-filters.

        Args:
            collection: Collection name.
            query_vec: Query embedding (float32 1-D).
            top_k: Maximum results.
            episode, source, speaker: SQL-style equality filters.
            episodes: Alternative to ``episode`` — restrict to a list.
            pub_date_min, pub_date_max: Inclusive date bounds.

        Returns:
            List of chunk dicts with cosine ``score`` (``1 - distance``)
            attached.
        """
        if not self.collection_exists(collection):
            return []
        t = self._table(collection)
        # ``metric="cosine"`` matters for unnormalized embedders (Perplexity's
        # context model emits int8-quantized unnormalized vectors with
        # norms ~500). LanceDB's default is L2, which would rank by
        # magnitude rather than direction and break similarity. For
        # pre-normalized embedders (E5, BGE) cosine and L2 produce
        # identical rankings, so this is safe for everyone.
        q = t.search(
            query_vec.astype(np.float32).tolist(),
            query_type="vector",
            vector_column_name="vector",
        ).metric("cosine")
        clause = _build_where(
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
        if clause:
            q = q.where(clause)
        rows = q.limit(top_k).to_list()
        out: list[dict] = []
        for r in rows:
            # LanceDB cosine distance is 1 - cosine_sim, so 1 - distance is
            # the cosine similarity in [0, 1] (assuming non-negative dot).
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
        episodes: list[str] | None = None,
        source: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
        fuzziness: int = 0,
    ) -> list[dict]:
        """Full-text search (Tantivy, tokenized).

        Creates the FTS index lazily on first call per collection.

        Args:
            collection: Collection name.
            query: Free-text query.
            top_k: Maximum results.
            episode, source, speaker: SQL-style equality filters.
            episodes: Alternative to ``episode`` — restrict to a list.
            pub_date_min, pub_date_max: Inclusive date bounds.
            fuzziness: Levenshtein edit distance for fuzzy matching (0 = exact).

        Returns:
            List of chunk dicts with BM25 ``score`` attached.
        """
        if not self.collection_exists(collection):
            return []
        t = self._table(collection)
        self._ensure_fts(collection, t)
        if fuzziness > 0:
            from lancedb.query import MatchQuery

            search_input = MatchQuery(query, "text", fuzziness=fuzziness)
        else:
            search_input = query
        q = t.search(search_input, query_type="fts")
        clause = _build_where(
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
        if clause:
            q = q.where(clause)
        try:
            rows = q.limit(top_k).to_list()
        except Exception:
            logger.opt(exception=True).warning("FTS query failed — treating as empty")
            return []
        return [
            {**_row_to_chunk(r), "score": float(r.get("_score", 0.0))} for r in rows
        ]

    def search_literal(
        self,
        collection: str,
        query: str,
        *,
        episode: str | None = None,
        episodes: list[str] | None = None,
        source: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
        max_dist: int = 1,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Three-tier phrase search: exact, accent variant, near-typo.

        FTS ``~{max_dist}`` pre-filters candidates; Python re-verifies with
        exact Levenshtein ≤ ``max_dist`` after accent folding.

        - ``exact``:       ``query.lower()`` is a substring of ``text.lower()``
        - ``accent_only``: accent-folded query is a substring of folded text
        - ``fuzzy_only``:  single-word query with a word within ``max_dist``
                           edits (after accent folding) in the chunk text
        """
        if not self.collection_exists(collection):
            return [], [], []

        folded_q = fold_text(query)  # hyphens → spaces + accent-fold
        query_lower = query.lower()

        # Tokenize the same way Tantivy does: split on non-word characters.
        # Only tokens ≥ 3 chars are used for FTS intersection — shorter tokens
        # (e.g. "c" from "c'est") are far too common to discriminate chunks.
        all_tokens = [t for t in _WORD_RE.split(query) if t]
        fts_tokens = list(dict.fromkeys(t for t in all_tokens if len(t) >= 3))
        single_word = len(fts_tokens) == 1

        # FTS pre-filter: fuzziness=2 via MatchQuery, per token. Tantivy keeps
        # accents in the index and its fuzzy query has a non-zero effective
        # prefix_length, so an accent-swap on the leading char (e.g. "etres"
        # → "êtres") won't match even at distance=1. We search both the
        # original token and its accent-folded form and union the hits to
        # cover both "accented query, no-accent typing" directions.
        # Multi-word: intersect per-token chunk hits so every token must
        # appear (approximately) in the same chunk.
        def _fts_token(token: str) -> list[dict]:
            hits = self.search_fts(
                collection,
                token,
                10_000,
                episode=episode,
                episodes=episodes,
                source=source,
                speaker=speaker,
                pub_date_min=pub_date_min,
                pub_date_max=pub_date_max,
                fuzziness=2,
            )
            ft = fold_text(token)
            if ft and ft != token:
                seen = {_chunk_key(h) for h in hits}
                for h in self.search_fts(
                    collection,
                    ft,
                    10_000,
                    episode=episode,
                    episodes=episodes,
                    source=source,
                    speaker=speaker,
                    pub_date_min=pub_date_min,
                    pub_date_max=pub_date_max,
                    fuzziness=2,
                ):
                    k = _chunk_key(h)
                    if k not in seen:
                        hits.append(h)
                        seen.add(k)
            return hits

        if len(fts_tokens) == 1:
            candidates = _fts_token(fts_tokens[0])
        else:
            hit_count: Counter = Counter()
            chunk_map: dict[str, dict] = {}
            for token in fts_tokens:
                for h in _fts_token(token):
                    key = _chunk_key(h)
                    hit_count[key] += 1
                    chunk_map.setdefault(key, h)
            candidates = [
                chunk_map[k] for k, n in hit_count.items() if n >= len(fts_tokens)
            ]

        exact: list[dict] = []
        accent_only: list[dict] = []
        fuzzy_only: list[dict] = []
        for c in candidates:
            text = c.get("text", "")
            lower = text.lower()
            if query_lower in lower:
                idx = lower.find(query_lower)
                exact.append(
                    {**c, "score": 1.0, "match_text": text[idx : idx + len(query)]}
                )
            else:
                folded_t = fold_text(text)
                if folded_q in folded_t:
                    span = _accent_span(text, folded_t, folded_q) or query
                    accent_only.append(
                        {
                            **c,
                            "score": _accent_score(query, span),
                            "match_text": span,
                        }
                    )
                else:
                    if single_word:
                        result = _fuzzy_match(fts_tokens[0], text, max_dist)
                        if result is not None:
                            score, span = result
                            fuzzy_only.append({**c, "score": score, "match_text": span})
                    else:
                        # Tolerance ~12% of phrase length: short queries stay
                        # strict, long ones accept one or two real typos.
                        phrase_max = max(1, len(folded_q) // 8)
                        hit = _approx_substring(folded_q, folded_t, phrase_max)
                        if hit is not None:
                            d, fs, fe = hit
                            if len(folded_t) == len(text):
                                span = text[fs:fe]
                            else:
                                span = (
                                    _find_original_span(text, folded_t[fs:fe])
                                    or folded_t[fs:fe]
                                )
                            score = 1.0 - d / max(len(folded_q), 1)
                            fuzzy_only.append({**c, "score": score, "match_text": span})

        exact.sort(key=lambda c: (c.get("episode", ""), c.get("start", 0.0)))
        accent_only.sort(key=lambda c: (c.get("episode", ""), c.get("start", 0.0)))
        fuzzy_only.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        return exact, accent_only, fuzzy_only

    def _ensure_fts(self, collection: str, table) -> None:
        """Ensure an ASCII-folding FTS index exists on the ``text`` column.

        ``ascii_folding=True`` + ``lower_case=True`` normalize both indexed
        text and the query, so accented terms in the corpus match unaccented
        queries (e.g. "etres" → "êtres"). Without folding, tantivy's fuzzy
        query has an effective leading-char prefix constraint that prevents
        1-edit matches when the first char is the accented one.

        A sentinel file (``.fts_folded_v1``) records the one-time upgrade so
        we rebuild only when a pre-existing non-folding index is found, not
        on every process restart.
        """
        if collection in self._fts_ready:
            return
        sentinel = self._path / f"{collection}.fts_folded_v1"
        try:
            existing = {idx.name for idx in table.list_indices()}
        except Exception:
            existing = set()
        has_text_idx = any("text" in n.lower() for n in existing)
        needs_rebuild = has_text_idx and not sentinel.exists()
        try:
            if not has_text_idx or needs_rebuild:
                table.create_fts_index(
                    "text",
                    replace=needs_rebuild,
                    ascii_folding=True,
                    lower_case=True,
                    stem=False,
                    remove_stop_words=False,
                )
                sentinel.touch()
                if needs_rebuild:
                    logger.info(
                        f"FTS index rebuilt with ASCII folding for {collection}"
                    )
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

    def speaker_stats_multi(self, collections: list[str]) -> list[dict]:
        """Aggregate :meth:`speaker_stats` across multiple collections.

        Merges per-speaker counts/duration/episodes so callers can pass
        every collection for a show (or every default-model collection)
        and get a single ranking. Used by the Discord bot's ``/speakers``
        command and the MCP ``speaker_stats`` tool.
        """
        merged: dict[str, dict] = {}
        for col in collections:
            for row in self.speaker_stats(col):
                m = merged.setdefault(
                    row["speaker"],
                    {"chunk_count": 0, "total_duration": 0.0, "episodes": 0},
                )
                m["chunk_count"] += row["chunk_count"]
                m["total_duration"] += row["total_duration"]
                m["episodes"] += row["episodes"]
        return sorted(
            (
                {
                    "speaker": sp,
                    "chunk_count": m["chunk_count"],
                    "total_duration": round(m["total_duration"], 2),
                    "episodes": m["episodes"],
                }
                for sp, m in merged.items()
            ),
            key=lambda x: x["chunk_count"],
            reverse=True,
        )

    def speaker_stats(self, collection: str) -> list[dict]:
        """Return per-speaker chunk counts / duration / episode reach.

        One pass over the collection — no per-speaker search required.
        Each record: ``{speaker, chunk_count, total_duration, episodes}``
        sorted by ``chunk_count`` descending. Chunks with no
        ``dominant_speaker`` are skipped (they'd tell us nothing about
        who is speaking).

        ``total_duration`` is a rough proxy for airtime — it sums each
        chunk's ``(end - start)`` attributed to its dominant speaker.
        """
        if not self.collection_exists(collection):
            return []
        rows = (
            self._table(collection)
            .search()
            .select(["episode", "dominant_speaker", "start", "end"])
            .limit(1_000_000)
            .to_list()
        )
        groups: dict[str, dict] = {}
        for r in rows:
            sp = r.get("dominant_speaker")
            if not sp:
                continue
            g = groups.setdefault(
                sp, {"chunk_count": 0, "total_duration": 0.0, "episodes": set()}
            )
            g["chunk_count"] += 1
            start = float(r.get("start") or 0.0)
            end = float(r.get("end") or 0.0)
            if end > start:
                g["total_duration"] += end - start
            ep = r.get("episode")
            if ep:
                g["episodes"].add(ep)
        return sorted(
            (
                {
                    "speaker": sp,
                    "chunk_count": g["chunk_count"],
                    "total_duration": round(g["total_duration"], 2),
                    "episodes": len(g["episodes"]),
                }
                for sp, g in groups.items()
            ),
            key=lambda x: x["chunk_count"],
            reverse=True,
        )

    def get_episode_stats(self, collection: str) -> list[dict]:
        """Return per-episode metadata.

        Each record:
        ``{episode, episode_title, pub_date, episode_number, description,
        source, chunk_count, duration, speakers}``.

        ``pub_date`` comes from the top-level column (migrated from legacy
        ``meta`` payloads on first open). ``description``,
        ``episode_number`` and ``source`` are pulled from the first chunk
        that carries them.
        """
        groups = self._aggregate_episodes(
            collection,
            with_speakers=True,
            meta_fields=("episode_title", "episode_number", "description"),
        )
        return [_episode_group_to_dict(ep, g) for ep, g in sorted(groups.items())]

    def get_episode(self, collection: str, episode: str) -> dict | None:
        """Return metadata for one episode, or ``None`` if not present.

        Same shape as :meth:`get_episode_stats` entries. Filters by
        episode at the query layer so single-episode lookups don't
        scan the full table.
        """
        if not episode:
            return None
        groups = self._aggregate_episodes(
            collection,
            with_speakers=True,
            meta_fields=("episode_title", "episode_number", "description"),
            where=f"episode = '{_escape(episode)}'",
        )
        if episode not in groups:
            return None
        return _episode_group_to_dict(episode, groups[episode])

    def list_episodes_filtered(
        self,
        collection: str,
        *,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
        title_contains: str | None = None,
        with_detail: bool = False,
    ) -> list[dict]:
        """Return a per-episode list for browsing UIs.

        Default record: ``{episode, episode_title, pub_date, episode_number,
        chunk_count, duration}`` — lightweight for UI pickers.

        ``with_detail=True`` additionally includes ``speakers`` (sorted list)
        and ``description``, at the cost of reading more columns. Used by
        callers that need a browseable catalogue without an extra
        :meth:`get_episode` round-trip per stem (MCP ``list_episodes``).
        """
        clause = _build_where(pub_date_min=pub_date_min, pub_date_max=pub_date_max)
        fields = (
            ("episode_title", "episode_number", "description")
            if with_detail
            else (
                "episode_title",
                "episode_number",
            )
        )
        groups = self._aggregate_episodes(
            collection,
            with_speakers=with_detail,
            meta_fields=fields,
            where=clause or None,
        )
        needle = (title_contains or "").strip().lower()
        out: list[dict] = []
        for ep in sorted(groups):
            g = groups[ep]
            if needle and needle not in f"{g['episode_title']} {ep}".lower():
                continue
            record: dict = {
                "episode": ep,
                "episode_title": g["episode_title"],
                "pub_date": g["pub_date"],
                "episode_number": g["episode_number"],
                "chunk_count": g["chunk_count"],
                "duration": g["duration"],
            }
            if with_detail:
                speakers = g.get("speakers")
                record["speakers"] = sorted(speakers) if speakers else []
                record["description"] = g.get("description", "")
            out.append(record)
        return out

    def _aggregate_episodes(
        self,
        collection: str,
        *,
        with_speakers: bool,
        meta_fields: tuple[str, ...],
        where: str | None = None,
    ) -> dict[str, dict]:
        """Scan a collection once, producing a per-episode accumulator dict.

        Shared backbone of :meth:`get_episode_stats`, :meth:`get_episode`,
        and :meth:`list_episodes_filtered`. Columns selected adapt to the
        fields the caller needs; meta JSON is only parsed when at least
        one requested ``meta_fields`` value is still missing.
        """
        if not self.collection_exists(collection):
            return {}
        cols = ["episode", "pub_date", "end", "meta"]
        if with_speakers or "source" in meta_fields:
            cols.append("source")
        if with_speakers:
            cols.append("dominant_speaker")
        q = self._table(collection).search()
        if where:
            q = q.where(where)
        rows = q.select(cols).limit(1_000_000).to_list()
        defaults: dict = {
            "chunk_count": 0,
            "duration": 0.0,
            "pub_date": "",
            "source": "",
        }
        if with_speakers:
            defaults["speakers"] = None  # set() inserted per-group below
        for field in meta_fields:
            defaults[field] = None if field == "episode_number" else ""
        groups: dict[str, dict] = {}
        for r in rows:
            ep = r.get("episode")
            if not ep:
                continue
            g = groups.get(ep)
            if g is None:
                g = {k: v for k, v in defaults.items()}
                if with_speakers:
                    g["speakers"] = set()
                groups[ep] = g
            g["chunk_count"] += 1
            end = float(r.get("end") or 0.0)
            if end > g["duration"]:
                g["duration"] = end
            if with_speakers:
                sp = r.get("dominant_speaker")
                if sp:
                    g["speakers"].add(sp)
            for col in ("source", "pub_date"):
                if col in g and not g[col] and r.get(col):
                    g[col] = r[col]
            missing = [
                f
                for f in meta_fields
                if (g[f] is None if f == "episode_number" else not g[f])
            ]
            if missing:
                try:
                    meta = json.loads(r.get("meta") or "{}")
                except Exception:
                    meta = {}
                for f in missing:
                    v = meta.get(f)
                    if f == "episode_number":
                        if v is not None:
                            g[f] = v
                    elif v:
                        g[f] = v
        return groups

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


def _episode_group_to_dict(episode: str, g: dict) -> dict:
    """Shape a ``_aggregate_episodes`` group into the public dict."""
    return {
        "episode": episode,
        "episode_title": g.get("episode_title", ""),
        "pub_date": g.get("pub_date", ""),
        "episode_number": g.get("episode_number"),
        "description": g.get("description", ""),
        "source": g.get("source", ""),
        "chunk_count": g["chunk_count"],
        "duration": g["duration"],
        "speakers": sorted(g.get("speakers") or ()),
    }


_PUB_DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_PUB_DATE_COMPACT_RE = re.compile(r"^\d{8}$")


def _normalize_pub_date(raw: Any) -> str | None:
    """Normalize a publication date to ``YYYY-MM-DD``.

    Accepts ISO 8601 (``2024-01-15``, ``2024-01-15T12:00:00Z``), RFC 2822
    (``Mon, 15 Jan 2024 12:00:00 GMT``), and YouTube's compact
    ``YYYYMMDD``. Returns ``None`` if *raw* is falsy or unparseable.
    Idempotent on already-normalized input.
    """
    if not raw:
        return None
    if not isinstance(raw, str):
        raw = str(raw)
    s = raw.strip()
    if not s:
        return None
    if _PUB_DATE_ISO_RE.match(s):
        return s[:10]
    if _PUB_DATE_COMPACT_RE.match(s):
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    try:
        dt = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        dt = None
    if dt is not None:
        return dt.date().isoformat()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _build_where(
    *,
    episode: str | None = None,
    episodes: list[str] | None = None,
    source: str | None = None,
    speaker: str | None = None,
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
) -> str:
    """Build a LanceDB WHERE clause from equality / range filters.

    ``episodes`` (list) takes precedence over ``episode`` (scalar) when
    both are given. Dates are normalized to ``YYYY-MM-DD``; invalid
    inputs raise ``ValueError``.
    """
    parts: list[str] = []
    if episodes:
        cleaned = [e for e in episodes if e]
        if cleaned:
            quoted = ", ".join(f"'{_escape(e)}'" for e in cleaned)
            parts.append(f"episode IN ({quoted})")
    elif episode:
        parts.append(f"episode = '{_escape(episode)}'")
    if source:
        parts.append(f"source = '{_escape(source)}'")
    if speaker:
        parts.append(f"dominant_speaker = '{_escape(speaker)}'")
    if pub_date_min:
        norm = _normalize_pub_date(pub_date_min)
        if not norm:
            raise ValueError(f"Invalid pub_date_min: {pub_date_min!r}")
        parts.append(f"pub_date >= '{norm}'")
    if pub_date_max:
        norm = _normalize_pub_date(pub_date_max)
        if not norm:
            raise ValueError(f"Invalid pub_date_max: {pub_date_max!r}")
        parts.append(f"pub_date <= '{norm}'")
    return " AND ".join(parts)


_FILTER_COLUMNS = (
    "chunk_index",
    "show",
    "episode",
    "source",
    "pub_date",
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


# ── Process-wide singleton ───────────────────────────────────────────────


@cache
def get_index_store() -> IndexStore:
    """Process-wide cached IndexStore using the default path resolution.

    Path resolution lives in ``_resolve_default_index_path`` — env var,
    canonical ``<data_dir>/index``, legacy XDG fallback, repo-local. Callers
    that need a different path should instantiate ``IndexStore(custom_path)``
    directly.
    """
    return IndexStore()
