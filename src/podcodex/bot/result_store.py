"""podcodex.bot.result_store — durable + in-RAM cache of search result refs.

Discord pagination buttons must survive both the 5-minute view timeout and bot
restarts. Instead of keeping rendered embeds (or whole chunks) alive in memory,
we store only the *references* needed to re-fetch each hit from LanceDB, keyed by
a short ``search_id`` baked into the button custom_ids.

Two tiers:
  1. RAM cache with a short TTL: hot path, instant, self-draining.
  2. SQLite: durable, survives restart, 30-day eviction.

The transcript text is never stored here; it is re-fetched from the index (see
``ui._fetch_episode_chunks``). Only the search-time scalars LanceDB lacks (score,
fuzzy/accent flags, match highlight) live in this cache.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

_RETENTION_SECONDS = 30 * 24 * 3600  # SQLite rows older than this are evicted
_RAM_TTL_SECONDS = 600  # hot in-RAM cache window (10 min)
_TOKEN_BYTES = 6  # 12-hex-char token; fits the custom_id budget with room to spare


@dataclass(slots=True)
class ResultRef:
    """Pointer to one search hit plus the search-time scalars LanceDB lacks.

    ``episode_title`` is a display-only convenience cached so the jump dropdown
    can label options without re-fetching every episode; it is deduped per
    episode on disk (see :func:`_encode_refs`).
    """

    collection: str
    episode: str
    chunk_index: int
    score: float = 0.0
    fuzzy_match: bool = False
    accent_match: bool = False
    match_text: str | None = None
    episode_title: str = ""

    @property
    def flags(self) -> int:
        """Bit-pack the two match booleans: bit0 = fuzzy, bit1 = accent."""
        return (1 if self.fuzzy_match else 0) | (2 if self.accent_match else 0)


@dataclass(slots=True)
class CachedSearch:
    """A cached, paginated message.

    Either ``refs`` (chunk-backed search/exact/random results, text re-fetched
    from LanceDB) or ``embeds`` (self-contained pages like the ``/shows`` episode
    list, stored verbatim as ``discord.Embed`` dicts) is populated, never both.
    """

    kind: str  # 'search' | 'exact' | 'random' | 'list'
    label: str
    query: str
    refs: list[ResultRef] = field(default_factory=list)
    embeds: list[dict] = field(default_factory=list)


def _intern(values: list[str]) -> tuple[list[str], dict[str, int]]:
    """Return ordered-unique list and a value -> index lookup."""
    uniq = list(dict.fromkeys(values))
    return uniq, {v: i for i, v in enumerate(uniq)}


def _encode_refs(refs: list[ResultRef]) -> dict:
    """Serialize refs, interning collection and episode names.

    Layout: ``{"cols": [...], "eps": [...], "ept": [...], "rows": [[col_idx,
    ep_idx, chunk_index, score, flags, match_text], ...]}``. Collections and
    episodes repeat across the hits of one search, so each is interned into its
    own list and referenced by index; ``ept`` holds one display title per
    distinct episode, aligned to ``eps``.
    """
    cols, col_idx = _intern([r.collection for r in refs])
    eps, ep_idx = _intern([r.episode for r in refs])
    titles = {r.episode: r.episode_title for r in refs}
    eptitles = [titles.get(e, "") for e in eps]
    rows = [
        [
            col_idx[r.collection],
            ep_idx[r.episode],
            r.chunk_index,
            r.score,
            r.flags,
            r.match_text,
        ]
        for r in refs
    ]
    return {"cols": cols, "eps": eps, "ept": eptitles, "rows": rows}


def _decode_refs(data: dict) -> list[ResultRef]:
    cols: list[str] = data["cols"]
    eps: list[str] = data["eps"]
    ept: list[str] = data.get("ept") or [""] * len(eps)
    out: list[ResultRef] = []
    for ci, ei, chunk_index, score, flags, match_text in data["rows"]:
        out.append(
            ResultRef(
                collection=cols[ci],
                episode=eps[ei],
                chunk_index=chunk_index,
                score=score,
                fuzzy_match=bool(flags & 1),
                accent_match=bool(flags & 2),
                match_text=match_text,
                episode_title=ept[ei],
            )
        )
    return out


def _encode(search: CachedSearch) -> str:
    """Serialize a cached page to compact JSON (refs- or embeds-backed)."""
    if search.embeds:
        payload: dict = {"embeds": search.embeds}
    else:
        payload = _encode_refs(search.refs)
    return json.dumps(payload, separators=(",", ":"))


def _decode(payload: str) -> tuple[list[ResultRef], list[dict]]:
    data = json.loads(payload)
    if "embeds" in data:
        return [], data["embeds"]
    return _decode_refs(data), []


class SearchCacheStore:
    """Tiered RAM + SQLite store for paginated search-result references.

    Thread-safe: the bot runs searches in ``run_in_executor`` worker threads, so
    every connection touch is serialized under a single lock. Operations are
    sub-millisecond, so the lock is not a contention point.
    """

    def __init__(self, db_path: Path) -> None:
        self._lock = threading.Lock()
        # search_id -> (expiry_epoch, CachedSearch)
        self._ram: dict[str, tuple[float, CachedSearch]] = {}
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        logger.info(f"Search-result cache at {db_path}")

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS search_cache (
              search_id   TEXT PRIMARY KEY,
              created_at  INTEGER NOT NULL,
              kind        TEXT NOT NULL,
              label       TEXT NOT NULL,
              query       TEXT NOT NULL,
              payload     TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS idx_cache_created
              ON search_cache(created_at);
            """
        )

    def save(self, search: CachedSearch) -> str:
        """Persist a search and return its new ``search_id``."""
        now = int(time.time())
        payload = _encode(search)
        with self._lock:
            search_id = self._insert(now, search, payload)
            self._conn.execute(
                "DELETE FROM search_cache WHERE created_at < ?",
                (now - _RETENTION_SECONDS,),
            )
            self._prune_ram(now)
            self._ram[search_id] = (now + _RAM_TTL_SECONDS, search)
        return search_id

    def _insert(self, now: int, search: CachedSearch, payload: str) -> str:
        """Insert with a fresh token, retrying on the (vanishingly rare) clash."""
        for _ in range(5):
            search_id = secrets.token_hex(_TOKEN_BYTES)
            try:
                self._conn.execute(
                    "INSERT INTO search_cache "
                    "(search_id, created_at, kind, label, query, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (search_id, now, search.kind, search.label, search.query, payload),
                )
                return search_id
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("could not allocate a unique search_id")

    def load(self, search_id: str) -> CachedSearch | None:
        """Return the cached search, RAM first then SQLite, or None if gone."""
        now = time.time()
        with self._lock:
            hit = self._ram.get(search_id)
            if hit is not None:
                expiry, search = hit
                if expiry > now:
                    return search
                del self._ram[search_id]  # expired; fall through to SQLite

            row = self._conn.execute(
                "SELECT kind, label, query, payload FROM search_cache "
                "WHERE search_id = ?",
                (search_id,),
            ).fetchone()
            if row is None:
                return None
            kind, label, query, payload = row
            try:
                refs, embeds = _decode(payload)
            except (ValueError, KeyError, TypeError):
                # Corrupt or legacy-format row: treat as a cache miss so the
                # button reports "expired" instead of throwing in the callback.
                logger.warning(f"Dropping undecodable search_cache row {search_id}")
                self._conn.execute(
                    "DELETE FROM search_cache WHERE search_id = ?", (search_id,)
                )
                return None
            search = CachedSearch(kind, label, query, refs, embeds)
            self._ram[search_id] = (now + _RAM_TTL_SECONDS, search)
            return search

    def _prune_ram(self, now: float) -> None:
        """Drop expired RAM entries so the hot cache cannot grow unbounded."""
        expired = [sid for sid, (exp, _) in self._ram.items() if exp <= now]
        for sid in expired:
            del self._ram[sid]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
