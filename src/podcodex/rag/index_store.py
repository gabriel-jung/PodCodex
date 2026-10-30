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
import threading
import time
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from loguru import logger
from pydantic import ValidationError

from podcodex.core.constants import is_remote_artwork_url

from podcodex.core._utils import normalize_pub_date
from podcodex.rag.hit import Hit

_COLLECTIONS_TABLE = "_collections"

# Subdirectory a Lance table commits new versions into. Stat'd by index_mtime
# because the table directory above it never changes on append.
_LANCE_VERSIONS_DIR = "_versions"
# How long ``compact`` keeps a version after a newer one replaced it. Every
# read opens the table afresh at its latest version (``_table``), so a
# replaced version is only read by a query already in flight when the
# replacement committed; this outlasts any of those. Lance's own default, a
# week, kept a full copy of the table per compaction: 6.6 GB of history for
# a 94 MB table after one show's re-index.
_COMPACT_KEEP_HISTORY = timedelta(minutes=2)
_NO_PRUNING = timedelta(days=365 * 100)
# Lance reads its own clock a moment after ``_history_window`` reads this one;
# the margin keeps the boundary version through that gap.
_PRUNE_MARGIN = timedelta(seconds=1)


def _history_window(versions: list[dict], keep: timedelta) -> timedelta:
    """``cleanup_older_than`` that keeps every version replaced within *keep*.

    Lance ages a version from its own commit, not from when a newer one
    replaced it. A table idle for an hour holds a latest version an hour
    old, which a compaction then replaces: a plain ``keep`` window would
    delete it under a reader that opened it a moment ago. So the window
    stretches back to the oldest version still to keep: the first one whose
    successor committed within *keep*. Read *versions* after the compaction
    commits, so the version it replaced counts as just replaced.
    """
    ordered = sorted(versions, key=lambda v: v["version"])
    stamps = [v["timestamp"] for v in ordered]
    now = datetime.now(stamps[-1].tzinfo)  # Lance's stamps are naive local
    oldest_kept = stamps[-1]
    for stamp, successor in zip(stamps, stamps[1:]):
        if now - successor < keep:
            oldest_kept = stamp
            break
    return now - oldest_kept + _PRUNE_MARGIN


# Scalar columns re-applied over the meta blob when a row is inflated into a
# Hit, and the projection every no-vector read selects (plus "meta"). Keep the
# two uses in lockstep: a column missing from the select comes back as the
# field default, silently (this is how pub_date used to vanish from /random).
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


def _original_slice(text: str, fs: int, fe: int) -> str:
    """The slice of ``text`` that folds to ``fold_text(text)[fs:fe]``.

    Every step of ``fold_text`` is per character, so folding char by char
    gives an exact offset map. Comparing lengths does not: a ligature that
    expands ("œ" to "oe") and a character that is dropped ("ß") cancel out
    and shift every offset after them.
    """
    if text.isascii():  # every ASCII char folds to exactly one char
        return text[fs:fe] if 0 <= fs < fe <= len(text) else ""
    offsets: list[int] = []
    for i, ch in enumerate(text):
        if len(offsets) >= fe:
            break
        offsets.extend([i] * (1 if ch.isascii() else len(fold_text(ch))))
    if not 0 <= fs < fe <= len(offsets):
        return ""
    end = offsets[fe - 1] + 1
    # Keep combining accents that follow the last matched letter.
    while end < len(text) and unicodedata.combining(text[end]):
        end += 1
    return text[offsets[fs] : end]


def _accent_span(text: str, folded_t: str, folded_q: str) -> str | None:
    """Locate the original-text slice that accent-folds to ``folded_q``."""
    idx = folded_t.find(folded_q)
    if idx < 0:
        return None
    return _original_slice(text, idx, idx + len(folded_q)) or None


def _accent_score(query: str, span: str) -> float:
    """Similarity between query and its matched original-text span."""
    return SequenceMatcher(None, query.lower(), span.lower()).ratio()


# Chunk-meta fields read per episode by get_episode_stats and get_episode,
# which promise the same record shape.
_EPISODE_META_FIELDS = (
    "episode_title",
    "episode_number",
    "broadcast_number",
    "description",
    "artwork_url",
)

# Exact-tier score for a match inside a longer word ("william" in
# "Williams"): below the whole-word 1.0 so word hits rank first, above any
# realistic accent/fuzzy ratio so the tiers keep their order.
_EXACT_SUBSTRING_SCORE = 0.99


_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)
# Shortest query token worth an FTS probe (and a fuzzy one) on its own.
_MIN_PROBE_LEN = 3
# Terms a fuzzy FTS probe may expand to. Tantivy's default of 50 silently
# drops neighbours once a common word has more than 50 terms within reach,
# so a typo query missed most chunks holding a word one edit away.
_FTS_MAX_EXPANSIONS = 1000
# Per-collection marker files next to the tables (``<collection>.<suffix>``):
# the FTS rebuild and its folded predecessor, plus the markers of the removed
# pub_date and episode_title backfills, which old indexes still carry.
_SENTINEL_SUFFIXES = ("fts_v2", "fts_folded_v1", "pub_date_col_v1", "episode_title_v1")
# Seconds before an FTS index build that failed is attempted again.
_FTS_RETRY_S = 300.0


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


def _fuzzy_match(
    query: str,
    text: str,
    max_dist: int,
    cache: dict[str, float | None] | None = None,
) -> tuple[float, str] | None:
    """Return (similarity, matched original span) if any word in text is within
    max_dist edits of the folded single-word query, else None.

    *cache* maps a raw word to its similarity (``None`` when too far). Pass
    one dict for every chunk of a query: the vocabulary repeats across chunks,
    and without it a common typo query folded and compared a million words.
    """
    q = fold_text(query)
    if cache is None:
        cache = {}
    best: tuple[float, str] | None = None
    for m in _WORD_TOKEN_RE.finditer(text):
        raw = m.group(0)
        if raw not in cache:
            w = fold_text(raw)
            ratio = None
            if w and abs(len(w) - len(q)) <= max_dist:
                d = _levenshtein(q, w, max_dist)
                if d <= max_dist:
                    ratio = 1.0 - d / max(len(q), len(w), 1)
            cache[raw] = ratio
        ratio = cache[raw]
        if ratio is not None and (best is None or ratio > best[0]):
            best = (ratio, raw)
    return best


def chunk_row_key(episode: str, chunk_index: int) -> str:
    """Identity of one chunk within a collection: its episode and position.

    Not the start time: chunks beginning in the same turn share it, and every
    chunk of an untimed transcript starts at 0.0. ``search_literal`` intersects
    keys built from raw rows with keys built from hits, so both sides must
    come through here.
    """
    return f"{episode}|{int(chunk_index)}"


def _chunk_key(chunk: Hit) -> str:
    return chunk_row_key(chunk.episode, chunk.chunk_index)


def _norm_label(label: str) -> str:
    """How a display name is compared when it keys a legacy password row."""
    return (label or "").strip().lower()


def approx_substring(
    pattern: str, text: str, max_dist: int
) -> tuple[int, int, int] | None:
    """Best substring of *text* within ``max_dist`` edits of *pattern*.

    Same contract as ``_sellers``, which does the matching, but run only
    where a match can be. Pigeonhole: ``max_dist`` edits touch at most
    ``max_dist`` of ``max_dist + 1`` disjoint pieces of the pattern, so any
    match contains one piece verbatim. The DP then runs on a small window
    around each occurrence of a piece instead of the whole chunk, which is
    what made a multi-word literal search over a large show take seconds.
    """
    pieces = max_dist + 1
    size = len(pattern) // pieces
    if not size:
        return _sellers(pattern, text, max_dist)
    windows: list[tuple[int, int]] = []
    for k in range(pieces):
        p_lo = k * size
        piece = pattern[p_lo : (k + 1) * size if k < pieces - 1 else None]
        pos = text.find(piece)
        while pos != -1:
            lo = max(0, pos - p_lo - max_dist)
            hi = min(len(text), pos - p_lo + len(pattern) + max_dist)
            windows.append((lo, hi))
            pos = text.find(piece, pos + 1)
    best: tuple[int, int, int] | None = None
    for lo, hi in _merge_spans(windows):
        hit = _sellers(pattern, text[lo:hi], max_dist)
        if hit is not None and (best is None or hit[0] < best[0]):
            best = (hit[0], hit[1] + lo, hit[2] + lo)
    return best


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and merge overlapping ``[lo, hi)`` spans."""
    merged: list[tuple[int, int]] = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _sellers(pattern: str, text: str, max_dist: int) -> tuple[int, int, int] | None:
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
    """True when *path* holds actual index content.

    The ownership marker is skipped on purpose. It is written whenever a new
    index directory is created, so counting it would make an otherwise empty
    canonical index look populated, and ``_resolve_default_index_path`` would
    stop falling through to the repo-local candidates.
    """
    from podcodex.rag.index_origin import ORIGIN_FILENAME

    try:
        return path.is_dir() and any(e.name != ORIGIN_FILENAME for e in path.iterdir())
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
        # Stable show identity (``show.toml`` id). Empty on rows written
        # before ids existed; the migration fills them in, and readers fall
        # back to ``show`` while it is empty so a partially migrated index
        # (or one rsynced mid-migration) still resolves.
        pa.field("show_id", pa.string()),
        # Display label, carried into the index because the bot reads an
        # rsynced copy with no show.toml anywhere.
        pa.field("show", pa.string()),
        pa.field("model", pa.string()),
        pa.field("chunker", pa.string()),
        pa.field("dim", pa.int32()),
        pa.field("created_at", pa.string()),
        # Channel-level show artwork (from show.toml). The bot only sees the
        # index, so show art must travel inside it, not in the show folder.
        pa.field("artwork_url", pa.string()),
    ]
)

_SHOW_PASSWORDS_TABLE = "_show_passwords"


def reserved_index_names() -> frozenset[str]:
    """Names a collection may never take, because index state already owns them.

    The index root holds the two sidecar tables and the ownership marker
    next to the chunk tables, and on-disk entries are addressed by prefix
    (``<name>.lance`` for the table directory, ``<name>.<ext>`` for its
    sidecars). A collection called ``index_origin`` therefore claims
    ``index_origin.json``, and one called ``_show_passwords`` claims that
    table, so an importer that only checks for path traversal would let a
    crafted archive replace the very files its allowlist exists to protect.

    Returned as stems, not filenames: the collision is on the prefix.
    """
    from podcodex.rag.index_origin import ORIGIN_FILENAME

    return frozenset(
        {
            _COLLECTIONS_TABLE,
            _SHOW_PASSWORDS_TABLE,
            ORIGIN_FILENAME.rsplit(".", 1)[0],
        }
    )


_SHOW_PASSWORDS_SCHEMA = pa.schema(
    [
        pa.field("show_id", pa.string()),
        # Display label. A show can be password-protected before it is
        # indexed, so ``_collections`` may hold no row to look one up in.
        pa.field("show_label", pa.string()),
        # Legacy key, still written as a mirror of the label so an older bot
        # build reading a newer index keeps resolving protected shows.
        pa.field("show", pa.string()),
        pa.field("password_hash", pa.string()),  # "sha256:<hex>"
    ]
)


# Guards the one-time show-id migration; see migrate_to_show_ids_if_owner.
_MIGRATION_LOCK = threading.Lock()

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

# Meta fields whose "unset" is ``None`` (not the empty string), so both their
# default and their missing-test differ from string-valued meta fields.
_INT_META_FIELDS = frozenset({"episode_number", "broadcast_number"})


class IndexStore:
    """LanceDB-backed store for chunk text, metadata, and embeddings.

    Args:
        path: Directory holding the LanceDB database. When ``None``, resolved
            via :func:`_resolve_default_index_path`.
    """

    # Global resolver mapping a show name to its on-disk folder, set by the
    # API only (never the bot, which has no show folders). Its presence is
    # also the "am I the app" gate for the show-id migration and the
    # collection-meta heal.
    _show_folder_resolver: "callable | None" = None

    @classmethod
    def set_show_folder_resolver(cls, fn) -> None:
        """Register a callable ``(show_name) -> Path | None`` for the backfill."""
        cls._show_folder_resolver = fn

    def migrate_to_show_ids_if_owner(self) -> int:
        """Run the name-keyed to id-keyed migration, if this process may.

        Gated on the show-folder resolver, which only the API registers: the
        bot reads an rsynced index with no ``show.toml`` anywhere and would
        mint ids that diverge from the app's. Safe to call repeatedly.
        """
        if type(self)._show_folder_resolver is None:
            return 0
        from podcodex.rag.show_id_migration import migrate_index_to_show_ids

        # functools.cache runs its factory outside any lock, so two threadpool
        # workers racing the first index-touching request can each build a
        # store and migrate concurrently. The lock is process-wide, not
        # per-instance, for exactly that reason.
        try:
            with _MIGRATION_LOCK:
                return migrate_index_to_show_ids(self)
        except Exception:
            # Not "unchanged": rows and show.toml ids written before the
            # failure stay written. The migration is idempotent, so the next
            # open resumes from them.
            logger.opt(exception=True).warning(
                "show-id migration failed part way; it resumes on the next open"
            )
            return 0

    def __init__(self, path: Path | str | None = None):
        import lancedb

        if path is None:
            resolved, reason = _resolve_default_index_path()
            self._path = resolved
            logger.info(f"IndexStore opened: {self._path} ({reason})")
        else:
            self._path = Path(path)
            logger.info(f"IndexStore opened: {self._path} (explicit path)")
        # Recorded before mkdir: an index this process creates is owned by
        # this machine, while one that already has content belongs to
        # whoever stamped it (see rag/index_origin).
        was_empty = not _dir_has_data(self._path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._path))
        from podcodex.rag.index_origin import stamp_origin_if_new

        stamp_origin_if_new(self._path, was_empty=was_empty)
        # Serializes mutating ops (password set/delete, chunk saves,
        # episode/collection delete, FTS index builds) within this process:
        # the store is a process-wide singleton shared across threadpool
        # handlers, and LanceDB writes are not internally locked. It does not
        # reach index subprocesses, which write chunks from their own store,
        # so cross-process writes must be safe on their own (see
        # _retry_concurrent and the merge_insert in ensure_collection).
        # Reentrant because locked sections call _table() and other locked
        # helpers.
        self._write_lock = threading.RLock()
        # Per-thread metadata memo of read_snapshot().
        self._snap = threading.local()
        self._fts_ready: set[str] = set()
        # Collections whose FTS build failed, with the monotonic time of the
        # failure: retried after _FTS_RETRY_S, not on every probe (each would
        # rebuild and log again) and not never (the bot runs for weeks).
        self._fts_failed: dict[str, float] = {}
        # Chunk tables this process has written to and not yet compacted.
        self._uncompacted: set[str] = set()
        self._pub_date_ready: set[str] = set()
        # Old tables whose pub_date column could not be added (see _row_cols).
        self._no_pub_date: set[str] = set()
        self._collections_schema_ready = False
        self._passwords_schema_ready = False
        # (index_mtime, collection-info); collection metadata is read on
        # every MCP tool call, bot-access request, and vector search.
        self._collection_info_cache: tuple[float, dict[str, dict]] | None = None

    @property
    def path(self) -> Path:
        """Filesystem root of the LanceDB index directory."""
        return self._path

    def _require_owner(self, action: str) -> None:
        """Refuse a write when this index belongs to another machine.

        Applied to the records that survive only in the index and that two
        machines can meaningfully disagree about: passwords, collection
        identity and labels, and collection deletion. See
        ``rag/index_origin`` for why derived data is deliberately exempt.
        """
        from podcodex.rag.index_origin import require_owner

        require_owner(self._path, action)

    # ── External-change detection & reconnection ─────────────────────────
    #
    # The index directory can be replaced (rsync) or extended (desktop app
    # adds a show, API sets a password) while the bot holds a live LanceDB
    # connection. `index_mtime` lets callers cheaply detect such changes;
    # `reconnect` rebuilds the connection so subsequent reads see the new
    # data without a process restart.

    def index_mtime(self) -> float:
        """Return the newest mtime across the index dir's tables.

        A changed value means something changed on disk (new table, new
        manifest, rsync). Callers use this as a staleness signal, and must
        compare for inequality rather than growth: ``rsync -a`` stamps
        destination mtimes from the source, so the value can move backwards.

        Each table's ``_versions`` subdirectory is stat'd alongside the table
        directory itself. Every LanceDB commit writes a manifest there, and
        POSIX only bumps a directory's mtime when its own entries change, so a
        table directory stays frozen at creation time no matter how many
        episodes are appended. Top-level entries alone would miss every append.
        """
        try:
            entries = list(self._path.iterdir())
        except OSError:
            return 0.0
        try:
            # Seeds the empty-index case, and catches a table directory being
            # removed outright (which bumps the parent and nothing else).
            latest = self._path.stat().st_mtime
        except OSError:
            latest = 0.0
        for e in entries:
            # NotADirectoryError (an OSError) covers the sentinel files.
            for target in (e, e / _LANCE_VERSIONS_DIR):
                try:
                    m = target.stat().st_mtime
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
        self._fts_failed.clear()
        self._pub_date_ready.clear()
        self._no_pub_date.clear()
        logger.info(f"IndexStore reconnected: {self._path}")

    # ── Internal: collections metadata table ─────────────────────────────

    @contextmanager
    def read_snapshot(self) -> Iterator[None]:
        """Read the index's metadata once for a whole read-only fan-out.

        A search over N collections asked, per collection, for the table list,
        the collection info (a stat of every table directory) and a fresh
        table handle, so its fixed cost grew with N squared. Inside this block
        the calling thread reuses the first answer to each. Per thread, since
        the store is shared across request threads; re-entrant; never wrap a
        write in it, the table list would go stale.
        """
        if getattr(self._snap, "state", None) is not None:
            yield
            return
        self._snap.state = {"tables": None, "info": None, "handles": {}}
        try:
            yield
        finally:
            self._snap.state = None

    def _table_names(self) -> list[str]:
        """Return LanceDB table names as a plain list."""
        snap = getattr(self._snap, "state", None)
        if snap is not None and snap["tables"] is not None:
            return snap["tables"]
        # Some LanceDB versions wrap the names in a response object with a
        # ``.tables`` attribute; fall back to iterating the result directly.
        result = self._db.list_tables()
        names = list(getattr(result, "tables", result))
        if snap is not None:
            snap["tables"] = names
        return names

    def _open_or_create(self, name: str, schema: pa.Schema):
        """Open a table, creating it if absent, safe against another process.

        An index subprocess can create the same table at the same moment, and
        _write_lock does not reach it; see ``_retry_concurrent``.
        """

        def _attempt():
            if name in self._table_names():
                return self._db.open_table(name)
            return self._db.create_table(name, schema=schema, exist_ok=True)

        return _retry_concurrent(_attempt)

    def _collections_table(self):
        table = self._open_or_create(_COLLECTIONS_TABLE, _COLLECTIONS_SCHEMA)
        self._ensure_collections_schema(table)
        return table

    def _ensure_collections_schema(self, table) -> None:
        """Add later columns (``artwork_url``, ``show_id``) to older tables."""
        if self._collections_schema_ready:
            return
        try:
            present = set(table.schema.names)
            missing = {
                col: "CAST('' AS STRING)"
                for col in ("artwork_url", "show_id")
                if col not in present
            }
            if missing:
                table.add_columns(missing)
        except Exception:
            logger.opt(exception=True).warning("Could not add columns to _collections")
            return
        self._collections_schema_ready = True

    def _table(self, name: str):
        """Open an existing collection table; raise if missing."""
        snap = getattr(self._snap, "state", None)
        if snap is not None and name in snap["handles"]:
            return snap["handles"][name]
        if name not in self._table_names():
            raise KeyError(f"Collection '{name}' does not exist")
        t = self._db.open_table(name)
        self._ensure_pub_date_column(name, t)
        if snap is not None:
            snap["handles"][name] = t
        return t

    @classmethod
    def _folder_for_row(cls, entry: dict) -> Path | None:
        """Show folder of a ``_collections`` row, in the app; else None.

        Identity first; the label is only a fallback for rows the migration
        has not stamped yet, since two shows can share one.
        """
        resolver = cls._show_folder_resolver
        if resolver is None:
            return None
        from podcodex.ingest.show_registry import folder_for_id

        row_id = (entry.get("show_id") or "").strip()
        folder = folder_for_id(row_id) if row_id else None
        if folder is None and entry.get("show"):
            found = resolver(entry["show"])
            folder = Path(found) if found else None
        return folder

    def _ensure_pub_date_column(self, collection: str, table) -> None:
        """Give a table from before the scalar ``pub_date`` column an empty one.

        Only the schema: ``save_chunks`` writes the column, so without it
        re-indexing any episode of an old table would fail. Existing rows are
        not backfilled; they carry "" and drop out of date filters until
        their episode is re-indexed, so a show upgrades episode by episode
        (dates still display, from the meta blob via ``Hit.effective_pub_date``).
        """
        if collection in self._pub_date_ready:
            return
        try:
            if "pub_date" not in set(table.schema.names):
                table.add_columns({"pub_date": "CAST('' AS STRING)"})
                logger.info(f"pub_date column added to '{collection}' (empty)")
        except Exception:
            # Reads must not select a column the table lacks (_row_cols);
            # date filters on it still fail, which is the truth. Not retried
            # on every open (each would log again): reconnect() or a restart
            # tries again.
            logger.opt(exception=True).warning(
                f"Could not add pub_date column to '{collection}'"
            )
            self._no_pub_date.add(collection)
        self._pub_date_ready.add(collection)

    def _row_cols(self, collection: str, *extra: str) -> list[str]:
        """The no-vector read projection for *collection*, plus *extra*.

        ``_FILTER_COLUMNS``, minus ``pub_date`` on an old table that could
        not be given the column, so its reads keep working without it.
        """
        cols = [
            c
            for c in _FILTER_COLUMNS
            if not (c == "pub_date" and collection in self._no_pub_date)
        ]
        return [*cols, *extra]

    def _passwords_table(self):
        table = self._open_or_create(_SHOW_PASSWORDS_TABLE, _SHOW_PASSWORDS_SCHEMA)
        self._ensure_passwords_schema(table)
        return table

    def _ensure_passwords_schema(self, table) -> None:
        """Add ``show_id`` / ``show_label`` to pre-id password tables."""
        if self._passwords_schema_ready:
            return
        try:
            present = set(table.schema.names)
            missing = {
                col: "CAST('' AS STRING)"
                for col in ("show_id", "show_label")
                if col not in present
            }
            if missing:
                table.add_columns(missing)
        except Exception:
            logger.opt(exception=True).warning(
                "Could not add columns to _show_passwords"
            )
            return
        self._passwords_schema_ready = True

    # ── Show password management ─────────────────────────────────────────

    def get_show_password_entries(self) -> dict[str, dict]:
        """Return ``{key: {show_id, label, password_hash}}`` for protected shows.

        The key is the show id when the row has one, and the legacy display
        name when it does not. Rows predating show ids, and rows in an index
        rsynced mid-migration, therefore stay resolvable.
        """
        if _SHOW_PASSWORDS_TABLE not in self._table_names():
            return {}
        rows = self._passwords_table().search().limit(10_000).to_list()
        out: dict[str, dict] = {}
        for r in rows:
            sid = (r.get("show_id") or "").strip()
            legacy = (r.get("show") or "").strip()
            key = self.show_key(r)
            if not key:
                continue
            out[key] = {
                "show_id": sid,
                "label": (r.get("show_label") or "").strip() or legacy or sid,
                "password_hash": r.get("password_hash", ""),
            }
        return out

    def password_entry_for_show(
        self, show_id: str, show_label: str = ""
    ) -> tuple[str, dict] | None:
        """``(key, entry)`` of a show's password row, or None if it has none.

        The id-keyed row first; else a legacy row keyed by one of the show's
        display names, compared the way the migration and
        ``_delete_legacy_password_rows`` compare them (case and surrounding
        whitespace ignored). *key* is what ``delete_show_password`` takes.
        """
        entries = self.get_show_password_entries()
        if show_id and show_id in entries:
            return show_id, entries[show_id]
        wanted = _norm_label(show_label)
        if not wanted:
            return None
        for key, entry in entries.items():
            if not entry["show_id"] and _norm_label(key) == wanted:
                return key, entry
        return None

    def get_show_passwords(self) -> dict[str, str]:
        """Return ``{key: password_hash}``; see ``get_show_password_entries``."""
        return {
            k: v["password_hash"] for k, v in self.get_show_password_entries().items()
        }

    def set_show_password(
        self,
        show: str,
        password_hash: str,
        show_label: str = "",
        *,
        legacy_label: str = "",
    ) -> None:
        """Set or replace the password hash for a show.

        Args:
            show: The show's stable id (``show.toml`` id).
            password_hash: Hash in ``sha256:<hex>`` form.
            show_label: Display name, stored alongside so the bot can name
                the show without a show folder to read.
            legacy_label: A former display name whose name-keyed row this
                replaces, when it differs from ``show_label`` (a show renamed
                before the show-id migration).

        Raises:
            IndexOwnershipError: when this index is a replica of another
                machine's, where the next sync would erase the change.
        """
        _require_show_key(show)
        self._require_owner(f"set the password for show {show!r}")
        label = (show_label or "").strip()
        with self._write_lock:
            t = self._passwords_table()
            # One commit: a delete then an add left the show public if the
            # process died, or the add failed, in between.
            t.merge_insert(
                "show_id"
            ).when_matched_update_all().when_not_matched_insert_all().execute(
                [
                    {
                        "show_id": show,
                        "show_label": label,
                        # Mirror of the label, so an older bot build reading
                        # this index still sees the show as protected.
                        "show": label,
                        "password_hash": password_hash,
                    }
                ]
            )
            # Only after the id-keyed row is in: a legacy row for the same
            # show that the migration has not rekeyed yet would otherwise
            # unlock under its old name forever.
            self._delete_legacy_password_rows(
                t, legacy_label, *self._labels_owned_by(show, label)
            )
        logger.debug(f"Password set for show {show!r}")

    @staticmethod
    def _delete_legacy_password_rows(table, *labels: str) -> None:
        """Delete name-keyed password rows whose label matches any of *labels*.

        Legacy rows only: an id-keyed row also carries its label in ``show``
        (a mirror for older bot builds), and another show may share that
        label. Matched in Python with ``_norm_label`` (case and surrounding
        whitespace), then deleted by the exact stored value, because SQL
        ``trim`` and ``lower`` do not normalize the way Python does.
        """
        wanted = {_norm_label(x) for x in labels} - {""}
        if not wanted:
            return
        legacy = "(show_id IS NULL OR show_id = '')"
        rows = table.search().where(legacy).select(["show"]).limit(10_000).to_list()
        for raw in {r.get("show") or "" for r in rows}:
            if _norm_label(raw) in wanted:
                table.delete(f"show = '{_escape(raw)}' AND {legacy}")

    def delete_show_password(self, show: str, show_label: str = "") -> None:
        """Remove password protection for a show (makes it public).

        Args:
            show: The show's id, or the label a legacy row is keyed by.
            show_label: The show's display name, whose legacy (name-keyed)
                row is cleared too, unless another show carries that name.

        Raises:
            IndexOwnershipError: when this index is a replica of another
                machine's, where the next sync would erase the change.
        """
        _require_show_key(show)
        self._require_owner(f"remove the password for show {show!r}")
        with self._write_lock:
            if _SHOW_PASSWORDS_TABLE not in self._table_names():
                return
            t = self._passwords_table()
            # Matches both shapes: an id-keyed row, and a legacy row whose
            # only key is the display name.
            t.delete(f"show_id = '{_escape(show)}'")
            self._delete_legacy_password_rows(
                t, show, *self._labels_owned_by(show, show_label)
            )
        logger.debug(f"Password removed for show {show!r}")

    def _labels_owned_by(self, show_id: str, label: str) -> list[str]:
        """``[label]`` when no other show carries it, else ``[]``.

        A legacy password row is keyed by a display name only, so it cannot
        say whose it is. Clearing one under a name another show also uses
        would unprotect that show; leaving it is the safe miss (it still
        protects only what it always protected). Callers that know a row is
        theirs (a former name) pass it explicitly instead.
        """
        wanted = _norm_label(label)
        if not wanted:
            return []
        for meta in self.rows_for_label(label):
            # An id-less row counts too: it may be another, unmigrated show.
            # (The migration names its own legacy row via ``legacy_label``.)
            if (meta.get("show_id") or "").strip() != show_id:
                return []
        for key, entry in self.get_show_password_entries().items():
            if (
                entry["show_id"]
                and key != show_id
                and _norm_label(entry["label"]) == wanted
            ):
                return []
        return [label]

    # ── Collection management ────────────────────────────────────────────

    def collection_exists(self, name: str) -> bool:
        """Return True if a collection with the given name exists."""
        return name in self._table_names()

    @staticmethod
    def show_key(meta: dict) -> str:
        """Identity of the show a ``_collections`` row belongs to.

        The stable id when the row carries one, else the display label, which
        is all a row written before ids existed has. The single definition of
        the "id else label" rule, so the fallback that keeps a pre-migration
        index resolvable cannot drift between call sites.
        """
        return (meta.get("show_id") or "").strip() or (meta.get("show") or "")

    @staticmethod
    def row_is_show(
        meta: dict, show_id: str, show_label: str, *, strict: bool = False
    ) -> bool:
        """Whether a ``_collections`` row belongs to the given show.

        Matching by label is case-insensitive, matching every other
        label-to-show resolver in the codebase.

        An empty *show_id* means two things depending on the caller. The bot
        does not know ids (it reads an rsynced index with no show folders), so
        by default it matches any row by label. With *strict*, it means "this
        show has no id" (an app folder never minted one), which can own only
        rows that have none either: a row with an id belongs to the show that
        has it, whatever it is called.
        """
        row_id = (meta.get("show_id") or "").strip()
        label = (show_label or "").strip().lower()
        row_label = (meta.get("show") or "").strip().lower()
        if strict and not show_id and row_id:
            return False
        if show_id:
            # A legacy row has no id, so its label is the only way back to it
            # until the migration stamps one on.
            return row_id == show_id or (
                not row_id and bool(label) and row_label == label
            )
        return bool(label) and row_label == label

    def resolve_collection(
        self,
        show_id: str,
        model: str,
        chunker: str,
        show_label: str = "",
        *,
        strict: bool = False,
    ) -> str | None:
        """Return the collection name for a show, or None when not indexed.

        The single way to get from a show to its collection. Callers must not
        rebuild the name themselves: the name is an internal detail, and a
        caller that reconstructs it is exactly what a show rename used to
        orphan.

        Args:
            show_id: The show's stable id. Empty for the bot, which reads an
                rsynced index with no show folders to read ids from.
            model: Embedding model key.
            chunker: Chunking strategy key.
            show_label: Display name, used to find collections written before
                ids existed, and as the bot's only handle on a show.
            strict: See ``row_is_show``; the app passes it for a show folder.
        """
        for name, meta in self.get_all_collection_info().items():
            if meta.get("model") != model or meta.get("chunker") != chunker:
                continue
            if self.row_is_show(meta, show_id, show_label, strict=strict):
                return name
        return None

    def collections_for_show(
        self,
        show_id: str,
        show_label: str = "",
        *,
        strict: bool = False,
        info: dict[str, dict] | None = None,
    ) -> list[str]:
        """Every collection belonging to a show, across models and chunkers.

        Args:
            strict: See ``row_is_show``; the app passes it for a show folder.
            info: ``get_all_collection_info()`` when the caller already holds
                it, to spare another index scan.
        """
        info = self.get_all_collection_info() if info is None else info
        return sorted(
            name
            for name, meta in info.items()
            if self.row_is_show(meta, show_id, show_label, strict=strict)
        )

    def show_id_for_label(self, label: str) -> str:
        """Stable id of the show with this display name, else the label itself.

        Index-backed, so it works in the bot, which has no show folders. The
        app has a folder-backed equivalent in ``ingest.show_registry`` that
        also covers shows with no collection yet.
        """
        if not (label or "").strip():
            return ""
        ids = sorted(
            {(m.get("show_id") or "").strip() for m in self.rows_for_label(label)}
            - {""}
        )
        # A real id first: an id-less row's identity is its label, which must
        # not be handed back while an id exists for the name.
        return ids[0] if ids else label

    def rows_for_label(self, label: str) -> list[dict]:
        """``_collections`` rows carrying this display name (case-insensitive).

        Labels are not unique (a registered show may keep a colliding name,
        and an unregistered show's label stays in the index), so a caller
        deciding access looks at all of them, not the first one found. The
        one definition of matching a typed name against stored labels.
        """
        target = (label or "").strip().lower()
        if not target:
            return []
        return [
            meta
            for meta in self.get_all_collection_info().values()
            if (meta.get("show") or "").strip().lower() == target
        ]

    def label_for_show_id(self, show_id: str) -> str:
        """Display name registered for a show id, else the id itself."""
        if not show_id:
            return ""
        for meta in self.get_all_collection_info().values():
            if (meta.get("show_id") or "").strip() == show_id:
                return (meta.get("show") or "").strip() or show_id
        return show_id

    def collection_label(self, name: str) -> str:
        """Display label registered for a collection ("" when unknown).

        Read from ``_collections`` rather than from chunk rows, so a rename
        needs one metadata write instead of touching every chunk.
        """
        return (
            (self.get_all_collection_info().get(name) or {}).get("show") or ""
        ).strip()

    def set_collection_identity(self, name: str, show_id: str, show: str) -> None:
        """Write the id and display label onto a collection's metadata row."""
        self._require_owner(f"set the identity of collection {name!r}")
        with self._write_lock:
            self._collections_table().update(
                where=f"name = '{_escape(name)}'",
                values={"show_id": show_id, "show": show},
            )
            self._collection_info_cache = None

    def ensure_collection_for_show(
        self,
        show_id: str,
        show: str,
        model: str,
        chunker: str,
        dim: int,
    ) -> str:
        """Return the collection for a show, creating it if it does not exist.

        The only way to create a collection. Resolution runs first, so a show
        indexed before ids existed keeps writing into the table it already
        has (under its legacy name) instead of growing a second one beside it.

        Returns:
            The collection name.
        """
        from podcodex.rag.store import collection_name

        existing = self.resolve_collection(show_id, model, chunker, show_label=show)
        name = existing or collection_name(show_id or show, model, chunker)
        self.ensure_collection(
            name=name,
            show=show,
            model=model,
            chunker=chunker,
            dim=dim,
            show_id=show_id,
        )
        return name

    def ensure_collection(
        self,
        name: str,
        show: str,
        model: str,
        chunker: str,
        dim: int,
        artwork_url: str = "",
        show_id: str = "",
    ) -> None:
        """Create the collection table + metadata row if missing (idempotent).

        Args:
            name: Canonical collection name (see ``store.collection_name``).
            show: Human-readable show name.
            model: Embedding model key.
            chunker: Chunking strategy key.
            dim: Embedding vector dimensionality.
            artwork_url: Channel-level show artwork URL ("" = unknown; a
                later call with a real URL fills it in on the existing row).
            show_id: Stable show identity. Empty only for callers that
                predate ids (legacy bundle imports); the migration fills
                those in later.
        """
        # Both writes tolerate another process doing the same first index at
        # the same time (two compute tasks run at once, each from its own
        # subprocess store, which _write_lock does not reach): the table goes
        # through _open_or_create, and the row is inserted only if absent.
        # Lance does not serialize two concurrent inserts, so a duplicate row
        # is still possible in that instant; every reader keys rows by name
        # (get_all_collection_info, list_collections), which makes it inert.
        self._open_or_create(name, _chunk_schema(dim))

        def _write_row() -> None:
            meta = self._collections_table()
            where = f"name = '{_escape(name)}'"
            existing = meta.search().where(where).limit(1).to_list()
            if not existing:
                meta.merge_insert("name").when_not_matched_insert_all().execute(
                    [
                        {
                            "name": name,
                            "show_id": show_id,
                            "show": show,
                            "model": model,
                            "chunker": chunker,
                            "dim": dim,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "artwork_url": artwork_url,
                        }
                    ]
                )
                self._collection_info_cache = None
            else:
                updates: dict[str, str] = {}
                if artwork_url and not (existing[0].get("artwork_url") or "").strip():
                    updates["artwork_url"] = artwork_url
                if show_id and not (existing[0].get("show_id") or "").strip():
                    updates["show_id"] = show_id
                if updates:
                    meta.update(where=f"name = '{_escape(name)}'", values=updates)
                    self._collection_info_cache = None

        _retry_concurrent(_write_row)

    def list_collections(
        self, show: str = "", model: str = "", chunker: str = ""
    ) -> list[str]:
        """List collection names, optionally filtered by show/model/chunker.

        Args:
            show: Exact display-label filter (empty = no filter). This is a
                filter, not a lookup: use ``collections_for_show`` to go from
                a show to its collections.
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
        # A set: two processes creating one collection at the same instant
        # can both insert its row (see ensure_collection).
        return sorted({r["name"] for r in q.limit(10_000).to_list()})

    def delete_collection(self, name: str) -> None:
        """Drop a collection table and its metadata row.

        Args:
            name: Collection name to delete.
        """
        self._require_owner(f"delete collection {name!r}")
        with self._write_lock:
            # Table first. If the drop fails nothing has changed; if the row
            # delete then fails, a listed collection with no table reads as
            # empty and the next ensure_collection recreates the table. The
            # other order could leave a hidden table that a later index
            # reopens, old chunks and all.
            if name in self._table_names():
                self._db.drop_table(name)
            self._collections_table().delete(f"name = '{_escape(name)}'")
            # The one-shot upgrade markers belong to the table: a collection
            # recreated under this name must run them again.
            for suffix in _SENTINEL_SUFFIXES:
                (self._path / f"{name}.{suffix}").unlink(missing_ok=True)
            self._fts_ready.discard(name)
            self._fts_failed.pop(name, None)
            self._pub_date_ready.discard(name)
            self._no_pub_date.discard(name)
        logger.debug(f"Deleted collection '{name}'")

    def get_collection_info(self, name: str) -> dict | None:
        """Return ``{show, model, chunker, dim}`` or ``None`` if unknown."""
        return self.get_all_collection_info().get(name)

    def get_all_collection_info(self) -> dict[str, dict]:
        """Return ``{collection_name: {show_id, show, model, chunker, dim, ...}}``.

        Cached against ``index_mtime`` so the table scan runs once per
        on-disk change rather than once per caller (hot path: MCP tools,
        bot-access, the ``search_vector`` dim guard).
        """
        snap = getattr(self._snap, "state", None)
        if snap is not None and snap["info"] is not None:
            return snap["info"]
        info = self._read_all_collection_info()
        if snap is not None:
            snap["info"] = info
        return info

    def _read_all_collection_info(self) -> dict[str, dict]:
        mtime = self.index_mtime()
        cached = self._collection_info_cache
        if cached is not None and cached[0] == mtime:
            return cached[1]
        rows = self._collections_table().search().limit(10_000).to_list()
        info = {
            r["name"]: {
                "show_id": (r.get("show_id") or "").strip(),
                "show": r["show"],
                "model": r["model"],
                "chunker": r["chunker"],
                "dim": int(r["dim"]),
                "artwork_url": (r.get("artwork_url") or "").strip(),
                # Identity of this incarnation: a collection dropped and
                # recreated under the same name gets a new one.
                "created_at": r.get("created_at") or "",
            }
            for r in rows
            if r.get("name")
        }
        self._heal_collection_meta(info)
        self._collection_info_cache = (mtime, info)
        return info

    def _heal_collection_meta(self, info: dict[str, dict]) -> None:
        """Reconcile each collection row against its show's ``show.toml``.

        Heals two fields the row carries only so the bot can read them
        without show folders: ``artwork_url`` and the display label.

        Pull, not push. A rename used to push the new label into the index at
        write time, which went stale forever if that one write failed, and
        did nothing at all for a show renamed while the index was offline,
        hand-edited, or imported. Reading reconciles instead, so the index
        converges on whatever ``show.toml`` currently says.

        Only the app runs this: the show-folder resolver is registered by the
        API and not by the bot, which reads an rsynced index with no show
        folders and serves whatever is stored.
        """
        resolver = type(self)._show_folder_resolver
        if resolver is None:
            return
        from podcodex.ingest.show import load_show_meta, show_display

        for name, entry in info.items():
            folder = self._folder_for_row(entry)
            if folder is None:
                continue

            updates: dict[str, str] = {}
            label = show_display(folder)
            if label and label != entry.get("show"):
                updates["show"] = label
            meta = load_show_meta(folder)
            if not (entry.get("show_id") or "").strip() and meta and meta.id:
                updates["show_id"] = meta.id
            if not entry.get("artwork_url"):
                url = (meta.artwork_url if meta else "").strip()
                # Only real URLs may leave show.toml: the field can hold the
                # local-file marker, which search consumers would otherwise
                # render as a literal artwork URL.
                if is_remote_artwork_url(url):
                    updates["artwork_url"] = url
            if not updates:
                continue
            try:
                self._collections_table().update(
                    where=f"name = '{_escape(name)}'", values=updates
                )
            except Exception:
                # Includes the ownership refusal on a replica. Healing is a
                # convenience on a read path; never fail the read for it.
                logger.opt(exception=True).debug(
                    f"collection meta heal skipped for '{name}'"
                )
                continue
            entry.update(updates)

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

    def episode_collection_summary(self, collection: str, episode: str) -> dict | None:
        """``(chunk_count, source)`` for an episode in a collection, or ``None``."""
        if not self.collection_exists(collection):
            return None
        t = self._table(collection)
        where = f"episode = '{_escape(episode)}'"
        count = int(t.count_rows(filter=where))
        if not count:
            return None
        rows = t.search().where(where).select(["source"]).limit(1).to_list()
        return {
            "chunk_count": count,
            "source": str(rows[0].get("source", "")) if rows else "",
        }

    def delete_episode(self, collection: str, episode: str) -> None:
        """Delete every chunk for an episode in a collection.

        Args:
            collection: Collection name.
            episode: Episode identifier.
        """
        with self._write_lock:
            if not self.collection_exists(collection):
                return
            t = self._table(collection)
            t.delete(f"episode = '{_escape(episode)}'")
            self._fts_ready.discard(collection)
            self._fts_failed.pop(collection, None)
            self._uncompacted.add(collection)
        logger.debug(f"Deleted episode '{episode}' from '{collection}'")

    def delete_episode_everywhere(
        self, show: str, episode: str, show_id: str = ""
    ) -> list[str]:
        """Delete an episode's chunks from every collection of one show.

        A show routinely has several collections (one per model/chunker pair),
        so removing an episode from a single named one, which is all
        ``DELETE /api/index/episode`` can do by contract, leaves it answering
        searches. Whole-episode deletion needs the fan-out, and it belongs
        here so the store's invariants (existence short-circuit, ``_fts_ready``
        discard, escaping) stay in one place.

        Raises on the first failure rather than collecting per-collection
        errors: callers treat a partial index delete as a total failure, so
        there is nothing useful to do with the rest.

        Args:
            show: Display name of the show.
            episode: Episode identifier (stem).
            show_id: The show's stable id, when the caller has one. Preferred
                over the name, which a rename can change out from under it.

        Returns:
            The collections the episode was actually removed from.
        """
        touched = []
        for col in self.collections_for_show(show_id, show_label=show):
            if not self.episode_is_indexed(col, episode):
                continue
            self.delete_episode(col, episode)
            touched.append(col)
        # A delete is the one write that frees space, and it was the only
        # one no caller compacted after: index jobs compact at the end of a
        # run, but deleting episodes without re-indexing left every removed
        # chunk's data files on disk (and in the rsync to the bot host).
        if touched:
            self.compact(touched)
        return touched

    def compact(self, collections: list[str] | None = None) -> list[str]:
        """Merge fragments and drop superseded versions of chunk tables.

        Lance is copy-on-write: every ``add``, ``delete`` and ``update``
        commits a new table version and leaves the previous data files in
        place. Nothing reclaims them on its own, so without this the index
        directory grows monotonically with every re-index (a delete plus an
        add per episode) and every metadata heal, and the whole history is
        what an rsync to the bot host ships.

        Called at the end of an index job, a batch and ``podcodex-reindex``,
        and after an episode delete. Never on the read path: it rewrites
        data files, so it belongs where a write already happened. Versions
        replaced more than ``_COMPACT_KEEP_HISTORY`` ago are dropped (see
        ``_history_window``), which still covers a reader in another process
        mid-query. Data files no version references yet (a write in flight)
        are never touched: Lance only deletes those after a week.

        Args:
            collections: Tables to compact. Defaults to the chunk tables this
                store has written to since the last call.

        Returns:
            The collections that were compacted.
        """
        names = list(self._uncompacted if collections is None else collections)
        done: list[str] = []
        for name in names:
            self._uncompacted.discard(name)
            if not self.collection_exists(name):
                continue
            try:
                with self._write_lock:
                    table = self._table(name)
                    # Compact first, prune second: pruned in the same call, the
                    # window would be set before the compaction commits, and
                    # the version that compaction replaces must count as just
                    # replaced. The second call has nothing left to merge.
                    table.optimize(cleanup_older_than=_NO_PRUNING)
                    table.optimize(
                        cleanup_older_than=_history_window(
                            table.list_versions(), _COMPACT_KEEP_HISTORY
                        )
                    )
                done.append(name)
            except Exception:
                logger.opt(exception=True).warning(f"Compaction skipped for '{name}'")
        if done:
            logger.info(f"Compacted {len(done)} collection(s): {', '.join(done)}")
        return done

    def list_episodes(self, collection: str) -> list[str]:
        """Return a sorted list of distinct episodes in the collection.

        Args:
            collection: Collection name.
        """
        return self._distinct(collection, "episode")

    @staticmethod
    def _source_record(row: dict) -> dict | None:
        try:
            meta = json.loads(row.get("meta") or "{}")
        except (TypeError, ValueError):
            return None
        if not isinstance(meta, dict) or not meta.get("source_hash"):
            return None
        return {
            "hash": str(meta["source_hash"]),
            "ref": str(meta.get("source_ref") or "auto"),
            "step": str(row.get("source") or ""),
            "version": str(meta.get("source_version") or ""),
        }

    def episode_source_record(self, collection: str, episode: str) -> dict | None:
        """``{hash, ref, step, version}`` recorded for one episode's chunks.

        ``hash`` is the content hash of the source version they were cut from,
        ``version`` its id, ``ref`` how it was chosen (auto / step / pinned),
        ``step`` the step it belongs to. None for chunks indexed before this
        was recorded.
        """
        if not self.collection_exists(collection):
            return None
        rows = (
            self._table(collection)
            .search()
            .where(f"episode = '{_escape(episode)}' AND chunk_index = 0")
            .select(["source", "meta"])
            .limit(1)
            .to_list()
        )
        return self._source_record(rows[0]) if rows else None

    def episode_source_records(self, collection: str) -> dict[str, dict]:
        """:meth:`episode_source_record` for every episode that has one.

        Read from each episode's first chunk only (every chunk of a run carries
        the same value), so this scans one small row per episode.
        """
        if not self.collection_exists(collection):
            return {}
        rows = (
            self._table(collection)
            .search()
            .where("chunk_index = 0")
            .select(["episode", "source", "meta"])
            .limit(1_000_000)
            .to_list()
        )
        out: dict[str, dict] = {}
        for row in rows:
            record = self._source_record(row)
            if record:
                out[str(row["episode"])] = record
        return out

    def collection_version(self, collection: str) -> int:
        """Dataset version of a collection's table, or 0 if it doesn't exist.

        Bumps on every write, and the table handle is opened fresh per call,
        so this is a cross-process-safe cache key for anything derived from a
        collection's contents. Costs a metadata read, not a scan.

        Args:
            collection: Collection name.
        """
        if not self.collection_exists(collection):
            return 0
        return int(self._table(collection).version)

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
        """Upsert chunks and their embeddings for an episode (idempotent).

        A new episode is appended. An indexed one is replaced in a single
        ``merge_insert`` keyed on ``(episode, chunk_index)`` that also deletes
        rows past the new chunk count, so re-indexing with a different
        transcript version leaves no stale chunks and no window where the
        episode has none.

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
                    "pub_date": normalize_pub_date(chunk.get("pub_date")) or "",
                    "dominant_speaker": str(speaker),
                    "start": float(chunk.get("start", 0.0)),
                    "end": float(chunk.get("end", 0.0)),
                    "text": str(chunk.get("text", "")),
                    "vector": embeddings[i].astype(np.float32).tolist(),
                    "meta": json.dumps(meta),
                }
            )
        with self._write_lock:
            # Opened under the lock, so a compaction or replace that finished
            # while this call waited is part of the view it writes on.
            t = self._table(collection)
            where = f"episode = '{_escape(episode)}'"
            if self.episode_is_indexed(collection, episode):
                # The by-source delete joins the whole table (tens of ms on a
                # large one), so only a replace pays for it.
                (
                    t.merge_insert(["episode", "chunk_index"])
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .when_not_matched_by_source_delete(where)
                    .execute(rows)
                )
                # merge_insert updates duplicate (episode, chunk_index) rows
                # rather than removing them, and the unlocked delete-then-add
                # this replaced could leave some. Rewrite such an episode.
                if t.count_rows(where) != len(rows):
                    t.delete(where)
                    t.add(rows)
            else:
                t.add(rows)
        self._fts_ready.discard(collection)
        self._fts_failed.pop(collection, None)
        self._uncompacted.add(collection)
        logger.debug(f"Saved {len(rows)} chunks for '{episode}' in '{collection}'")

    # ── Read helpers ─────────────────────────────────────────────────────

    def load_chunks_no_embeddings(self, collection: str, episode: str) -> list[Hit]:
        """Return chunks for an episode without their vectors.

        Used by the indexer for stale-source detection.

        Args:
            collection: Collection name.
            episode: Episode identifier.

        Returns:
            Hits sorted by ``chunk_index``.
        """
        if not self.collection_exists(collection):
            return []
        t = self._table(collection)
        rows = (
            t.search()
            .where(f"episode = '{_escape(episode)}'")
            .select(self._row_cols(collection, "meta"))
            .limit(100_000)
            .to_list()
        )
        label = self.collection_label(collection)
        out = [_row_to_chunk(r, label) for r in rows]
        out.sort(key=lambda c: c.chunk_index)
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
            .select(self._row_cols(collection, "vector", "meta"))
            .limit(100_000)
            .to_list()
        )
        label = self.collection_label(collection)
        out: list[dict] = []
        for r in rows:
            v = np.asarray(r.get("vector") or [], dtype=np.float32)
            # Inspector-only projection: stays a plain dict so the vector_*
            # diagnostics keys don't pollute the Hit model.
            chunk = _row_to_chunk(r, label).model_dump(exclude_none=True)
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
        chunk_index: int | None = None,
        window: int = 3,
        at_time: float | None = None,
    ) -> list[Hit]:
        """Return chunks around a center chunk, ordered by position.

        Used to expand a retrieved hit with its neighbors so callers
        (MCP ``get_context``) get enough surrounding dialogue for
        grounded answers. The center is named either by ``chunk_index`` or
        by ``at_time`` (seconds); both resolve against a single episode load.

        Args:
            collection: Collection name.
            episode: Episode identifier.
            chunk_index: The center chunk's position in the episode.
            window: Number of chunks to include on each side. Clamped to
                ``[0, ...]``. ``0`` returns only the center chunk.
            at_time: Alternative to ``chunk_index``: center on the chunk
                covering this timestamp (nearest by start when in a gap).

        Returns:
            Slice of the episode's chunks covering
            ``[center - window, center + window]``, inclusive, sorted by
            ``chunk_index``. Empty if the episode is empty, neither locator
            is given, or a ``chunk_index`` is not found.
        """
        window = max(0, window)
        chunks = self.load_chunks_no_embeddings(collection, episode)
        if not chunks:
            return []
        if chunk_index is not None:
            center = next(
                (i for i, c in enumerate(chunks) if c.chunk_index == chunk_index),
                -1,
            )
        elif at_time is not None:
            center = _center_pos_by_time(chunks, at_time)
        else:
            return []
        if center < 0:
            return []
        lo = max(0, center - window)
        hi = min(len(chunks), center + window + 1)
        return chunks[lo:hi]

    def count_chunks(
        self,
        collection: str,
        *,
        episode: str | None = None,
        episodes: list[str] | None = None,
        source: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> int:
        """Count chunks matching the filters (no rows marshalled).

        Used by ``Retriever.random`` with :meth:`chunk_at`, which keep
        ``speaker`` to chunks the speaker dominates: a random quote from a
        chunk they only interjected in would be their one-word turn.
        """
        if not self.collection_exists(collection):
            return 0
        t = self._table(collection)
        clause = build_where(
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
            speaker_turns=False,
        )
        return t.count_rows(clause) if clause else t.count_rows()

    def chunk_at(
        self,
        collection: str,
        offset: int,
        *,
        episode: str | None = None,
        episodes: list[str] | None = None,
        source: str | None = None,
        speaker: str | None = None,
        pub_date_min: str | None = None,
        pub_date_max: str | None = None,
    ) -> Hit | None:
        """Return the single chunk at ``offset`` within the filtered scan.

        Pairs with :meth:`count_chunks` so ``Retriever.random`` can pick one
        chunk without loading (and validating) the whole collection. Returns
        None when the offset is past the end, which the caller treats as a
        stale count and retries.
        """
        if not self.collection_exists(collection):
            return None
        t = self._table(collection)
        q = t.search()
        clause = build_where(
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
            speaker_turns=False,  # see count_chunks
        )
        if clause:
            q = q.where(clause)
        rows = (
            q.select(self._row_cols(collection, "meta"))
            .offset(offset)
            .limit(1)
            .to_list()
        )
        return (
            _row_to_chunk(rows[0], self.collection_label(collection)) if rows else None
        )

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
    ) -> list[Hit]:
        """ANN vector search with optional pre-filters.

        Args:
            collection: Collection name.
            query_vec: Query embedding (float32 1-D).
            top_k: Maximum results.
            episode, source, speaker: SQL-style equality filters.
            episodes: Alternative to ``episode`` — restrict to a list.
            pub_date_min, pub_date_max: Inclusive date bounds.

        Returns:
            Hits with cosine ``score`` (``1 - distance``) attached.
        """
        if not self.collection_exists(collection):
            return []
        # Guard against an embedder/collection mismatch: a wrong-dim vector
        # raises a raw LanceDB error (swallowed upstream into empty results),
        # and a same-dim wrong-model vector silently ranks nonsense. Fail loud.
        info = self.get_collection_info(collection)
        if info is not None and query_vec.shape[-1] != info["dim"]:
            raise ValueError(
                f"query vector dim {query_vec.shape[-1]} does not match "
                f"collection '{collection}' dim {info['dim']} "
                f"(model {info.get('model')!r}): embedder/collection mismatch"
            )
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
        clause = build_where(
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
        if clause:
            q = q.where(clause)
        # Project away the vector: no caller reads it back, and marshalling
        # dim floats per row dominates the read.
        rows = (
            q.select(self._row_cols(collection, "meta", "_distance"))
            .limit(top_k)
            .to_list()
        )
        label = self.collection_label(collection)
        out: list[Hit] = []
        for r in rows:
            hit = _row_to_chunk(r, label)
            # LanceDB cosine distance is 1 - cosine_sim, so 1 - distance is
            # the cosine similarity in [0, 1] (assuming non-negative dot).
            hit.score = max(0.0, 1.0 - float(r.get("_distance", 1.0)))
            out.append(hit)
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
    ) -> list[Hit]:
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
            Hits with BM25 ``score`` attached.
        """
        rows = self._fts_rows(
            collection,
            query,
            fuzziness,
            self._row_cols(collection, "meta"),
            top_k,
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )
        label = self.collection_label(collection)
        out: list[Hit] = []
        for r in rows:
            hit = _row_to_chunk(r, label)
            hit.score = float(r.get("_score", 0.0))
            out.append(hit)
        return out

    def _fts_rows(
        self,
        collection: str,
        query: str,
        fuzziness: int,
        columns: list[str],
        limit: int,
        *,
        table=None,
        **filters,
    ) -> list[dict]:
        """Rows of one filtered FTS probe, the read behind every FTS search.

        ``[]`` when the collection does not exist. *table* is an already open
        handle: a caller running many probes (``search_literal``) opens the
        table once, which saves a table listing and open per probe and keeps
        every probe on the same version of the index.

        A probe that fails raises: an empty list would tell exact search and
        ``exact_counts`` that the phrase is absent when the truth is unknown.
        The one failure handled here is a missing inverted index on a table
        this store believed indexed, which is what another process dropping
        and recreating the table (``podcodex-reindex``) leaves behind: the
        readiness mark is dropped and the probe retried once after a rebuild.
        Matched by message (Lance raises it untyped);
        ``test_fts_rebuilds_after_another_process_recreates_the_table``
        produces the real error, so a reworded one fails there.
        """
        if table is None:
            if not self.collection_exists(collection):
                return []
            table = self._table(collection)
        # _score named explicitly: Lance only adds it to a projection by
        # default for now, and warns on every probe that this will stop.
        # pub_date re-checked here: the caller built *columns* before the
        # table was opened, and opening is what finds it missing.
        columns = [
            c
            for c in [*columns, "_score"]
            if not (c == "pub_date" and collection in self._no_pub_date)
        ]
        try:
            return (
                self._fts_probe(table, collection, query, fuzziness, filters)
                .select(columns)
                .limit(limit)
                .to_list()
            )
        except Exception as exc:
            if "INVERTED index" not in str(exc) or collection not in self._fts_ready:
                raise
        logger.info(f"FTS index missing on '{collection}': rebuilding once")
        self._fts_ready.discard(collection)
        return (
            self._fts_probe(table, collection, query, fuzziness, filters)
            .select(columns)
            .limit(limit)
            .to_list()
        )

    def _fts_probe(self, t, collection: str, query: str, fuzziness: int, filters):
        """The filtered FTS query of ``_fts_rows``, before projection and limit."""
        self._ensure_fts(collection, t)
        if fuzziness > 0:
            from lancedb.query import MatchQuery

            search_input = MatchQuery(
                query,
                "text",
                fuzziness=fuzziness,
                max_expansions=_FTS_MAX_EXPANSIONS,
            )
        else:
            search_input = query
        q = t.search(search_input, query_type="fts")
        clause = build_where(**filters)
        return q.where(clause) if clause else q

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
    ) -> tuple[list[Hit], list[Hit], list[Hit]]:
        """Three-tier phrase search: exact, accent variant, near-typo.

        FTS ``~{max_dist}`` pre-filters candidates; Python re-verifies with
        exact Levenshtein ≤ ``max_dist`` after accent folding.

        - ``exact``:       ``query.lower()`` is a substring of ``text.lower()``.
                           Whole-word occurrences (non-letter or text edge on
                           both sides) score 1.0 and rank first; matches
                           inside a longer word ("william" in "Williams")
                           stay in the tier at 0.99 and rank after them,
                           chronologically within each group.
        - ``accent_only``: accent-folded query is a substring of folded text
        - ``fuzzy_only``:  single-word query (3+ chars) with a word within
                           ``max_dist`` edits (after accent folding) in the
                           chunk text; a multi-word query matches the whole
                           phrase approximately instead

        No candidate cap: every FTS probe is limited to the collection's row
        count, so a query made of common words still finds every match.
        """
        if not self.collection_exists(collection):
            return [], [], []

        folded_q = fold_text(query)  # hyphens → spaces + accent-fold
        query_lower = query.lower()

        # Tokenize the same way Tantivy does: split on non-word characters.
        # Tokens shorter than _MIN_PROBE_LEN (e.g. "c" from "c'est") are too
        # common to discriminate chunks, so they only probe when the query
        # has nothing longer ("AI", "TV"), and then as exact tokens: a 2-edit
        # fuzzy probe on a two-letter word matches nearly every chunk.
        all_tokens = [t for t in _WORD_RE.split(query) if t]
        if not all_tokens:
            return [], [], []
        long_tokens = [t for t in all_tokens if len(t) >= _MIN_PROBE_LEN]
        probe_tokens = list(dict.fromkeys(long_tokens or all_tokens))
        # Decided on every word, not only the probed ones: "le chat" is a
        # phrase, not the single word "chat".
        fuzzy_word = all_tokens[0] if len(all_tokens) == 1 and long_tokens else None
        limit = max(self.count_rows(collection), 1)

        # FTS pre-filter via MatchQuery, per token, allowing exactly the edits
        # the Python check below accepts: max_dist for a single word, the
        # phrase tolerance for a phrase (every token of a phrase within k
        # edits is itself within k). More only floods the candidate pool
        # with chunks the check then rejects, at a build cost each. Tantivy
        # keeps accents in the index and its fuzzy query has a non-zero
        # effective prefix_length, so an accent-swap on the leading char
        # (e.g. "etres" → "êtres") won't match on edits alone: we probe both
        # the original token and its accent-folded form and union the hits.
        # Tolerance ~12% of phrase length: short queries stay strict, long
        # ones accept one or two real typos.
        phrase_max = max(1, len(folded_q) // 8)
        token_edits = min(max_dist if fuzzy_word is not None else phrase_max, 2)
        filters = dict(
            episode=episode,
            episodes=episodes,
            source=source,
            speaker=speaker,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
        )

        def _variants(token: str) -> list[tuple[str, int]]:
            # Always an exact probe too: a fuzzy MatchQuery expands to at most
            # _FTS_MAX_EXPANSIONS terms, and nothing guarantees the query term
            # itself is among them ("the" has thousands of close neighbours).
            ft = fold_text(token)
            forms = [token, ft] if ft and ft != token else [token]
            edits = [0, token_edits] if len(token) >= _MIN_PROBE_LEN else [0]
            return [(form, fuzziness) for form in forms for fuzziness in edits]

        label = self.collection_label(collection)
        # One handle for every probe below; see _fts_rows.
        table = self._table(collection)

        def _probe(form: str, fuzziness: int, columns: list[str]) -> list[dict]:
            # An exact-token probe that fails raises: without it the answer is
            # unknown. A fuzzy one only widens the near-typo tier, so its
            # failure (a Lance error on an odd expansion) costs that tier for
            # this form, logged, not the whole search.
            try:
                return self._fts_rows(
                    collection,
                    form,
                    fuzziness,
                    columns,
                    limit,
                    table=table,
                    **filters,
                )
            except Exception:
                if not fuzziness:
                    raise
                logger.opt(exception=True).warning(
                    f"Fuzzy FTS probe failed for {form!r} on '{collection}': "
                    "near-typo matches may be missing"
                )
                return []

        def _hydrate(token: str, keep: set[str] | None = None) -> list[Hit]:
            # Rows are deduped before they are built: the exact and fuzzy
            # probes overlap almost entirely, and building a Hit (json.loads
            # plus validation) is the cost here.
            hits: list[Hit] = []
            seen: set[str] = set()
            for form, fuzziness in _variants(token):
                rows = _probe(form, fuzziness, self._row_cols(collection, "meta"))
                for r in rows:
                    k = chunk_row_key(r["episode"], r["chunk_index"])
                    if k not in seen and (keep is None or k in keep):
                        seen.add(k)
                        hits.append(_row_to_chunk(r, label))
            return hits

        if len(probe_tokens) == 1:
            candidates = _hydrate(probe_tokens[0])
        else:
            # A chunk survives only if it matched *every* token: intersect
            # key-only probes, then hydrate the survivors from the rarest
            # token's probe, the smallest one that holds them all. (Loading
            # them by key instead builds one filter clause per episode, which
            # Lance caps at 500 conditions and can crash on well before that.)
            keys: set[str] | None = None
            sizes: dict[str, int] = {}
            for token in probe_tokens:
                token_keys: set[str] = set()
                for form, fuzziness in _variants(token):
                    rows = _probe(form, fuzziness, ["episode", "chunk_index"])
                    token_keys |= {
                        chunk_row_key(r["episode"], r["chunk_index"]) for r in rows
                    }
                sizes[token] = len(token_keys)
                keys = token_keys if keys is None else keys & token_keys
                if not keys:
                    break
            candidates = _hydrate(min(sizes, key=sizes.get), keys) if keys else []

        word_cache: dict[str, float | None] = {}
        exact: list[Hit] = []
        accent_only: list[Hit] = []
        fuzzy_only: list[Hit] = []
        for c in candidates:
            text = c.text
            lower = text.lower()
            if len(lower) != len(text):
                # A few chars ("İ") lowercase to two, which would shift every
                # offset below. Leave those as they are.
                lower = "".join(
                    ch if len(ch.lower()) != 1 else ch.lower() for ch in text
                )
            if query_lower in lower:
                # Prefer a whole-word occurrence over the first occurrence:
                # a chunk with both "Williams" and "William" is a word match.
                idx = lower.find(query_lower)
                word = False
                i = idx
                while i != -1:
                    before_ok = i == 0 or not lower[i - 1].isalnum()
                    end = i + len(query_lower)
                    after_ok = end >= len(lower) or not lower[end].isalnum()
                    if before_ok and after_ok:
                        idx, word = i, True
                        break
                    i = lower.find(query_lower, i + 1)
                # Candidates are freshly built for this query (search_fts →
                # _row_to_chunk) and land in exactly one tier: mutate in place.
                c.score = 1.0 if word else _EXACT_SUBSTRING_SCORE
                c.match_text = text[idx : idx + len(query)]
                exact.append(c)
            else:
                folded_t = fold_text(text)
                if folded_q in folded_t:
                    span = _accent_span(text, folded_t, folded_q) or query
                    c.score = _accent_score(query, span)
                    c.match_text = span
                    accent_only.append(c)
                else:
                    if fuzzy_word is not None:
                        result = _fuzzy_match(fuzzy_word, text, max_dist, word_cache)
                        if result is not None:
                            c.score, c.match_text = result
                            fuzzy_only.append(c)
                    else:
                        hit = approx_substring(folded_q, folded_t, phrase_max)
                        if hit is not None:
                            d, fs, fe = hit
                            span = _original_slice(text, fs, fe) or folded_t[fs:fe]
                            c.score = 1.0 - d / max(len(folded_q), 1)
                            c.match_text = span
                            fuzzy_only.append(c)

        # Word matches (1.0) before superstring matches (0.99), each group
        # chronological.
        exact.sort(key=lambda c: (-c.score, c.episode, c.start))
        accent_only.sort(key=lambda c: (c.episode, c.start))
        fuzzy_only.sort(key=lambda c: c.score, reverse=True)

        return exact, accent_only, fuzzy_only

    def _ensure_fts(self, collection: str, table) -> None:
        """Ensure an ASCII-folding FTS index exists on the ``text`` column.

        ``ascii_folding=True`` + ``lower_case=True`` normalize both indexed
        text and the query, so accented terms in the corpus match unaccented
        queries (e.g. "etres" → "êtres"). Without folding, tantivy's fuzzy
        query has an effective leading-char prefix constraint that prevents
        1-edit matches when the first char is the accented one.

        A sentinel file (``.fts_v2``) records the one-time upgrade so we
        rebuild only when a pre-existing stale index is found, not on every
        process restart. v2 exists because indexes written by older lancedb
        versions silently no-op fuzzy ``MatchQuery`` (exact-token hits only,
        zero for unknown tokens), starving ``search_literal``'s candidate
        pool; a rebuild under the current version restores fuzzy matching.
        The earlier ``.fts_folded_v1`` (accent-folding upgrade) is superseded
        and removed on migration.
        """
        if collection in self._fts_ready:
            return
        failed_at = self._fts_failed.get(collection)
        if failed_at is not None and time.monotonic() - failed_at < _FTS_RETRY_S:
            return
        sentinel = self._path / f"{collection}.fts_v2"
        legacy_sentinel = self._path / f"{collection}.fts_folded_v1"
        try:
            # Decided under the lock: two first queries racing would
            # otherwise both build, and the second fail on the first's index.
            with self._write_lock:
                if collection in self._fts_ready:
                    return
                try:
                    existing = {idx.name for idx in table.list_indices()}
                except Exception:
                    existing = set()
                has_text_idx = any("text" in n.lower() for n in existing)
                needs_rebuild = has_text_idx and not sentinel.exists()
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
                    legacy_sentinel.unlink(missing_ok=True)
                    if needs_rebuild:
                        logger.info(
                            f"FTS index rebuilt (v2, fuzzy-capable) for {collection}"
                        )
                self._fts_failed.pop(collection, None)
                self._fts_ready.add(collection)
        except Exception:
            # Not marked ready, so no sentinel: retried after _FTS_RETRY_S.
            logger.opt(exception=True).warning(
                f"FTS index creation failed for {collection}"
            )
            self._fts_failed[collection] = time.monotonic()

    # ── Stats ────────────────────────────────────────────────────────────

    def list_sources(self, collection: str) -> list[str]:
        """Return sorted distinct ``source`` values in a collection."""
        return self._distinct(collection, "source")

    def collection_summary(self, collection: str) -> dict:
        """Single-scan ``{episodes, chunks, sources, episode_names}`` for
        ``/api/search/stats``; ``episode_names`` lets a caller count distinct
        episodes across collections."""
        if not self.collection_exists(collection):
            return {"episodes": 0, "chunks": 0, "sources": [], "episode_names": []}
        t = self._table(collection)
        rows = t.search().select(["episode", "source"]).limit(1_000_000).to_list()
        episodes = {r["episode"] for r in rows if r.get("episode")}
        sources = sorted({r["source"] for r in rows if r.get("source")})
        return {
            "episodes": len(episodes),
            "chunks": len(rows),
            "sources": sources,
            "episode_names": sorted(episodes),
        }

    def list_speakers(self, collection: str) -> list[str]:
        """Return sorted distinct ``dominant_speaker`` values in a collection.

        Labels that name nobody are left out: they drive speaker pickers in
        the search panel and the bot's autocomplete, where NARRATOR_SPEAKER
        would offer "filter by Narrator" for episodes that were simply never
        diarized. Chunks keep the stored value, so an explicit query for it
        still matches.
        """
        from podcodex.core._utils import is_unattributed

        return [
            s
            for s in self._distinct(collection, "dominant_speaker")
            if not is_unattributed(s)
        ]

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
        ``dominant_speaker``, or the current no-diarization placeholder, are
        skipped: they'd tell us nothing about who is speaking.

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
        from podcodex.core._utils import LEGACY_NARRATOR_SPEAKER, is_unattributed

        groups: dict[str, dict] = {}
        for r in rows:
            sp = r.get("dominant_speaker")
            # Placeholders ("", UNKNOWN, NoDiarization) are nobody. The legacy
            # "Narrator" stays: a show may declare a real speaker by that
            # name, and the index cannot tell which.
            if is_unattributed(sp, declared=(LEGACY_NARRATOR_SPEAKER,)):
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
            collection, with_speakers=True, meta_fields=_EPISODE_META_FIELDS
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
            meta_fields=_EPISODE_META_FIELDS,
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
        clause = build_where(pub_date_min=pub_date_min, pub_date_max=pub_date_max)
        fields = (
            ("episode_title", "episode_number", "broadcast_number", "description")
            if with_detail
            else (
                "episode_title",
                "episode_number",
                "broadcast_number",
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
                "broadcast_number": g.get("broadcast_number"),
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
        table = self._table(collection)  # opening fills _no_pub_date
        cols = ["episode", "end", "meta"]
        if collection not in self._no_pub_date:
            cols.append("pub_date")
        if with_speakers or "source" in meta_fields:
            cols.append("source")
        if with_speakers:
            cols.append("dominant_speaker")
        q = table.search()
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
            defaults[field] = None if field in _INT_META_FIELDS else ""
        groups: dict[str, dict] = {}
        meta_dated: set[str] = set()
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
                if (g[f] is None if f in _INT_META_FIELDS else not g[f])
            ]
            # A row from before the scalar column carries its date only in
            # the meta blob (old tables upgrade episode by episode). Looked up
            # once per episode: an undated one would otherwise parse every row.
            need_date = not g["pub_date"] and ep not in meta_dated
            if missing or need_date:
                try:
                    meta = json.loads(r.get("meta") or "{}")
                except Exception:
                    meta = {}
                if need_date:
                    meta_dated.add(ep)
                    g["pub_date"] = (
                        normalize_pub_date(
                            meta.get("pub_date") or meta.get("rss_pub_date")
                        )
                        or ""
                    )
                for f in missing:
                    v = meta.get(f)
                    if f in _INT_META_FIELDS:
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


def _retry_concurrent(fn, attempts: int = 5):
    """Run *fn*, retrying what another process writing the same table causes.

    Two index subprocesses can create one table or commit to one metadata
    table at the same instant, and no lock reaches across processes. Lance
    then reports a table it lists but cannot open yet ("was not found") or
    refuses the later commit ("Incompatible transaction", a commit conflict);
    both succeed on a re-read. Anything else raises at once.

    Lance raises these untyped, so they are matched by message; a lancedb
    upgrade that rewords them fails
    ``test_two_stores_creating_one_collection_at_once_both_succeed``, which
    produces them for real, rather than silently disabling the retry.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except (ValueError, RuntimeError) as exc:
            msg = str(exc)
            transient = (
                "was not found" in msg
                or "Incompatible transaction" in msg
                or "onflict" in msg
            )
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
    return None


def _require_show_key(show: str) -> None:
    """Refuse an empty show key on a password write.

    ``show_id = ''`` is exactly how a legacy (pre-id) row is stored, so an
    empty key would rewrite or delete every legacy password at once.
    """
    if not (show or "").strip():
        raise ValueError("show id is required")


def _escape(s: str) -> str:
    """Escape single quotes for LanceDB SQL WHERE clauses."""
    return s.replace("'", "''")


def _json_turn(speaker: str) -> str:
    """``"speaker": "<name>"`` as ``json.dumps`` writes it into ``meta``."""
    return json.dumps({"speaker": speaker})[1:-1]


def _like_literal(s: str) -> str:
    """Escape LIKE's metacharacters (``_`` is in every ``SPEAKER_NN``)."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def chunk_map_from_chunks(chunks: list[Hit], text_preview: int = 0) -> list[dict]:
    """Project loaded chunks into the lightweight chunk-map shape.

    Pure counterpart of :meth:`IndexStore.get_chunk_map` so a caller that has
    already loaded the episode's chunks (e.g. to also build a transcript) can
    reuse them instead of reloading.
    """
    from podcodex.core._utils import format_hms

    out: list[dict] = []
    for c in chunks:
        turns = c.speakers or []
        speakers = (
            sorted({t.speaker for t in turns if t.speaker})
            if turns
            else ([c.dominant_speaker] if c.dominant_speaker else [])
        )
        entry: dict = {
            "chunk_index": c.chunk_index,
            "start": c.start,
            "end": c.end,
            "start_hms": format_hms(c.start),
            "speakers": speakers,
        }
        if text_preview > 0:
            entry["text_preview"] = c.text[:text_preview]
        out.append(entry)
    return out


def _center_pos_by_time(chunks: list[Hit], at_time: float) -> int:
    """List position of the chunk covering ``at_time`` (nearest by start on a gap).

    Assumes ``chunks`` is non-empty and ordered by position; always returns a
    valid index.
    """
    for i, c in enumerate(chunks):
        if c.start <= at_time < c.end:
            return i
    return min(range(len(chunks)), key=lambda i: abs(chunks[i].start - at_time))


def _episode_group_to_dict(episode: str, g: dict) -> dict:
    """Shape a ``_aggregate_episodes`` group into the public dict."""
    return {
        "episode": episode,
        "episode_title": g.get("episode_title", ""),
        "pub_date": g.get("pub_date", ""),
        "episode_number": g.get("episode_number"),
        "broadcast_number": g.get("broadcast_number"),
        "description": g.get("description", ""),
        "artwork_url": g.get("artwork_url", ""),
        "source": g.get("source", ""),
        "chunk_count": g["chunk_count"],
        "duration": g["duration"],
        "speakers": sorted(g.get("speakers") or ()),
    }


def build_where(
    *,
    episode: str | None = None,
    episodes: list[str] | None = None,
    source: str | None = None,
    speaker: str | None = None,
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
    speaker_turns: bool = True,
) -> str:
    """Build a LanceDB WHERE clause from equality / range filters.

    ``episodes`` (list) takes precedence over ``episode`` (scalar) when
    both are given. Dates are normalized to ``YYYY-MM-DD``; invalid
    inputs raise ``ValueError``. ``speaker`` matches a chunk where they have
    a turn; ``speaker_turns=False`` keeps it to chunks they dominate.
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
    if speaker and not speaker_turns:
        parts.append(f"dominant_speaker = '{_escape(speaker)}'")
    elif speaker:
        # A semantic chunk spans several turns, so a speaker who is never
        # dominant (a guest who interjects) would be unreachable. Their turns
        # live in the ``meta`` JSON; match the key/value exactly as json.dumps
        # wrote it, with LIKE's own metacharacters escaped.
        parts.append(
            f"(dominant_speaker = '{_escape(speaker)}' "
            f"OR meta LIKE '%{_escape(_like_literal(_json_turn(speaker)))}%')"
        )
    if pub_date_min:
        norm = normalize_pub_date(pub_date_min)
        if not norm:
            raise ValueError(f"Invalid pub_date_min: {pub_date_min!r}")
        parts.append(f"pub_date >= '{norm}'")
    if pub_date_max:
        norm = normalize_pub_date(pub_date_max)
        if not norm:
            raise ValueError(f"Invalid pub_date_max: {pub_date_max!r}")
        parts.append(f"pub_date <= '{norm}'")
    return " AND ".join(parts)


def _row_to_chunk(row: dict, show_label: str = "") -> Hit:
    """Re-inflate a LanceDB row into a ``Hit``.

    ``show_label`` overrides the row's own ``show`` value. The label lives in
    ``_collections``, one row per collection, so a rename costs one metadata
    write; the per-chunk copy is written for debuggability and goes stale the
    moment a show is renamed. Callers always know their collection, so they
    always have the authoritative label.

    The single point where hits are born: drops the vector column and
    LanceDB internals (``_distance``, ``_score``, ``_relevance_score``),
    merges the ``meta`` JSON payload back in, validates as ``Hit``.
    Unknown legacy meta keys land in ``Hit.model_extra``.
    """
    meta_raw = row.get("meta") or "{}"
    try:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
    except Exception:
        meta = {}
    # A null in the meta blob means "absent" (older chunkers wrote explicit
    # nulls); the model's field defaults cover those, but validation would
    # reject the None. Turns need the same treatment one level down: their
    # start/end are non-optional floats, so a single nulled offset would
    # otherwise fail the whole row into the degrade path below.
    chunk: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if key == "speakers" and isinstance(value, list):
            value = [
                {k: v for k, v in turn.items() if v is not None}
                if isinstance(turn, dict)
                else turn
                for turn in value
            ]
        chunk[key] = value
    if chunk.get("pub_date") and not row.get("pub_date"):
        # A row from before the scalar column (never backfilled): keep the
        # meta date for display, where Hit.effective_pub_date normalizes it.
        chunk.setdefault("rss_pub_date", chunk["pub_date"])
    for key in _FILTER_COLUMNS:
        if key in row and row[key] is not None:
            chunk[key] = row[key]
    if show_label:
        chunk["show"] = show_label
    try:
        return Hit.model_validate(chunk)
    except ValidationError as e:
        # A single un-coercible legacy meta value (e.g. episode_number stored
        # as "S2E4") must not take down every read for the whole show. Degrade
        # to the fixed columns, which are schema-typed and always valid.
        # Fires per row, so keep it to one line (no traceback).
        logger.warning(
            f"Dropping unparseable meta for chunk "
            f"{row.get('episode', '?')}#{row.get('chunk_index', '?')}: "
            f"{e.errors()[0].get('msg', 'invalid')}"
        )
        columns = {
            k: row[k] for k in _FILTER_COLUMNS if k in row and row[k] is not None
        }
        return Hit.model_validate(columns)


# ── Process-wide singleton ───────────────────────────────────────────────


# `functools.cache` does not serialize a first call: two threads arriving
# together (API routes in worker threads, MCP tools) would each open the index,
# and opening is not passive (ownership stamp, meta heal, show-id migration).
_default_store_lock = threading.RLock()


@cache
def _default_store() -> IndexStore:
    store = IndexStore()
    # Off the boot path by construction: get_index_store is lazy, and this is
    # the first moment the app has both an open index and its show folders.
    store.migrate_to_show_ids_if_owner()
    return store


def get_index_store() -> IndexStore:
    """Process-wide cached IndexStore using the default path resolution.

    Path resolution lives in ``_resolve_default_index_path``: env var,
    canonical ``<data_dir>/index``, repo-local fallback. Callers
    that need a different path should instantiate ``IndexStore(custom_path)``
    directly.
    """
    if _default_store.cache_info().currsize:
        return _default_store()  # opened already: no lock
    with _default_store_lock:
        return _default_store()


get_index_store.cache_clear = _default_store.cache_clear  # type: ignore[attr-defined]
