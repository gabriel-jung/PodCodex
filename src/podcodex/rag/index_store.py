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


def _accent_score(query: str, text: str, folded_t: str, folded_q: str) -> float:
    """Similarity between query and its matched span in text (accent tier).

    When no ligatures are present, the folded-text index maps directly to the
    original text position.  When ligatures exist (ﬁ→fi shifts offsets), a
    short scan locates the correct original span.
    """
    if len(folded_t) == len(text):
        idx = folded_t.find(folded_q)
        span = text[idx : idx + len(folded_q)]
    else:
        span = _find_original_span(text, folded_q) or query
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


def _fuzzy_match(query: str, text: str, max_dist: int) -> float | None:
    """Return similarity score if any word in text is within max_dist edits of
    the folded single-word query, else None."""
    q = fold_text(query)
    best: float | None = None
    for raw in _WORD_RE.split(text):
        if not raw:
            continue
        w = fold_text(raw)
        d = _levenshtein(q, w, max_dist)
        if d <= max_dist:
            ratio = 1.0 - d / max(len(q), len(w), 1)
            if best is None or ratio > best:
                best = ratio
    return best


def _chunk_key(chunk: dict) -> str:
    return f"{chunk.get('episode', '')}|{chunk.get('start', 0)}"


_MIN_APPROX_LEN = 6  # tokens shorter than this must match exactly in multi-word fuzzy


def _phrase_fuzzy(
    match_words: list[str],
    text: str,
    max_dist: int,
    phrase_len: int,
) -> float | None:
    """Sliding-window phrase fuzzy match.

    Scans the text in windows of ``phrase_len + 2`` tokens. Within each window,
    every token in ``match_words`` must have a match within ``max_dist`` edits
    (tokens shorter than ``_MIN_APPROX_LEN`` must match exactly). Exactly one
    token may be approximate; if all are exact the window is skipped (phrase
    would have been caught by substring tiers).

    Args:
        match_words: Filtered query tokens to match (≥ 3 chars).
        text: Chunk text to search.
        max_dist: Maximum Levenshtein distance for long tokens.
        phrase_len: Total original token count (including short ones), used to
            size the sliding window so filtered tokens don't shrink it.

    Returns:
        Best window score or None if no qualifying window found.
    """
    folded_qws = [fold_text(qw) for qw in match_words]
    raw_tokens = [w for w in _WORD_RE.split(text) if w]
    text_tokens = [fold_text(w) for w in raw_tokens]
    window = phrase_len + 2  # +2 slack for phrase variation
    best: float | None = None

    for i in range(max(1, len(text_tokens) - window + 1)):
        win_folded = text_tokens[i : i + window]
        approx_count = 0
        approx_qw = ""
        approx_j = -1
        ok = True
        used: set[int] = set()
        for qw in folded_qws:
            best_d, best_j = 999, -1
            for j, tw in enumerate(win_folded):
                if j in used:
                    continue
                d = _levenshtein(qw, tw, best_d - 1)
                if d < best_d:
                    best_d, best_j = d, j
            effective_max = max_dist if len(qw) >= _MIN_APPROX_LEN else 0
            if best_d > effective_max:
                ok = False
                break
            if best_d > 0:
                approx_count += 1
                if approx_count > 1:
                    ok = False
                    break
                approx_qw, approx_j = qw, best_j
            if best_j >= 0:
                used.add(best_j)
        if ok and approx_count == 1:
            # Score: char match on the one approximate word only — using the
            # full phrase would dilute the typo signal into near-100%.
            fm = win_folded[approx_j]
            matching = sum(
                b.size
                for b in SequenceMatcher(None, approx_qw, fm).get_matching_blocks()
            )
            score = matching / max(len(approx_qw), 1)
            if best is None or score > best:
                best = score
    return best


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
        fuzziness: int = 0,
    ) -> list[dict]:
        """Full-text search (Tantivy, tokenized).

        Creates the FTS index lazily on first call per collection.

        Args:
            collection: Collection name.
            query: Free-text query.
            top_k: Maximum results.
            episode, source, speaker: SQL-style equality filters.
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
        clause = _build_where(episode=episode, source=source, speaker=speaker)
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
        source: str | None = None,
        speaker: str | None = None,
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

        # FTS pre-filter: fuzziness=2 via MatchQuery on accent-folded token(s).
        # Tantivy keeps accents in the index, so accent mismatch + typo = 2 edits.
        # For multi-word: run one fuzzy FTS per token and intersect chunk hits so
        # every meaningful token must appear (approximately) in the same chunk.
        if len(fts_tokens) == 1:
            candidates = self.search_fts(
                collection,
                fold_text(fts_tokens[0]),
                10_000,
                episode=episode,
                source=source,
                speaker=speaker,
                fuzziness=2,
            )
        else:
            hit_count: Counter = Counter()
            chunk_map: dict[str, dict] = {}
            for token in fts_tokens:
                ft = fold_text(token)
                for h in self.search_fts(
                    collection,
                    ft,
                    10_000,
                    episode=episode,
                    source=source,
                    speaker=speaker,
                    fuzziness=2,
                ):
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
            if query_lower in text.lower():
                exact.append({**c, "score": 1.0})
            else:
                folded_t = fold_text(text)
                if folded_q in folded_t:
                    accent_only.append(
                        {**c, "score": _accent_score(query, text, folded_t, folded_q)}
                    )
                else:
                    score = (
                        _fuzzy_match(fts_tokens[0], text, max_dist)
                        if single_word
                        else _phrase_fuzzy(
                            fts_tokens, text, max_dist, phrase_len=len(all_tokens)
                        )
                    )
                    if score is not None:
                        fuzzy_only.append({**c, "score": score})

        exact.sort(key=lambda c: (c.get("episode", ""), c.get("start", 0.0)))
        accent_only.sort(key=lambda c: (c.get("episode", ""), c.get("start", 0.0)))
        fuzzy_only.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        return exact, accent_only, fuzzy_only

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
        """Return per-episode ``{episode, episode_title, chunk_count, duration, speakers}``."""
        if not self.collection_exists(collection):
            return []
        rows = (
            self._table(collection)
            .search()
            .select(["episode", "dominant_speaker", "end", "meta"])
            .limit(1_000_000)
            .to_list()
        )
        groups: dict[str, dict] = {}
        for r in rows:
            ep = r.get("episode")
            if ep is None:
                continue
            g = groups.setdefault(
                ep,
                {
                    "chunk_count": 0,
                    "duration": 0.0,
                    "speakers": set(),
                    "episode_title": "",
                },
            )
            g["chunk_count"] += 1
            end = float(r.get("end") or 0.0)
            if end > g["duration"]:
                g["duration"] = end
            sp = r.get("dominant_speaker")
            if sp:
                g["speakers"].add(sp)
            if not g["episode_title"]:
                try:
                    meta = json.loads(r.get("meta") or "{}")
                except Exception:
                    meta = {}
                if meta.get("episode_title"):
                    g["episode_title"] = meta["episode_title"]
        return [
            {
                "episode": ep,
                "episode_title": g["episode_title"],
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
