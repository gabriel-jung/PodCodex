"""Search routes — hybrid retrieval over the global LanceDB IndexStore."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import AUDIO_EXTS, get_index_store
from podcodex.rag.hit import Hit, SpeakerTurn

# podcodex.rag.search_service is imported inside the handlers below, not
# here: it pulls lancedb, pyarrow and numpy (~300 ms) and this module is on
# the API's startup import path, while none of it is needed to answer
# anything but a search. The store is opened at startup anyway by the
# lifespan's warmup thread, so the first search does not wait on it either.
if TYPE_CHECKING:
    from podcodex.rag.search_service import SearchCollection

router = APIRouter()

# Retrieval handlers below are sync def: embedding inference and LanceDB
# scans take seconds, and running them on the event loop froze every other
# endpoint. FastAPI's threadpool takes them instead. This lock preserves the
# serialization the event loop used to provide: the shared embedder's
# encode()/tokenizer thread-safety is unproven, so concurrent searches queue
# here rather than interleave inside the model.
_retrieval_lock = threading.Lock()


def _resolve_req_cols(show: str, model: str, chunking: str) -> list[SearchCollection]:
    """Resolve a request's show to its collections via the shared resolver.

    Shared by ``search_query``, ``exact_search``, and ``random_quote``: all
    three pick collections the same way, override-first then falling back
    through show prefs and defaults so a show indexed only under a
    non-default model stays reachable.
    """
    from podcodex.rag.search_service import (
        load_show_rag_prefs,
        resolve_collections,
    )

    return resolve_collections(
        get_index_store().get_all_collection_info(),
        shows=[show],
        show_prefs=load_show_rag_prefs(),
        override=(model, chunking),
    )


# Cache key combines folder mtime + show.toml mtime so that renaming a show
# (which only touches show.toml) invalidates the cached display name.
# Cached value: (key, show_name, folder_path, {stem: audio_path}).
_AUDIO_LOOKUP_CACHE: dict[
    str, tuple[tuple[float, float], str, str, dict[str, str]]
] = {}


def _build_audio_lookup() -> dict[str, dict]:
    """Per-request map: show name → {"folder": str, "audio": {stem: audio_path}}.

    ``folder`` is the show folder path; combined with an episode stem it
    yields ``output_dir`` for episodes that have no audio file (e.g. YouTube
    flat-extraction or subtitle-only imports).
    """
    from podcodex.api.routes.config import _load
    from podcodex.ingest.show import SHOW_META_FILENAME, load_show_meta

    cfg = _load()
    active_folders = set(cfg.show_folders)
    # Drop entries for folders no longer in config (folder unregistered).
    for stale in [k for k in _AUDIO_LOOKUP_CACHE if k not in active_folders]:
        _AUDIO_LOOKUP_CACHE.pop(stale, None)

    out: dict[str, dict] = {}
    for folder_path in cfg.show_folders:
        p = Path(folder_path)
        try:
            folder_m = p.stat().st_mtime
        except (FileNotFoundError, NotADirectoryError):
            continue
        if not p.is_dir():
            continue
        try:
            meta_m = (p / SHOW_META_FILENAME).stat().st_mtime
        except (FileNotFoundError, OSError):
            meta_m = 0.0
        key = (folder_m, meta_m)

        cached = _AUDIO_LOOKUP_CACHE.get(folder_path)
        if cached is not None and cached[0] == key:
            out[cached[1]] = {"folder": cached[2], "audio": cached[3]}
            continue

        meta = load_show_meta(p)
        name = (meta.name if meta else None) or p.name
        stems: dict[str, str] = {}
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                stems[f.stem] = str(f)
        _AUDIO_LOOKUP_CACHE[folder_path] = (key, name, folder_path, stems)
        out[name] = {"folder": folder_path, "audio": stems}
    return out


# ── Embedder warm-up ─────────────────────────────────────


_warm_started = False
_warm_lock = threading.Lock()


def _warm_show_sync(show: str) -> None:
    """Resolve the show the way a search does, then load that embedder.

    Runs on a worker thread, which is what lets it call ``_resolve_req_cols``
    — that reaches ``rag.search_service`` (lancedb + pyarrow + numpy), the
    import this whole change moved off the request path. Resolution has to
    go through the same helper the handlers use: a show can pin a
    non-default model through its RAG prefs, and guessing from the raw
    collection list would warm a multi-GB model no search then uses.
    """
    try:
        from podcodex.rag.defaults import DEFAULT_CHUNKING, DEFAULT_MODEL

        cols = _resolve_req_cols(show, DEFAULT_MODEL, DEFAULT_CHUNKING)
        if not cols:
            # Nothing indexed for this show, so there is no model to load and
            # a search would return nothing either. Release the latch: on a
            # fresh install the first show opened is usually un-indexed, and
            # burning the one shot here would leave every later search cold.
            _release_warm_latch()
            return

        from podcodex.rag.retriever import get_retriever

        get_retriever(cols[0].model).embedder
        logger.info(f"search warm: retriever ready (model={cols[0].model})")
    except Exception:
        # Best-effort: the search path builds it on demand anyway, so a
        # failure costs latency, never correctness.
        _release_warm_latch()
        logger.opt(exception=True).debug("search warm failed")


def _release_warm_latch() -> None:
    global _warm_started
    with _warm_lock:
        _warm_started = False


def _warm_show_async(show: str) -> None:
    """Load the show's embedder in the background, once per process.

    A cold first search costs ~6 s: torch and sentence_transformers import
    (~3 s), then the model loads from disk (~2.5 s). Every search after it
    is ~60 ms. Starting on the signal that the search panel opened spends
    that while the user types instead of after they hit enter, and a
    session that never opens search never pays it.

    Once per process, not once per model: the expensive half is the import,
    which is shared, and residency is already bounded by the caches in
    ``rag.retriever`` and ``rag.embedder``. A second cap here would be a
    second answer about the same memory. The latch is released again if the
    attempt loaded nothing, so a no-op does not spend it.
    """
    global _warm_started
    if not show:
        return
    with _warm_lock:
        if _warm_started:
            return
        _warm_started = True
    threading.Thread(
        target=_warm_show_sync, args=(show,), name="search-warm", daemon=True
    ).start()


# ── Config ────────────────────────────────────────────────


@router.get("/config")
def search_config() -> dict:
    """Return available models, chunking strategies, and defaults.

    Deliberately does *not* warm the embedder, even though it is fetched on
    mount: it carries no show, so it could only guess at the model, and
    ``SearchPanel`` fetches ``/stats`` alongside it. Warming from both meant
    loading the index-wide default *and* the show's model — up to ~5 GB
    resident, one of it useless. ``/stats`` knows which one a search will
    actually use.
    """
    from podcodex.rag.defaults import (
        ALPHA,
        CHUNKING_STRATEGIES,
        DEFAULT_CHUNKING,
        DEFAULT_MODEL,
        MODELS,
        TOP_K,
    )

    return {
        "models": {
            key: {"label": spec.label, "description": spec.description}
            for key, spec in MODELS.items()
        },
        "chunking_strategies": CHUNKING_STRATEGIES,
        "defaults": {
            "model": DEFAULT_MODEL,
            "chunking": DEFAULT_CHUNKING,
            "alpha": ALPHA,
            "top_k": TOP_K,
        },
    }


# ── Query ─────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    top_k: int = 5
    alpha: float = 0.5
    episode: str | None = None
    episodes: list[str] | None = None
    speaker: str | None = None
    source: str | None = None
    pub_date_min: str | None = None
    pub_date_max: str | None = None

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        """Validate that top_k is at least 1."""
        if v < 1:
            raise ValueError("top_k must be at least 1")
        return v

    @field_validator("alpha")
    @classmethod
    def alpha_in_range(cls, v: float) -> float:
        """Validate that alpha is between 0.0 and 1.0 inclusive."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        return v


class SearchResult(BaseModel):
    text: str
    episode: str
    episode_stem: str = ""
    episode_number: int | None = None
    audio_path: str = ""
    output_dir: str = ""
    speaker: str
    start: float
    end: float
    score: float
    source: str
    pub_date: str = ""
    speakers: list[SpeakerTurn] | None = None
    accent_match: bool = False
    fuzzy_match: bool = False
    match_text: str | None = None


@router.post("/query", response_model=list[SearchResult])
def search_query(req: SearchRequest) -> list[dict]:
    """Hybrid search over the global LanceDB index."""
    from podcodex.rag.defaults import MODELS
    from podcodex.rag.search_service import hybrid_search as svc_hybrid_search

    if req.model not in MODELS:
        raise HTTPException(400, f"Unknown model: {req.model}")

    cols = _resolve_req_cols(req.show, req.model, req.chunking)
    logger.info(
        "Search: show={!r} cols={!r} episode={!r}",
        req.show,
        [c.name for c in cols],
        req.episode,
    )
    if not cols:
        return []
    try:
        with _retrieval_lock:
            results = svc_hybrid_search(
                req.query,
                cols,
                top_k=req.top_k,
                alpha=req.alpha,
                episode=req.episode,
                episodes=req.episodes,
                speaker=req.speaker,
                source=req.source,
                pub_date_min=req.pub_date_min,
                pub_date_max=req.pub_date_max,
            )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        logger.opt(exception=True).warning("Search failed for show {}", req.show)
        results = []

    logger.info("Search: {} result(s)", len(results))
    audio_lookup = _build_audio_lookup()
    return [_result_to_dict(r, audio_lookup) for r, _col in results]


def _result_to_dict(r: Hit, audio_lookup: dict[str, dict] | None = None) -> dict:
    stem = r.episode
    show_entry = (
        audio_lookup.get(r.show)
        if audio_lookup is not None and stem and r.show
        else None
    ) or {}
    audio_path = (show_entry.get("audio") or {}).get(stem, "")
    folder = show_entry.get("folder", "")
    output_dir = str(Path(folder) / stem) if folder and stem else ""
    return {
        "text": r.text,
        "episode": r.display_title,
        "episode_stem": stem,
        "episode_number": r.episode_number,
        "audio_path": audio_path,
        "output_dir": output_dir,
        # speaker_label is turn-first: a /random hit flattened to one turn
        # reports that turn's speaker, not the chunk's dominant one.
        "speaker": r.speaker_label,
        "start": r.start,
        "end": r.end,
        "score": r.score or 0.0,
        "source": r.source,
        "pub_date": r.effective_pub_date,
        "speakers": r.speakers,
        "accent_match": r.accent_match,
        "fuzzy_match": r.fuzzy_match,
        "match_text": r.match_text,
    }


# ── Exact (token-match) search ───────────────────────────


class ExactRequest(BaseModel):
    query: str
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    episode: str | None = None
    episodes: list[str] | None = None
    speaker: str | None = None
    source: str | None = None
    pub_date_min: str | None = None
    pub_date_max: str | None = None


@router.post("/exact", response_model=list[SearchResult])
def exact_search(req: ExactRequest) -> list[dict]:
    """Phrase search: returns all exact, accent-variant, and near-typo matches."""
    from podcodex.rag.search_service import exact_search as svc_exact_search

    cols = _resolve_req_cols(req.show, req.model, req.chunking)
    if not cols:
        return []
    try:
        with _retrieval_lock:
            hits = svc_exact_search(
                req.query,
                cols,
                episode=req.episode,
                episodes=req.episodes,
                speaker=req.speaker,
                source=req.source,
                pub_date_min=req.pub_date_min,
                pub_date_max=req.pub_date_max,
            )
    except ValueError as e:
        raise HTTPException(400, str(e))
    audio_lookup = _build_audio_lookup()
    return [_result_to_dict(h, audio_lookup) for h, _col in hits]


# ── Random quote ─────────────────────────────────────────


class RandomRequest(BaseModel):
    show: str
    model: str = "bge-m3"
    chunking: str = "semantic"
    episode: str | None = None
    episodes: list[str] | None = None
    speaker: str | None = None
    source: str | None = None
    pub_date_min: str | None = None
    pub_date_max: str | None = None


@router.post("/random", response_model=SearchResult | None)
def random_quote(req: RandomRequest) -> dict | None:
    """Pick a random indexed chunk (optionally filtered)."""
    from podcodex.rag.search_service import random_quote as svc_random_quote

    cols = _resolve_req_cols(req.show, req.model, req.chunking)
    if not cols:
        return None
    try:
        with _retrieval_lock:
            picked = svc_random_quote(
                cols,
                episode=req.episode,
                episodes=req.episodes,
                speaker=req.speaker,
                source=req.source,
                pub_date_min=req.pub_date_min,
                pub_date_max=req.pub_date_max,
            )
    except ValueError as e:
        raise HTTPException(400, str(e))
    if picked is None:
        return None
    chunk, _col = picked
    return _result_to_dict(
        chunk.model_copy(update={"score": 1.0}), _build_audio_lookup()
    )


# ── Distinct speakers ────────────────────────────────────


@router.get("/speakers")
def list_indexed_speakers(
    show: str,
    model: str = "bge-m3",
    chunking: str = "semantic",
) -> list[str]:
    """Distinct ``dominant_speaker`` values in a show's collection.

    Resolves through the shared resolver, same as query/exact/random, so a
    show indexed only under a non-default model still returns its speakers
    instead of silently missing.
    """
    cols = _resolve_req_cols(show, model, chunking)
    if not cols:
        return []
    return get_index_store().list_speakers(cols[0].name)


# ── Index stats ──────────────────────────────────────────


@router.get("/stats")
def index_stats(show: str = "") -> dict:
    """Return index statistics, optionally scoped to one show.

    Also the embedder warm signal: ``SearchPanel`` fetches this when it
    mounts, which is the earliest indication a search is coming, and unlike
    ``/config`` it knows the show — so the warm can resolve the model that
    show will actually search with.
    """
    _warm_show_async(show)

    local = get_index_store()
    collections = local.list_collections(show=show)

    stats: list[dict] = []
    total_episodes = 0
    total_chunks = 0
    for col in collections:
        info = local.get_collection_info(col)
        summary = local.collection_summary(col)
        stats.append(
            {
                "collection": col,
                "model": info["model"] if info else "",
                "chunking": info["chunker"] if info else "",
                "episodes": summary["episodes"],
                "chunks": summary["chunks"],
                "sources": summary["sources"],
            }
        )
        total_episodes += summary["episodes"]
        total_chunks += summary["chunks"]
    return {
        "collections": stats,
        "total_episodes": total_episodes,
        "total_chunks": total_chunks,
    }
