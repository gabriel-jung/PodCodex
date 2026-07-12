"""Search routes — hybrid retrieval over the global LanceDB IndexStore."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, field_validator

from podcodex.api.routes._helpers import AUDIO_EXTS, get_index_store
from podcodex.core._utils import humanize_stem
from podcodex.rag.index_store import _normalize_pub_date
from podcodex.rag.search_service import (
    SearchCollection,
    exact_search as svc_exact_search,
    hybrid_search as svc_hybrid_search,
    load_show_rag_prefs,
    random_quote as svc_random_quote,
    resolve_collections,
)

router = APIRouter()


def _resolve_req_cols(show: str, model: str, chunking: str) -> list[SearchCollection]:
    """Resolve a request's show to its collections via the shared resolver.

    Shared by ``search_query``, ``exact_search``, and ``random_quote``: all
    three pick collections the same way, override-first then falling back
    through show prefs and defaults so a show indexed only under a
    non-default model stays reachable.
    """
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


# ── Config ────────────────────────────────────────────────


@router.get("/config")
async def search_config() -> dict:
    """Return available models, chunking strategies, and defaults."""
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
    speakers: list[dict] | None = None
    accent_match: bool = False
    fuzzy_match: bool = False
    match_text: str | None = None


@router.post("/query", response_model=list[SearchResult])
async def search_query(req: SearchRequest) -> list[dict]:
    """Hybrid search over the global LanceDB index."""
    from podcodex.rag.defaults import MODELS

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


def _result_to_dict(r: dict, audio_lookup: dict[str, dict] | None = None) -> dict:
    stem = r.get("episode", "")
    title = r.get("episode_title") or humanize_stem(stem)
    show_name = r.get("show", "")
    show_entry = (
        audio_lookup.get(show_name)
        if audio_lookup is not None and stem and show_name
        else None
    ) or {}
    audio_path = (show_entry.get("audio") or {}).get(stem, "")
    folder = show_entry.get("folder", "")
    output_dir = str(Path(folder) / stem) if folder and stem else ""
    return {
        "text": r.get("text", ""),
        "episode": title,
        "episode_stem": stem,
        "episode_number": r.get("episode_number"),
        "audio_path": audio_path,
        "output_dir": output_dir,
        "speaker": r.get("dominant_speaker", r.get("speaker", "")),
        "start": r.get("start", 0.0),
        "end": r.get("end", 0.0),
        "score": r.get("score", 0.0),
        "source": r.get("source", ""),
        "pub_date": (
            r.get("pub_date") or _normalize_pub_date(r.get("rss_pub_date")) or ""
        ),
        "speakers": r.get("speakers"),
        "accent_match": bool(r.get("accent_match", False)),
        "fuzzy_match": bool(r.get("fuzzy_match", False)),
        "match_text": r.get("match_text"),
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
async def exact_search(req: ExactRequest) -> list[dict]:
    """Phrase search: returns all exact, accent-variant, and near-typo matches."""
    cols = _resolve_req_cols(req.show, req.model, req.chunking)
    if not cols:
        return []
    try:
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
async def random_quote(req: RandomRequest) -> dict | None:
    """Pick a random indexed chunk (optionally filtered)."""
    cols = _resolve_req_cols(req.show, req.model, req.chunking)
    if not cols:
        return None
    try:
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
    return _result_to_dict({**chunk, "score": 1.0}, _build_audio_lookup())


# ── Distinct speakers ────────────────────────────────────


@router.get("/speakers")
async def list_indexed_speakers(
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
async def index_stats(show: str = "") -> dict:
    """Return index statistics, optionally scoped to one show."""
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
