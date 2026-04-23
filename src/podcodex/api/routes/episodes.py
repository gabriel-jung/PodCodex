"""Episode metadata routes — keyed lookup and filtered browsing.

Distinct from search: these endpoints return per-episode metadata
(title, pub_date, duration, description, speakers) without running a
vector / FTS query. They power the frontend's episode card and the
MCP ``get_episode`` / ``list_episodes`` tools.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from podcodex.api.routes._helpers import get_index_store
from podcodex.core._utils import episode_display
from podcodex.rag.store import collection_name

router = APIRouter()


class EpisodeListItem(BaseModel):
    episode: str
    episode_title: str = ""
    pub_date: str = ""
    episode_number: int | None = None
    chunk_count: int = 0
    duration: float = 0.0


class EpisodeMeta(BaseModel):
    episode: str
    episode_title: str = ""
    pub_date: str = ""
    episode_number: int | None = None
    description: str = ""
    source: str = ""
    chunk_count: int = 0
    duration: float = 0.0
    speakers: list[str] = []


def _fill_title(d: dict) -> dict:
    d["episode_title"] = episode_display(d)
    return d


@router.get("/{show}", response_model=list[EpisodeListItem])
async def list_show_episodes(
    show: str,
    model: str = "bge-m3",
    chunking: str = "semantic",
    pub_date_min: str | None = None,
    pub_date_max: str | None = None,
    title_contains: str | None = None,
) -> list[dict]:
    """List episodes in a collection, optionally filtered by date / title."""
    col = collection_name(show, model, chunking)
    local = get_index_store()
    try:
        items = local.list_episodes_filtered(
            col,
            pub_date_min=pub_date_min,
            pub_date_max=pub_date_max,
            title_contains=title_contains,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return [_fill_title(dict(x)) for x in items]


@router.get("/{show}/{episode_stem}", response_model=EpisodeMeta)
async def get_show_episode(
    show: str,
    episode_stem: str,
    model: str = "bge-m3",
    chunking: str = "semantic",
) -> dict:
    """Return metadata for a single episode in a collection."""
    col = collection_name(show, model, chunking)
    local = get_index_store()
    meta = local.get_episode(col, episode_stem)
    if meta is None:
        raise HTTPException(404, f"Episode not found: {episode_stem}")
    return _fill_title(dict(meta))
