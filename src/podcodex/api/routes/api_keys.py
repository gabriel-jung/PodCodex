"""Routes for the named API key pool.

CRUD plus a `scan-env` endpoint that seeds the pool from `*_API_KEY`
env vars (and the legacy `secrets.env` file). Pool entries are
returned with masked values; full values are only used by the LLM
resolver in-process — never sent to the frontend.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from podcodex.core.api_keys import (
    APIKey,
    APIKeyPublic,
    api_keys_path,
    discover_env_keys,
    find_key,
    load_keys,
    merge_discovered,
    save_keys,
    to_public,
)

router = APIRouter()


class ListResponse(BaseModel):
    path: str
    keys: list[APIKeyPublic]


class CreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    value: str = Field(..., min_length=1)
    suggested_provider: str | None = None


class UpdateRequest(BaseModel):
    """All fields optional — only provided ones are touched."""

    value: str | None = None
    suggested_provider: str | None = None


class ScanResponse(BaseModel):
    added: list[str]
    keys: list[APIKeyPublic]


@router.get("", response_model=ListResponse)
async def list_keys() -> ListResponse:
    file = load_keys()
    return ListResponse(
        path=str(api_keys_path()),
        keys=[to_public(k) for k in file.keys],
    )


@router.post("", response_model=APIKeyPublic, status_code=201)
async def create_key(req: CreateRequest) -> APIKeyPublic:
    file = load_keys()
    if find_key(file, req.name) is not None:
        raise HTTPException(status_code=409, detail=f"Key '{req.name}' already exists")
    new_key = APIKey(
        name=req.name,
        value=req.value,
        suggested_provider=req.suggested_provider,
        source="ui",
    )
    file.keys.append(new_key)
    save_keys(file)
    return to_public(new_key)


@router.patch("/{name}", response_model=APIKeyPublic)
async def update_key(name: str, req: UpdateRequest) -> APIKeyPublic:
    file = load_keys()
    key = find_key(file, name)
    if key is None:
        raise HTTPException(status_code=404, detail=f"Key '{name}' not found")
    if req.value is not None:
        key.value = req.value
    if req.suggested_provider is not None:
        # Allow clearing via empty string.
        key.suggested_provider = req.suggested_provider or None
    save_keys(file)
    return to_public(key)


@router.delete("/{name}", status_code=204)
async def delete_key(name: str) -> None:
    file = load_keys()
    if find_key(file, name) is None:
        raise HTTPException(status_code=404, detail=f"Key '{name}' not found")
    file.keys = [k for k in file.keys if k.name != name]
    save_keys(file)


@router.post("/scan-env", response_model=ScanResponse)
async def scan_env() -> ScanResponse:
    """Seed the pool with `*_API_KEY` vars from the environment.

    Existing names are never overwritten — manual edits stick. Returns
    the names that were newly added and the full updated pool.
    """
    file = load_keys()
    discovered = discover_env_keys()
    file, added = merge_discovered(file, discovered)
    if added:
        save_keys(file)
    return ScanResponse(
        added=added,
        keys=[to_public(k) for k in file.keys],
    )
