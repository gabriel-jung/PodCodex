"""Routes for provider profiles.

Built-ins (openai/anthropic/mistral/ollama) are read-only and live in
code. Custom profiles are always openai-compatible — they need a
``base_url`` and persist to disk. Names cannot collide with built-ins.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from podcodex.core.provider_profiles import (
    BUILTIN_PROFILES,
    CustomProfile,
    ProviderProfile,
    find_custom,
    is_builtin,
    list_all,
    load_custom,
    save_custom,
)

router = APIRouter()


class ListResponse(BaseModel):
    profiles: list[ProviderProfile]


class CreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    base_url: str = Field(..., min_length=1)


class UpdateRequest(BaseModel):
    base_url: str | None = None


@router.get("", response_model=ListResponse)
async def list_profiles() -> ListResponse:
    return ListResponse(profiles=list_all())


@router.post("", response_model=ProviderProfile, status_code=201)
async def create_profile(req: CreateRequest) -> ProviderProfile:
    if is_builtin(req.name):
        raise HTTPException(
            status_code=409, detail=f"'{req.name}' collides with a built-in profile"
        )
    file = load_custom()
    if find_custom(file, req.name) is not None:
        raise HTTPException(
            status_code=409, detail=f"Profile '{req.name}' already exists"
        )
    new = CustomProfile(name=req.name, base_url=req.base_url)
    file.profiles.append(new)
    save_custom(file)
    return ProviderProfile(
        name=new.name,
        type="openai-compatible",
        base_url=new.base_url,
        builtin=False,
    )


@router.patch("/{name}", response_model=ProviderProfile)
async def update_profile(name: str, req: UpdateRequest) -> ProviderProfile:
    if is_builtin(name):
        raise HTTPException(status_code=403, detail="Built-in profiles are read-only")
    file = load_custom()
    profile = find_custom(file, name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    if req.base_url is not None:
        new_url = req.base_url.strip()
        if not new_url:
            raise HTTPException(status_code=400, detail="base_url cannot be empty")
        profile.base_url = new_url
    save_custom(file)
    return ProviderProfile(
        name=profile.name,
        type="openai-compatible",
        base_url=profile.base_url,
        builtin=False,
    )


@router.delete("/{name}", status_code=204)
async def delete_profile(name: str) -> None:
    if is_builtin(name):
        raise HTTPException(
            status_code=403, detail="Built-in profiles cannot be deleted"
        )
    file = load_custom()
    if find_custom(file, name) is None:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    file.profiles = [p for p in file.profiles if p.name != name]
    save_custom(file)


# Re-exported for tests that want to assert against the static list.
__all__ = ["router", "BUILTIN_PROFILES"]
