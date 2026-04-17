"""Discord bot access control — per-show password management.

Replaces the ``podcodex-bot --manage-passwords`` CLI so the desktop app
can set, rotate, and remove show passwords without a terminal. Password
plaintext is returned exactly once in the HTTP response body (never
logged, never stored); the IndexStore only keeps the SHA-256 hash.

The bot process (wherever it runs) reads the same IndexStore on its next
``/admin`` refresh, so no hot-restart on the bot side either.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from podcodex.api.routes._helpers import get_index_store

router = APIRouter()


_MIN_MANUAL_LEN = 16
_GENERATED_BYTES = 16  # secrets.token_urlsafe(16) → 22 chars


# ── Response models ─────────────────────────────────────────────────────


class ShowAccess(BaseModel):
    show: str
    is_protected: bool


class ShowPasswordSet(BaseModel):
    show: str
    password: str  # plaintext, returned once only
    generated: bool


class SetPasswordRequest(BaseModel):
    password: str | None = Field(default=None, description="Omit to generate.")


# ── Helpers ─────────────────────────────────────────────────────────────


def _all_show_names() -> list[str]:
    """Every show name known to the IndexStore (from collection metadata)."""
    info = get_index_store().get_all_collection_info()
    names = {meta.get("show") for meta in info.values() if meta.get("show")}
    return sorted(names)


def _hash(password: str) -> str:
    return f"sha256:{hashlib.sha256(password.encode()).hexdigest()}"


# ── Routes ──────────────────────────────────────────────────────────────


@router.get("/passwords", response_model=list[ShowAccess])
async def list_passwords() -> list[ShowAccess]:
    """Return every indexed show with its password-protection status."""
    store = get_index_store()
    protected = set(store.get_show_passwords().keys())
    return [
        ShowAccess(show=name, is_protected=name in protected)
        for name in _all_show_names()
    ]


@router.get("/passwords/{show}", response_model=ShowAccess)
async def get_password_status(show: str) -> ShowAccess:
    """Per-show protection status."""
    if show not in _all_show_names():
        raise HTTPException(404, f"Unknown show {show!r}.")
    protected = show in get_index_store().get_show_passwords()
    return ShowAccess(show=show, is_protected=protected)


@router.post("/passwords/{show}", response_model=ShowPasswordSet)
async def set_password(show: str, payload: SetPasswordRequest) -> ShowPasswordSet:
    """Set or rotate the password for a show.

    If ``payload.password`` is empty or omitted the server generates a
    strong 22-char URL-safe token. Otherwise the supplied password is
    used after a minimum-length check (prevents accidentally weak
    passwords; use the generator for something robust).
    """
    if show not in _all_show_names():
        raise HTTPException(404, f"Unknown show {show!r}.")

    supplied = (payload.password or "").strip()
    generated = not supplied
    if generated:
        plaintext = secrets.token_urlsafe(_GENERATED_BYTES)
    else:
        if len(supplied) < _MIN_MANUAL_LEN:
            raise HTTPException(
                422,
                f"Manual passwords must be at least {_MIN_MANUAL_LEN} characters. "
                "Omit the password field to auto-generate a strong one.",
            )
        plaintext = supplied

    get_index_store().set_show_password(show, _hash(plaintext))
    return ShowPasswordSet(show=show, password=plaintext, generated=generated)


@router.delete("/passwords/{show}", status_code=204)
async def delete_password(show: str) -> None:
    """Remove password protection — the show becomes public to the bot."""
    if show not in _all_show_names():
        raise HTTPException(404, f"Unknown show {show!r}.")
    get_index_store().delete_show_password(show)
