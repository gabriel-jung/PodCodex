"""Discord bot access control — per-show password management.

Replaces the ``podcodex-bot --manage-passwords`` CLI so the desktop app
can set, rotate, and remove show passwords without a terminal. Password
plaintext is returned exactly once in the HTTP response body (never
logged, never stored); the IndexStore only keeps the SHA-256 hash.

The bot process (wherever it runs) reads the same IndexStore on its next
``/admin`` refresh, so no hot-restart on the bot side either.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from podcodex.core.show_passwords import hash_show_password
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


def _protected_ids() -> set[str]:
    """Ids (and legacy display names) of every password-protected show."""
    return set(get_index_store().get_show_password_entries().keys())


def _key_for(show: str, *, mint: bool = False) -> str:
    """Password-table key for a show named *show*.

    Its stable id when the show is a registered folder, falling back to the
    display name for a show that exists only inside the index (nothing to
    mint an id into), which is also how a legacy row is keyed.

    Args:
        show: Display name.
        mint: Whether to create an id when the show has none. Only the write
            handlers pass this: a GET must not rewrite ``show.toml``, which
            would turn a status poll into a 500 on a read-only folder.
    """
    from podcodex.ingest.show import load_show_meta
    from podcodex.ingest.show_registry import folder_for_label

    folder = folder_for_label(show)
    if folder is None:
        return show
    meta = load_show_meta(folder)
    if meta and meta.id:
        return meta.id
    if not mint:
        return show
    from podcodex.ingest.show import ensure_show_id

    return ensure_show_id(folder)


def _all_show_names() -> set[str]:
    """Every known show name: registered show folders plus indexed collections.

    Registered (but not-yet-indexed) shows are included so access can be
    configured before a show is indexed.
    """
    names: set[str] = set()
    try:
        from podcodex.core.app_config import load_config
        from podcodex.ingest.show import load_show_meta

        for folder_path in load_config().show_folders:
            folder = Path(folder_path)
            if not folder.is_dir():
                continue
            meta = load_show_meta(folder)
            names.add((meta.name if meta else None) or folder.name)
    except Exception:
        logger.opt(exception=True).warning("Failed to read registered show folders")
    info = get_index_store().get_all_collection_info()
    names.update(meta.get("show", "") for meta in info.values())
    names.discard("")
    return names


# ── Routes ──────────────────────────────────────────────────────────────


@router.get("/passwords", response_model=list[ShowAccess])
def list_passwords() -> list[ShowAccess]:
    """Return every indexed show with its password-protection status."""
    protected = _protected_ids()
    # One pass over the registered folders, rather than one per show: _key_for
    # walks them all, so calling it in the comprehension is quadratic.
    from podcodex.ingest.show import load_show_meta, show_display
    from podcodex.ingest.show_registry import registered_folders

    by_label: dict[str, set[str]] = {}
    for folder in registered_folders():
        meta = load_show_meta(folder)
        # A set, not one id: two shows may share a display name, and reading
        # the second one's status off the first one's id would be wrong.
        by_label.setdefault(show_display(folder), set()).add(
            (meta.id if meta else "") or show_display(folder)
        )
    return [
        ShowAccess(
            show=name,
            is_protected=bool(by_label.get(name, {name}) & protected),
        )
        for name in sorted(_all_show_names())
    ]


@router.get("/passwords/{show}", response_model=ShowAccess)
def get_password_status(show: str) -> ShowAccess:
    """Per-show protection status."""
    if show not in _all_show_names():
        raise HTTPException(404, f"Unknown show {show!r}.")
    return ShowAccess(show=show, is_protected=_key_for(show) in _protected_ids())


@router.post("/passwords/{show}", response_model=ShowPasswordSet)
def set_password(show: str, payload: SetPasswordRequest) -> ShowPasswordSet:
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

    from podcodex.rag.index_origin import IndexOwnershipError

    try:
        get_index_store().set_show_password(
            _key_for(show, mint=True), hash_show_password(plaintext), show_label=show
        )
    except IndexOwnershipError as exc:
        raise HTTPException(409, str(exc)) from exc
    return ShowPasswordSet(show=show, password=plaintext, generated=generated)


@router.delete("/passwords/{show}", status_code=204)
def delete_password(show: str) -> None:
    """Remove password protection — the show becomes public to the bot."""
    if show not in _all_show_names():
        raise HTTPException(404, f"Unknown show {show!r}.")
    from podcodex.rag.index_origin import IndexOwnershipError

    try:
        get_index_store().delete_show_password(_key_for(show, mint=True))
    except IndexOwnershipError as exc:
        raise HTTPException(409, str(exc)) from exc
