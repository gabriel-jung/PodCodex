"""Discord bot access control — per-show password management.

The desktop app's counterpart to the ``podcodex-bot --manage-passwords``
CLI (which still ships, for a bot host with no app). Password plaintext is
returned exactly once in the HTTP response body (never logged, never
stored); the IndexStore only keeps the SHA-256 hash.

The bot process (wherever it runs) reads the same IndexStore on its next
``/admin-reload``, so no hot-restart on the bot side either. The show is
addressed by its password-table key (``show_id``), never its display name:
two shows may share a name. The key is the ``show.toml`` id, or the display
name for a show that has none yet (an index-only show, or a folder that was
never minted one), which is how the password table keys such a show too.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import NamedTuple

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from podcodex.core.show_passwords import hash_show_password
from podcodex.api.routes._helpers import get_index_store

router = APIRouter()


_MIN_MANUAL_LEN = 16
_GENERATED_BYTES = 16  # secrets.token_urlsafe(16) → 22 chars


# ── Response models ─────────────────────────────────────────────────────


class ShowAccess(BaseModel):
    show_id: str  # password-table key: the show id, else its display name
    show: str  # display name
    is_protected: bool


class ShowPasswordSet(BaseModel):
    show_id: str
    show: str
    password: str  # plaintext, returned once only
    generated: bool


class SetPasswordRequest(BaseModel):
    password: str | None = Field(default=None, description="Omit to generate.")


# ── Helpers ─────────────────────────────────────────────────────────────


def _protected_ids() -> set[str]:
    """Ids (and legacy display names) of every password-protected show."""
    return set(get_index_store().get_show_password_entries().keys())


class _KnownShow(NamedTuple):
    label: str
    folder: Path | None  # None for a show known only to the index


def _known_shows() -> dict[str, _KnownShow]:
    """Every show access can be configured for, by password-table key.

    Registered folders (so access can be set before a show is indexed) plus
    shows known only to the index. A registered folder names its own show:
    its label comes from ``show.toml``, and an index row carrying the same id
    adds nothing. A pre-id index row, keyed by label, is dropped when a
    registered show already goes by that label: it is that show's
    unmigrated collection, not another show.
    """
    from podcodex.ingest.show import load_show_meta, show_display
    from podcodex.ingest.show_registry import registered_folders
    from podcodex.rag.index_store import IndexStore

    out: dict[str, _KnownShow] = {}
    try:
        for folder in registered_folders():
            meta = load_show_meta(folder)
            label = show_display(folder)
            out[(meta.id if meta else "") or label] = _KnownShow(label, folder)
    except Exception:
        logger.opt(exception=True).warning("Failed to read registered show folders")
    registered_labels = {k.label.strip().lower() for k in out.values()}
    for meta in get_index_store().get_all_collection_info().values():
        key = IndexStore.show_key(meta)
        label = (meta.get("show") or "").strip() or key
        if not key or key in out:
            continue
        if not (meta.get("show_id") or "").strip() and (
            label.lower() in registered_labels
        ):
            continue
        out[key] = _KnownShow(label, None)
    return out


def _require_known(show_id: str) -> _KnownShow:
    known = _known_shows().get(show_id)
    if known is None:
        raise HTTPException(404, f"Unknown show {show_id!r}.")
    return known


def _write_key(show_id: str, known: _KnownShow) -> str:
    """The key a password is stored under.

    Mints an id for a registered folder that has none, so the password
    follows the show through a rename. Only ``set_password`` calls this: a
    GET must not rewrite ``show.toml``, which would turn a status poll into a
    500 on a read-only folder.
    """
    if known.folder is None:
        return show_id
    from podcodex.ingest.show import ensure_show_id

    return ensure_show_id(known.folder)


# ── Routes ──────────────────────────────────────────────────────────────


@router.get("/passwords", response_model=list[ShowAccess])
def list_passwords() -> list[ShowAccess]:
    """Return every known show with its password-protection status.

    One row per show, not per display name: two shows sharing a name are two
    rows, each with its own status.
    """
    protected = _protected_ids()
    return [
        ShowAccess(show_id=key, show=known.label, is_protected=key in protected)
        for key, known in sorted(
            _known_shows().items(), key=lambda kv: (kv[1].label.lower(), kv[0])
        )
    ]


@router.get("/password", response_model=ShowAccess)
def get_password_status(show_id: str = Query(...)) -> ShowAccess:
    """Per-show protection status."""
    known = _require_known(show_id)
    return ShowAccess(
        show_id=show_id, show=known.label, is_protected=show_id in _protected_ids()
    )


@router.post("/password", response_model=ShowPasswordSet)
def set_password(
    payload: SetPasswordRequest, show_id: str = Query(...)
) -> ShowPasswordSet:
    """Set or rotate the password for a show.

    If ``payload.password`` is empty or omitted the server generates a
    strong 22-char URL-safe token. Otherwise the supplied password is
    used after a minimum-length check (prevents accidentally weak
    passwords; use the generator for something robust).
    """
    known = _require_known(show_id)

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

    key = _write_key(show_id, known)
    try:
        get_index_store().set_show_password(
            key, hash_show_password(plaintext), show_label=known.label
        )
    except IndexOwnershipError as exc:
        raise HTTPException(409, str(exc)) from exc
    return ShowPasswordSet(
        show_id=key, show=known.label, password=plaintext, generated=generated
    )


@router.delete("/password", status_code=204)
def delete_password(show_id: str = Query(...)) -> None:
    """Remove password protection — the show becomes public to the bot."""
    _require_known(show_id)
    from podcodex.rag.index_origin import IndexOwnershipError

    # By the key as given, no minting: a show without an id cannot have an
    # id-keyed row to remove.
    try:
        get_index_store().delete_show_password(show_id)
    except IndexOwnershipError as exc:
        raise HTTPException(409, str(exc)) from exc
