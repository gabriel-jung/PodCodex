"""Conflict detection + policy for import.

Pure logic — callers (CLI, API) map their own UX (prompts, modals) to a
``ConflictPolicy`` value before calling ``import_archive``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from podcodex.bundle.manifest import Mode


class ConflictPolicy(StrEnum):
    """How to resolve folder/collection name collisions on import."""

    RENAME = "rename"  # auto-suffix, keep both
    REPLACE = "replace"  # overwrite existing
    ABORT = "abort"  # raise ConflictError


class ConflictError(Exception):
    """Raised when ``ConflictPolicy.ABORT`` is set and a collision is detected."""


def resolve_policy(value: str, mode: Mode) -> ConflictPolicy:
    """Translate a CLI/API ``on_conflict`` argument into a :class:`ConflictPolicy`.

    ``"auto"`` resolves to ``REPLACE`` for index-only bundles (re-deploy
    semantics) and ``RENAME`` for full bundles (no clobber by default).
    """
    if value == "auto":
        return (
            ConflictPolicy.REPLACE if mode == Mode.INDEX_ONLY else ConflictPolicy.RENAME
        )
    return ConflictPolicy(value)


def rename_suffix(name: str, taken: set[str], suffix: str = "-imported") -> str:
    """Return a non-colliding name by appending ``suffix`` (and counters if needed).

    Example:
        rename_suffix("show-a", {"show-a"})
            -> "show-a-imported"
        rename_suffix("show-a", {"show-a", "show-a-imported"})
            -> "show-a-imported-2"
    """
    candidate = f"{name}{suffix}"
    if candidate not in taken:
        return candidate
    i = 2
    while f"{candidate}-{i}" in taken:
        i += 1
    return f"{candidate}-{i}"


def detect_folder_conflict(target: Path) -> bool:
    """True if a folder already exists at ``target``."""
    return target.exists()
