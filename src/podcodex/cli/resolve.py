"""Shared CLI helpers for show resolution and registered-show enumeration."""

from __future__ import annotations

import sys
from pathlib import Path

from podcodex.ingest.show import show_display


def resolve_show_folder(arg: str) -> tuple[Path, str]:
    """Return ``(absolute folder path, show display name)`` for a CLI arg.

    Accepts either a filesystem path or a show name registered in the app
    config. Calls :func:`sys.exit` with a helpful message on failure so
    individual scripts get consistent UX.
    """
    candidate = Path(arg).expanduser()
    if candidate.is_dir():
        return candidate.resolve(), show_display(candidate)

    from podcodex.api.routes.config import _load as _load_cfg

    cfg = _load_cfg()
    target = arg.strip().lower()
    for folder_path in cfg.show_folders:
        child = Path(folder_path)
        if not child.is_dir():
            continue
        name = show_display(child)
        if name.strip().lower() == target:
            return child.resolve(), name

    sys.exit(f"Show not found: {arg!r}. Pass a folder path or a registered show name.")


def all_registered_show_folders() -> list[Path]:
    """Registered show folders that exist on disk, in registration order."""
    from podcodex.api.routes.config import _load as _load_cfg

    cfg = _load_cfg()
    out: list[Path] = []
    for folder_path in cfg.show_folders:
        child = Path(folder_path)
        if child.is_dir():
            out.append(child.resolve())
    return out


def default_shows_dir() -> Path | None:
    """Best-effort default for ``--shows-dir`` — parent of first registered show.

    Returns ``None`` when no registered shows exist; callers should require
    an explicit ``--shows-dir`` in that case.
    """
    shows = all_registered_show_folders()
    if not shows:
        return None
    return shows[0].parent.resolve()
