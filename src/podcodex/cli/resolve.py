"""Shared CLI helpers for show resolution and registered-show enumeration."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from podcodex.ingest.show import SHOW_META_FILENAME, show_display
from podcodex.ingest.show_registry import folder_for_label, registered_folders


def resolve_show_folder(arg: str) -> tuple[Path, str]:
    """Return ``(absolute folder path, show display name)`` for a CLI arg.

    A registered show name wins, resolved by ``show_registry`` so the CLI
    picks the same show as the app when two share a name. Otherwise the
    argument must be a show folder: a directory holding ``show.toml``, or a
    registered folder. An arbitrary directory is refused, since exporting it
    would write a ``show.toml`` into it. Calls :func:`sys.exit` with a
    helpful message on failure so individual scripts get consistent UX.
    """
    folder = folder_for_label(arg)
    if folder is not None:
        return folder.resolve(), show_display(folder)

    candidate = Path(arg).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
        registered = {f.resolve() for f in registered_folders()}
        if (resolved / SHOW_META_FILENAME).is_file() or resolved in registered:
            return resolved, show_display(resolved)
        sys.exit(
            f"Not a show folder (no {SHOW_META_FILENAME}): {resolved}. "
            "Add it to the app (or pass a registered show name) first."
        )

    sys.exit(f"Show not found: {arg!r}. Pass a folder path or a registered show name.")


def all_registered_show_folders() -> list[Path]:
    """Registered show folders that exist on disk, in registration order."""
    return [f.resolve() for f in registered_folders()]


def default_shows_dir() -> Path | None:
    """Best-effort default for ``--shows-dir`` — parent of first registered show.

    Returns ``None`` when no registered shows exist; callers should require
    an explicit ``--shows-dir`` in that case.
    """
    shows = all_registered_show_folders()
    if not shows:
        return None
    return shows[0].parent.resolve()


@contextmanager
def expected_errors() -> Iterator[None]:
    """Turn the failures a user can cause into a one-line exit, not a traceback.

    A missing archive, a wrong file, a name conflict or a replica index is
    something to fix and retry, not a bug report.
    """
    from podcodex.bundle import ArchiveCorruptError, ConflictError, ManifestVersionError
    from podcodex.rag.index_origin import IndexOwnershipError

    try:
        yield
    except (
        ArchiveCorruptError,
        ConflictError,
        ManifestVersionError,
        IndexOwnershipError,
        ValueError,
        OSError,
    ) as exc:
        # A broad ValueError/OSError can also be a bug, which needs the stack:
        # PODCODEX_DEBUG=1 lets it through.
        import os

        if os.environ.get("PODCODEX_DEBUG"):
            raise
        sys.exit(f"error: {exc} (set PODCODEX_DEBUG=1 for the traceback)")
