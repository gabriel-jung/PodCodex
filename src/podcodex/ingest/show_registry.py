"""podcodex.ingest.show_registry — resolve registered shows by label or id.

Show display names are labels, not identities: users rename them, two shows
may share one, and ``show.toml`` is hand-editable. Everything that has only a
name to go on resolves through here, so "which show is this" has exactly one
definition and one tie-breaking rule.

Uniqueness is enforced on the write path (``PUT /api/shows/{folder}/meta``
rejects a colliding rename) and merely tolerated here: a duplicate resolves to
the oldest matching folder and logs, rather than failing or picking at random.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from podcodex.ingest.show import load_show_meta, show_display


def registered_folders() -> list[Path]:
    """Every registered show folder that still exists on disk."""
    from podcodex.core.app_config import load_config

    try:
        folders = load_config().show_folders
    except Exception:
        logger.opt(exception=True).warning("show registry: cannot read config")
        return []
    return [p for p in (Path(f) for f in folders) if p.is_dir()]


def folders_for_label(label: str) -> list[Path]:
    """All registered folders whose display name matches *label*, case-folded."""
    target = (label or "").strip().lower()
    if not target:
        return []
    return [
        f for f in registered_folders() if show_display(f).strip().lower() == target
    ]


def folder_for_label(label: str) -> Path | None:
    """The folder for a display name, or None when no registered show matches.

    On a duplicate, the oldest folder wins so repeated calls agree with each
    other, and the ambiguity is logged rather than hidden.
    """
    matches = folders_for_label(label)
    if not matches:
        return None
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: (p.stat().st_ctime, str(p)))
        logger.warning(
            f"show registry: {len(matches)} shows are called {label!r}; "
            f"resolving to {matches[0]}"
        )
    return matches[0]


def show_id_for_label(label: str) -> str:
    """Stable id of the show with this display name ("" when unknown or unminted)."""
    folder = folder_for_label(label)
    if folder is None:
        return ""
    meta = load_show_meta(folder)
    return meta.id if meta else ""


def folder_for_id(show_id: str) -> Path | None:
    """The registered folder carrying this show id, or None.

    Identity-first lookup, unlike ``folder_for_label``: this one cannot be
    wrong, because nothing lets a user change an id.
    """
    if not show_id:
        return None
    for folder in registered_folders():
        meta = load_show_meta(folder)
        if meta and meta.id == show_id:
            return folder
    return None


def label_is_taken(label: str, *, excluding: Path) -> bool:
    """True when another registered show already uses this display name.

    Args:
        label: The proposed display name.
        excluding: The folder being renamed, which does not count against
            itself (and lets a show whose name already collides still be
            edited and saved).
    """
    target = excluding.resolve()
    return any(f.resolve() != target for f in folders_for_label(label))
