"""podcodex.core.machine_id — stable per-machine identity.

Used to decide whether this process owns an index directory or is looking at
a replica of someone else's. Deliberately dependency-free and stored outside
the index: an id kept *inside* the index would travel with it on rsync, which
is exactly the thing it has to tell apart.

Resolution order:
  1. ``PODCODEX_MACHINE_ID`` (explicit override; needed for deployments whose
     data dir is ephemeral, such as a bot container run without the
     ``podcodex_data`` volume that compose declares).
  2. ``<data_dir>/machine_id``, generated on first use.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from loguru import logger

MACHINE_ID_ENV = "PODCODEX_MACHINE_ID"
MACHINE_ID_FILENAME = "machine_id"


_PROCESS_ID: str | None = None


def machine_id() -> str:
    """Return this machine's stable id, generating and persisting one if needed.

    Never raises: an unwritable data dir degrades to a process-lifetime id
    (the same one for every call in this process), which makes this process
    look like a replica rather than let it wrongly claim ownership of an
    index it may not own.
    """
    global _PROCESS_ID
    override = os.environ.get(MACHINE_ID_ENV, "").strip()
    if override:
        return override

    from podcodex.core.app_paths import data_dir

    path = Path(data_dir()) / MACHINE_ID_FILENAME
    existing = _read(path)
    if existing:
        return existing

    generated = uuid.uuid4().hex
    try:
        _publish(path, generated)
    except FileExistsError:
        # Another process minted one first (the app and a worker starting
        # together on a fresh install); theirs is the machine's id.
        return _read(path) or generated
    except OSError as exc:
        logger.warning(f"Could not persist machine id to {path}: {exc}")
        if _PROCESS_ID is None:
            _PROCESS_ID = generated
        return _PROCESS_ID
    return generated


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def _publish(path: Path, value: str) -> None:
    """Write *value* to *path* only if nothing usable is there yet.

    Complete and exclusive (``atomic_write(exclusive=True)``): this file
    decides index ownership, so neither a torn write nor two processes each
    keeping the id they generated may happen. An empty or unreadable file
    (truncated by a sync conflict or a full disk) is no id at all and is
    replaced, or every call would mint a new one.
    """
    from podcodex.core._utils import atomic_write

    def write(tmp: Path) -> None:
        tmp.write_text(value, encoding="utf-8")

    try:
        atomic_write(path, write, exclusive=True)
    except FileExistsError:
        if _read(path):
            raise
        atomic_write(path, write)
