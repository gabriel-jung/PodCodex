"""podcodex.core.machine_id — stable per-machine identity.

Used to decide whether this process owns an index directory or is looking at
a replica of someone else's. Deliberately dependency-free and stored outside
the index: an id kept *inside* the index would travel with it on rsync, which
is exactly the thing it has to tell apart.

Resolution order:
  1. ``PODCODEX_MACHINE_ID`` (explicit override; needed for deployments whose
     data dir is ephemeral, such as the bot container, which by default mounts
     only the index directory).
  2. ``<data_dir>/machine_id``, generated on first use.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from loguru import logger

MACHINE_ID_ENV = "PODCODEX_MACHINE_ID"
MACHINE_ID_FILENAME = "machine_id"


def machine_id() -> str:
    """Return this machine's stable id, generating and persisting one if needed.

    Never raises: an unwritable data dir degrades to a process-lifetime id,
    which makes this process look like a replica rather than let it wrongly
    claim ownership of an index it may not own.
    """
    override = os.environ.get(MACHINE_ID_ENV, "").strip()
    if override:
        return override

    from podcodex.core.app_paths import data_dir

    path = Path(data_dir()) / MACHINE_ID_FILENAME
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except (OSError, UnicodeDecodeError):
        pass

    generated = uuid.uuid4().hex
    try:
        from podcodex.core._utils import atomic_write

        # Atomic on purpose: this file decides index ownership, and a torn
        # write would read back as a different machine.
        atomic_write(path, lambda p: p.write_text(generated, encoding="utf-8"))
    except OSError as exc:
        logger.warning(f"Could not persist machine id to {path}: {exc}")
    return generated
