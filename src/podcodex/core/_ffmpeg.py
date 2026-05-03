"""Resolve the system ffmpeg binary path.

PodCodex shells out to ffmpeg for clip extraction, voice-sample upload
conversion, and through whisperx / faster-whisper which hard-code the
bare ``"ffmpeg"`` command. Bundling GPL-built ffmpeg (libx264 / libx265)
would contaminate the MIT release — see LICENSE_AUDIT.md — so we rely on
a system install instead.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

PODCODEX_FFMPEG_EXE_ENV = "PODCODEX_FFMPEG_EXE"


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Absolute path to the ffmpeg binary, or ``"ffmpeg"`` as a last resort.

    Resolution order:
      1. ``$PODCODEX_FFMPEG_EXE`` (override for non-PATH installs).
      2. ``shutil.which("ffmpeg")``.
      3. Bare ``"ffmpeg"`` — subprocess raises ``FileNotFoundError``;
         :func:`ffmpeg_available` lets callers pre-check.
    """
    override = os.environ.get(PODCODEX_FFMPEG_EXE_ENV, "").strip()
    if override:
        return override
    return shutil.which("ffmpeg") or "ffmpeg"


def ffmpeg_available() -> bool:
    """``True`` if a usable ffmpeg binary is reachable.

    Not cached — install-after-startup must reflect immediately so the
    frontend dialog can dismiss without a backend restart. Cost is ~one
    PATH walk per call; only ``/health`` and ``/system/extras`` hit it.
    """
    override = os.environ.get(PODCODEX_FFMPEG_EXE_ENV, "").strip()
    if override:
        return os.path.isfile(override)
    return shutil.which("ffmpeg") is not None


def ffmpeg_override_dir() -> str | None:
    """Parent dir of ``$PODCODEX_FFMPEG_EXE`` if set to a real file, else None.

    Used at startup to prepend the override dir to PATH so libraries that
    hard-code bare ``"ffmpeg"`` (whisperx, faster-whisper) resolve to the
    same binary that :func:`ffmpeg_exe` returns.
    """
    override = os.environ.get(PODCODEX_FFMPEG_EXE_ENV, "").strip()
    if not override:
        return None
    p = Path(override)
    return str(p.parent) if p.is_file() else None
