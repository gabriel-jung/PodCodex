"""Resolve the bundled ffmpeg binary path.

``imageio-ffmpeg`` ships a vendored static ffmpeg in its wheel (Linux,
macOS, Windows). We reuse that one binary across direct subprocess
calls (clip extraction, voice-sample upload conversion) and as the
PATH entry that whisperx / faster-whisper find when they shell out
with a bare ``"ffmpeg"`` command.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Absolute path to the vendored ffmpeg binary.

    Honours ``IMAGEIO_FFMPEG_EXE`` if set so users can point at a
    system ffmpeg for debugging. Falls back to ``imageio_ffmpeg``'s
    bundled binary, then to bare ``"ffmpeg"`` if the package is
    missing — that last branch only matters when somebody runs the
    code without installing the ``pipeline`` extra.
    """
    override = os.environ.get("IMAGEIO_FFMPEG_EXE", "").strip()
    if override:
        return override
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"
