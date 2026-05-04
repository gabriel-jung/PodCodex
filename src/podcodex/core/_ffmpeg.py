"""Resolve the system ffmpeg binary path.

PodCodex shells out to ffmpeg for clip extraction, voice-sample upload
conversion, and through whisperx / faster-whisper which hard-code the
bare ``"ffmpeg"`` command. Bundling GPL-built ffmpeg (libx264 / libx265)
would contaminate the MIT release — see LICENSE_AUDIT.md — so we rely on
a system install instead.

Resolution order for an explicit override (highest priority first):
  1. ``$PODCODEX_FFMPEG_EXE`` env var. Tauri injects this at sidecar
     spawn from the persisted setting; CLI users can also export it.
  2. ``ffmpeg_exe_override`` field in ``config.json``. Set via the
     Settings → ffmpeg picker; survives restarts independently of env.
  3. ``shutil.which("ffmpeg")``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from podcodex.core.app_config import load_config, strip_user_path

PODCODEX_FFMPEG_EXE_ENV = "PODCODEX_FFMPEG_EXE"


def _resolve_override() -> str:
    """Return a validated absolute path to the override binary, or ``""``.

    Tries env var, then ``config.json``. Returns ``""`` if neither
    resolves to an existing file, so callers fall back to PATH without
    re-checking the source.
    """
    for raw in (
        os.environ.get(PODCODEX_FFMPEG_EXE_ENV, ""),
        _override_from_config(),
    ):
        candidate = strip_user_path(raw)
        if candidate and Path(candidate).is_file():
            return candidate
    return ""


def _override_from_config() -> str:
    """Read ``ffmpeg_exe_override`` from the persisted app config.

    Failure is swallowed — config corruption shouldn't kill ffmpeg
    resolution; the missing-override warning in :func:`log_ffmpeg_status`
    surfaces it instead.
    """
    try:
        return load_config().ffmpeg_exe_override or ""
    except Exception:
        return ""


def ffmpeg_exe() -> str:
    """Absolute path to the ffmpeg binary, or ``"ffmpeg"`` as a last resort.

    Not cached: when the user installs ffmpeg or updates the override and
    clicks re-check, the next call must reflect the new state.
    """
    override = _resolve_override()
    if override:
        return override
    return shutil.which("ffmpeg") or "ffmpeg"


def ffmpeg_available() -> bool:
    """``True`` if a usable ffmpeg binary is reachable.

    Cost is ~one PATH walk per call; only ``/api/health`` and
    ``/api/system/extras`` hit it.
    """
    if _resolve_override():
        return True
    return shutil.which("ffmpeg") is not None


def probe_ffmpeg(path: str | None = None, timeout: float = 3.0) -> dict:
    """Invoke ``<ffmpeg> -version`` and report whether it's actually callable.

    ``path`` lets the validate-ffmpeg endpoint probe a candidate before
    persisting it; default uses :func:`ffmpeg_exe`. Stronger than
    :func:`ffmpeg_available` — catches a wrong-arch binary, a corrupted
    install, or a stale override pointing at a deleted file.
    """
    target = path if path is not None else ffmpeg_exe()
    try:
        result = subprocess.run(
            [target, "-version"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "path": None, "error": "ffmpeg not found on PATH"}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "path": target, "error": f"{type(exc).__name__}: {exc}"}

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()[:200]
        return {
            "ok": False,
            "path": target,
            "error": f"ffmpeg -version exit {result.returncode}: {stderr}",
        }

    first_line = result.stdout.decode("utf-8", "replace").splitlines()[:1]
    return {
        "ok": True,
        "path": target,
        "version": first_line[0] if first_line else "",
    }


def log_ffmpeg_status() -> None:
    """Run :func:`probe_ffmpeg` and emit a clear startup log line.

    Also surfaces the silent-fallback case: an override is set but the
    file doesn't resolve, so the panel's setting looks accepted but
    pipeline calls actually use system PATH (or fail).
    """
    from loguru import logger

    raw_env = os.environ.get(PODCODEX_FFMPEG_EXE_ENV, "").strip()
    raw_cfg = _override_from_config()
    resolved = _resolve_override()
    if (raw_env or raw_cfg) and not resolved:
        logger.warning(
            "ffmpeg override set but did not resolve to a file (env={!r}, config={!r}). "
            "Falling back to system PATH. Strip surrounding quotes if any.",
            raw_env,
            raw_cfg,
        )

    result = probe_ffmpeg()
    if result["ok"]:
        logger.info(
            "ffmpeg probe: OK ({} at {})", result.get("version", ""), result["path"]
        )
        return
    logger.error(
        "ffmpeg probe: NOT AVAILABLE — pipeline steps will fail. "
        "Install ffmpeg and ensure it is on PATH, or set ${} to its absolute path. "
        "Detail: {}",
        PODCODEX_FFMPEG_EXE_ENV,
        result["error"],
    )


def ffmpeg_override_dir() -> str | None:
    """Parent dir of the resolved override binary, or None.

    Used at startup to prepend the override dir to PATH so libraries that
    hard-code bare ``"ffmpeg"`` (whisperx, faster-whisper) resolve to the
    same binary that :func:`ffmpeg_exe` returns.
    """
    resolved = _resolve_override()
    if not resolved:
        return None
    return str(Path(resolved).parent)
