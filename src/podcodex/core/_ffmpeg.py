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
  4. Windows-only fallback: live registry PATH + winget/scoop/chocolatey
     shim dirs + winget ffmpeg package payload dirs. PATH is captured
     per-process at logon so a winget install during a session is
     invisible to ``shutil.which`` until reboot. The payload scan covers
     a half-broken winget state: package present on disk but the Links
     shim missing and the user-PATH entry lost (installers rewriting the
     user PATH are a known cause) — winget still reports "installed"
     while nothing resolves the binary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
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


def _winget_ffmpeg_payload_dirs(pkg_root: Path) -> list[str]:
    """``bin/`` dirs holding ``ffmpeg.exe`` inside winget package payloads.

    Gyan.FFmpeg extracts to ``<pkg_root>/Gyan.FFmpeg[.Full]_<source>/
    ffmpeg-<ver>-full_build/bin/ffmpeg.exe``. Newest version first so a
    leftover older payload never shadows the current one.
    """
    out: list[str] = []
    try:
        for pkg in pkg_root.glob("*FFmpeg*"):
            for exe in pkg.glob("*/bin/ffmpeg.exe"):
                out.append(str(exe.parent))
            if (pkg / "bin" / "ffmpeg.exe").is_file():
                out.append(str(pkg / "bin"))
    except OSError:
        return []
    return sorted(out, reverse=True)


@lru_cache(maxsize=1)
def _windows_extra_dirs() -> tuple[str, ...]:
    """Windows-only: dirs to scan when in-process PATH misses fresh installs.

    PATH is captured per-process at logon. ``winget install Gyan.FFmpeg``
    appends to the user PATH in the registry but the running app
    (and even an in-app restart, which inherits Tauri's PATH) won't see
    it until the user signs out. Read PATH from the registry directly,
    plus the standard winget/scoop/chocolatey shim dirs.

    Cached for the process lifetime: the registry only changes via
    OS-level installs and the user gets a fresh process via Restart.
    Returns a tuple so it stays hashable.
    """
    if sys.platform != "win32":
        return ()
    dirs: list[str] = []
    try:
        import winreg

        keys = (
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        )
        for root, sub in keys:
            try:
                with winreg.OpenKey(root, sub) as k:
                    raw, _ = winreg.QueryValueEx(k, "Path")
                    expanded = os.path.expandvars(raw or "")
                    dirs.extend(p for p in expanded.split(";") if p)
            except OSError:
                continue
    except ImportError:
        pass

    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        winget_root = Path(local) / "Microsoft" / "WinGet"
        dirs.append(str(winget_root / "Links"))
        dirs.extend(_winget_ffmpeg_payload_dirs(winget_root / "Packages"))
    program_files = os.environ.get("ProgramFiles", "")
    if program_files:
        # winget --scope machine payload root
        dirs.extend(
            _winget_ffmpeg_payload_dirs(Path(program_files) / "WinGet" / "Packages")
        )
    user = os.environ.get("USERPROFILE", "")
    if user:
        dirs.append(os.path.join(user, "scoop", "shims"))
    program_data = os.environ.get("ProgramData", "")
    if program_data:
        dirs.append(os.path.join(program_data, "chocolatey", "bin"))
    return tuple(dirs)


def _which_with_fallback() -> str | None:
    """Resolve ffmpeg via PATH; on Windows fall back to registry + shim dirs.

    Side effect: when the fallback resolves a binary whose dir isn't on
    PATH, prepends it so subsequent subprocess spawns (whisperx,
    faster-whisper) hardcoding bare ``"ffmpeg"`` find the same binary.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    extra = _windows_extra_dirs()
    if not extra:
        return None
    found = shutil.which("ffmpeg", path=os.pathsep.join(extra))
    if not found:
        return None
    bin_dir = str(Path(found).parent)
    existing = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in existing:
        os.environ["PATH"] = os.pathsep.join([bin_dir, *existing])
    return found


def ffmpeg_exe() -> str:
    """Absolute path to the ffmpeg binary, or ``"ffmpeg"`` as a last resort.

    Not cached: when the user installs ffmpeg or updates the override and
    clicks re-check, the next call must reflect the new state.
    """
    override = _resolve_override()
    if override:
        return override
    return _which_with_fallback() or "ffmpeg"


def ffmpeg_available() -> bool:
    """``True`` if a usable ffmpeg binary is reachable.

    Cost is ~one PATH walk per call (plus one registry read on Windows
    when PATH misses); only ``/api/health`` and ``/api/system/extras``
    hit it.
    """
    if _resolve_override():
        return True
    return _which_with_fallback() is not None


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
    """Parent dir of the resolved ffmpeg binary, or None.

    Used at startup to prepend the dir to PATH so libraries that hard-code
    bare ``"ffmpeg"`` (whisperx, faster-whisper) resolve to the same
    binary that :func:`ffmpeg_exe` returns. Honours the explicit override
    first, then the Windows registry / shim-dir fallback so a winget
    install discovered post-logon also lands on the worker PATH.
    """
    resolved = _resolve_override() or _which_with_fallback()
    if not resolved:
        return None
    return str(Path(resolved).parent)
