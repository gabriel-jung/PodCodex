"""PyInstaller entrypoint for the bundled desktop server.

Runs uvicorn with the FastAPI app object directly (no string import path,
so the frozen binary doesn't need importlib magic). Sets ML cache env
vars from ``PODCODEX_DATA_DIR`` *before* any torch/transformers import,
which the route modules pull in transitively.

This module is intentionally a no-op for the standard dev workflow:

* ``make dev-api`` / ``podcodex-api`` use ``app.py:main()`` and never touch
  this file.
* If invoked directly without ``PODCODEX_DATA_DIR`` set (e.g. someone runs
  ``python -m podcodex.api.server`` from a checkout), the cache wiring
  short-circuits so the dev's ``~/.cache/huggingface`` etc. are untouched.

The frozen binary always receives ``PODCODEX_DATA_DIR`` from the Tauri
shell, so production behaviour is unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _wire_ml_caches() -> None:
    """Point HF / torch / transformers caches at PODCODEX_DATA_DIR/models.

    The Tauri shell sets ``PODCODEX_DATA_DIR`` to the platform-native app
    data directory before spawning this binary. When unset (someone ran
    server.py directly outside the sidecar context) we leave the standard
    HF/torch caches alone so a dev checkout keeps using ``~/.cache/...``
    and shared model downloads aren't relocated.
    """
    data_dir = os.environ.get("PODCODEX_DATA_DIR")
    if not data_dir:
        return  # dev / direct-invocation path: keep system-default caches.

    models_dir = Path(data_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(models_dir / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(models_dir / "huggingface" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(models_dir / "torch"))
    os.environ.setdefault(
        "TRANSFORMERS_CACHE", str(models_dir / "huggingface" / "transformers")
    )
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME", str(models_dir / "sentence-transformers")
    )


def _redirect_stdio_to_logfile() -> None:
    """Replace stdout/stderr with a log file when running frozen.

    The PyInstaller binary inherits a pipe from Tauri. ``multiprocessing.spawn``
    re-executes the same binary for each child and calls ``_flush_std_streams``
    on the parent right before fork — that flush sees the inherited pipe in a
    broken state under PyInstaller's bootstrap, raising ``BrokenPipeError`` and
    killing every transcription/index/synthesis subprocess.

    Pointing Python's stdio at a real file (and dup2'ing the OS-level FDs so
    the child inherits the same file, not the broken pipe) sidesteps the
    flush failure entirely. The Tauri shell's pipe drain still works for the
    bootloader's own output; ours just lands in ``logs/server.log`` instead.
    """
    if not getattr(sys, "frozen", False):
        return  # dev path: keep tty stdio.

    data_dir = os.environ.get("PODCODEX_DATA_DIR")
    if not data_dir:
        return  # bare frozen run with no data dir: leave stdio alone.

    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "server.log"
    log_fp = open(log_path, "a", buffering=1, encoding="utf-8")

    # Reassign Python-level streams so logger/print writes go to the file.
    sys.stdout = log_fp
    sys.stderr = log_fp

    # Also dup2 the OS-level FDs so children inherit a writable file, not the
    # broken pipe — multiprocessing's flush operates on the OS-level FDs.
    try:
        os.dup2(log_fp.fileno(), 1)
        os.dup2(log_fp.fileno(), 2)
    except OSError:
        pass


def _wire_native_binaries() -> None:
    """Expose bundled ffmpeg / yt-dlp paths to libraries that shell out.

    The Rust shell resolves the absolute paths from the Tauri externalBin
    layout and passes them via env. ``IMAGEIO_FFMPEG_EXE`` is honoured by
    imageio-ffmpeg (whisperx, librosa); yt-dlp lives at ``YT_DLP_BINARY``
    and is invoked by subprocess instead of the bundled Python lib so it
    can be hot-swapped between releases.

    Both binaries' parent directories are also prepended to ``PATH`` so any
    library that resolves them via shutil.which keeps working.
    """
    extra_path: list[str] = []

    ffmpeg = os.environ.get("FFMPEG_BINARY")
    if ffmpeg and Path(ffmpeg).exists():
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg)
        extra_path.append(str(Path(ffmpeg).parent))

    ytdlp = os.environ.get("YT_DLP_BINARY")
    if ytdlp and Path(ytdlp).exists():
        extra_path.append(str(Path(ytdlp).parent))

    if extra_path:
        existing = os.environ.get("PATH", "").split(os.pathsep)
        new_segments = [p for p in extra_path if p not in existing]
        if new_segments:
            os.environ["PATH"] = os.pathsep.join(new_segments + existing)


def main() -> None:
    _wire_ml_caches()
    _redirect_stdio_to_logfile()
    _wire_native_binaries()

    # Imports happen after env wiring so torch/transformers see the right caches.
    import uvicorn

    from podcodex.api.app import app

    host = os.environ.get("PODCODEX_API_HOST", "127.0.0.1")
    port = int(os.environ.get("PODCODEX_API_PORT", "18811"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    # PyInstaller frozen entry — sys.frozen is set when bundled.
    if getattr(sys, "frozen", False):
        # Multiprocessing support for PyInstaller bundles (torch DataLoader, etc.)
        from multiprocessing import freeze_support

        freeze_support()
    main()
