"""PyInstaller entrypoint for the bundled desktop server.

Runs uvicorn with the FastAPI app object directly (no string import path,
so the frozen binary doesn't need importlib magic). Calls
``bootstrap_for_bundled_sidecar``, which wires the ML cache env vars under
the data dir before anything imports torch or transformers, installs the
platform monkey-patches and configures loguru.

``make dev-api`` (``app.py:create_dev_app``) and ``podcodex-api``
(``app.py:main()``) bootstrap with ``bootstrap_for_dev`` and never run this
file's ``main``. The frozen binary receives ``PODCODEX_DATA_DIR`` from the Tauri
shell.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


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
    bootloader's own output; ours lands in ``logs/stdio.log`` (native crash
    output, faulthandler dumps, child tracebacks). Not ``server.log``: loguru
    rotates that one by renaming, which raw descriptors cannot follow.
    """
    if not getattr(sys, "frozen", False):
        return  # dev path: keep tty stdio.

    data_dir = os.environ.get("PODCODEX_DATA_DIR")
    if not data_dir:
        return  # bare frozen run with no data dir: leave stdio alone.

    from podcodex.core.app_paths import stdio_log_path

    log_path = stdio_log_path(data_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncated per launch (never rotated, so this keeps it to one run), then
    # reopened for append: every writer (these fds, inherited by children,
    # and _early_child_log's own handle) must be O_APPEND or one overwrites
    # the others' lines at a stale offset.
    log_path.write_text("", encoding="utf-8")
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
    """Prepend yt-dlp, ffmpeg override, and standard package-manager dirs to PATH.

    Libraries that shell out with bare ``"yt-dlp"`` / ``"ffmpeg"`` need
    these on PATH. ``YT_DLP_BINARY`` (Tauri-injected) lets us hot-swap
    yt-dlp without rebuilding the sidecar; ``PODCODEX_FFMPEG_EXE`` is
    the user-facing override for non-PATH ffmpeg installs.

    Also injects standard install dirs (``/opt/homebrew/bin`` etc.) when
    missing — macOS GUI processes inherit ``launchd``'s minimal PATH and
    don't see brew/MacPorts by default. Tauri's spawn does this on its
    side; we mirror it here so MCP / dev / direct-CLI runs benefit too.
    """
    from podcodex.core._ffmpeg import ffmpeg_override_dir

    extra_path: list[str] = []

    ytdlp = os.environ.get("YT_DLP_BINARY")
    if ytdlp and Path(ytdlp).is_file():
        extra_path.append(str(Path(ytdlp).parent))

    ffmpeg_dir = ffmpeg_override_dir()
    if ffmpeg_dir:
        extra_path.append(ffmpeg_dir)

    if sys.platform == "darwin":
        candidates = ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin")
    elif sys.platform.startswith("linux"):
        candidates = ("/usr/local/bin", "/snap/bin", "/var/lib/flatpak/exports/bin")
    else:
        candidates = ()
    extra_path.extend(p for p in candidates if Path(p).is_dir())

    if extra_path:
        existing = os.environ.get("PATH", "").split(os.pathsep)
        new_segments = [p for p in extra_path if p not in existing]
        if new_segments:
            os.environ["PATH"] = os.pathsep.join(new_segments + existing)


def main() -> None:
    _redirect_stdio_to_logfile()
    _wire_native_binaries()

    # bootstrap installs platform monkey-patches and configures loguru.
    # Must run before importing app (which transitively imports torch,
    # transformers, FlagEmbedding — any of which can trip on the patches'
    # absence).
    from podcodex.bootstrap import bootstrap_for_bundled_sidecar

    bootstrap_for_bundled_sidecar()

    # Off the critical path on purpose. log_ffmpeg_status spawns
    # ``ffmpeg -version`` (3 s timeout) purely to write one startup log
    # line — nothing reads its result, and bootstrap's own
    # _check_system_ffmpeg already set the capability flag with a cheap
    # PATH walk. Running it inline delayed the uvicorn bind by ~0.3 s on
    # every launch. Daemon thread, so it can never hold up shutdown.
    import threading

    from podcodex.core._ffmpeg import log_ffmpeg_status

    threading.Thread(target=log_ffmpeg_status, name="ffmpeg-probe", daemon=True).start()

    import uvicorn

    from podcodex.api.app import app

    # Bind is loopback-only by design and not configurable: the sidecar is a
    # single-user local backend with no auth. A non-loopback bind would turn
    # every unauthenticated route (arbitrary FS read, show delete, GPU install)
    # into a remote primitive, so we don't expose an env knob for it.
    host = "127.0.0.1"
    port = int(os.environ.get("PODCODEX_API_PORT", "18811"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
    )


def _handle_version_flag() -> None:
    """Probe support for the launcher's version-mismatch check.

    The Tauri shell calls ``<binary> --version`` to verify the installed GPU
    sidecar matches the running app version (see lib.rs spawn_backend_if_needed).
    Handled here, before any heavy imports or stdio redirection, so the check
    is cheap (~150 ms on the GPU --onedir bundle) and lands on the original
    stdout the launcher is reading.
    """
    if "--version" in sys.argv[1:]:
        from podcodex import __version__

        print(f"podcodex-server {__version__}")
        sys.exit(0)


def _handle_mcp_flag() -> None:
    """Run the MCP stdio server instead of uvicorn.

    Invoked when Claude Desktop spawns ``podcodex-server.exe --mcp``.
    JSON-RPC flows over the child's real stdin/stdout, so we deliberately
    do NOT call ``_redirect_stdio_to_logfile`` — FastMCP's stdio transport
    reads/writes those pipes directly. Logs go to stderr, captured by
    Claude Desktop and surfaced in its log viewer.

    Auto-resolves ``PODCODEX_DATA_DIR`` so the MCP process shares the
    GUI app's ``models/`` and LanceDB index.
    """
    if "--mcp" not in sys.argv[1:]:
        return

    if not os.environ.get("PODCODEX_DATA_DIR"):
        # app_paths.data_dir() resolves to the platform-native appdata
        # location matching what Tauri's app_data_dir() picks. Importing
        # it triggers podcodex/__init__.py, which is now passive — no
        # side effects, so this is safe to do before bootstrap runs.
        from podcodex.core.app_paths import data_dir

        os.environ["PODCODEX_DATA_DIR"] = str(data_dir())
    # Anything writing to stdout corrupts JSON-RPC. tqdm/HF progress bars
    # are the usual offenders during model load — silence them. Libraries
    # that ignore the env still go to stdout, but cached embedder loads
    # don't trigger downloads, so this covers the common path.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    _wire_native_binaries()

    from podcodex.bootstrap import bootstrap_for_mcp_stdio

    bootstrap_for_mcp_stdio()

    from podcodex.core._ffmpeg import log_ffmpeg_status

    log_ffmpeg_status()

    from podcodex.mcp.server import main as mcp_main

    mcp_main()
    sys.exit(0)


if __name__ == "__main__":
    _handle_version_flag()
    # PyInstaller frozen entry — sys.frozen is set when bundled.
    if getattr(sys, "frozen", False):
        # Multiprocessing support for PyInstaller bundles (torch DataLoader, etc.)
        from multiprocessing import freeze_support

        freeze_support()
    _handle_mcp_flag()
    main()
