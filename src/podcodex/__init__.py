"""Podcodex: Podcast transcription and intelligence."""

import logging
import os
import shutil
import sys
from pathlib import Path

from loguru import logger


def _disable_hf_hub_symlinks_on_windows() -> None:
    """Force ``huggingface_hub`` to copy files instead of creating symlinks.

    HF Hub's default cache structure relies on ``os.symlink`` (snapshot →
    blob). On Windows this requires the ``SeCreateSymbolicLinkPrivilege``
    privilege, granted by Developer Mode or admin elevation. Worse, even
    when symlinks are created, ctranslate2's C++ ``fopen`` on the
    ``model.bin`` symlink fails with 'Unable to open file'.

    Patch ``_create_symlink`` (called from ``_hf_hub_download_to_cache_dir``)
    to always copy. Loses dedup across model versions but produces regular
    files every library can open without elevated privileges.

    Linux/macOS: symlinks work fine, this is a no-op.
    """
    if sys.platform != "win32":
        return
    try:
        from huggingface_hub import file_download as _hf_file_download
    except Exception:  # noqa: BLE001 — never crash app startup on this
        return

    def _copy_instead_of_symlink(src, dst, new_blob: bool = False) -> None:
        try:
            if os.path.exists(dst):
                os.remove(dst)
        except OSError:
            pass
        # If src is itself a symlink (rare in HF cache, but defensive), follow it
        # so we copy the actual content, not a chain of links.
        real_src = os.path.realpath(src)
        shutil.copy(real_src, dst)

    _hf_file_download._create_symlink = _copy_instead_of_symlink


_disable_hf_hub_symlinks_on_windows()


def _hide_windows_console_flashes() -> None:
    """Suppress brief cmd.exe flashes from subprocess.Popen on Windows.

    Many libraries we depend on shell out to console-subsystem binaries
    (whisperx -> ffmpeg, faster-whisper, nvidia-smi probes, uv pip).
    Without ``CREATE_NO_WINDOW`` each call flashes a console window for a
    fraction of a second — visible and ugly in a GUI-subsystem app.

    Patch ``subprocess.Popen.__init__`` to OR ``CREATE_NO_WINDOW`` into
    whatever the caller passed for ``creationflags``. Safe everywhere:
    GUI-subsystem children (explorer.exe, our own --noconsole sidecar)
    aren't affected by the flag because they don't allocate consoles
    anyway. Caller-supplied flags are preserved bitwise.

    No-op on macOS/Linux (no Windows console concept).
    """
    if sys.platform != "win32":
        return
    import subprocess

    if not hasattr(subprocess, "CREATE_NO_WINDOW"):
        return  # very old Python — skip silently
    original_init = subprocess.Popen.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["creationflags"] = (
            kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        )
        return original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init


_hide_windows_console_flashes()


def _patch_transformers_inspect_getsource() -> None:
    """Make transformers' docstring decorator survive a no-source bundle.

    ``transformers.utils.doc.get_docstring_indentation_level`` calls
    ``inspect.getsource(func)`` while *defining* a model class (the
    ``@add_start_docstrings_to_model_forward`` decorator runs at import).
    PyInstaller bundles ship ``.pyc`` files but no ``.py`` source, so
    ``inspect.getsource`` raises ``OSError: could not get source code``
    and the entire model module fails to import.

    This bites us via FlagEmbedding's reranker tree (transitively pulled
    in by ``bge-m3`` embeddings) and would also hit the regular API
    server the first time anyone runs a search. Returning 0 is safe —
    the decorator only uses it to re-indent the docstring, and we never
    read those docstrings at runtime.
    """
    try:
        from transformers.utils import doc as _doc

        _orig = _doc.get_docstring_indentation_level

        def _safe_get_indent(func):
            try:
                return _orig(func)
            except OSError:
                return 0

        _doc.get_docstring_indentation_level = _safe_get_indent
    except Exception:  # noqa: BLE001 — never crash app startup on this
        pass


_patch_transformers_inspect_getsource()


_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<level>{message}</level>"
)

logger.remove()

# Pick exactly ONE primary sink for this process:
#
#   MCP stdio mode (--mcp from Claude Desktop) → stderr only.
#       Stdout is the JSON-RPC channel — must stay clean. Claude Desktop
#       captures stderr and shows it in its log viewer, so a file sink
#       would just duplicate that.
#
#   Bundled sidecar (Tauri spawn, frozen + PODCODEX_DATA_DIR) → file only.
#       server.py has already redirected sys.stderr to logs/server.log
#       before this import, so an stderr sink would write every record
#       to that same file twice.
#
#   Dev (uvicorn from .venv)                                   → stderr only.
_is_mcp_mode = bool(os.environ.get("PODCODEX_MCP_MODE"))
_use_file_sink = bool(
    not _is_mcp_mode
    and getattr(sys, "frozen", False)
    and os.environ.get("PODCODEX_DATA_DIR")
)

if not _use_file_sink and sys.stderr is not None:
    try:
        logger.add(sys.stderr, format=_LOG_FORMAT, level="INFO")
    except Exception:  # noqa: BLE001 — log setup must never crash import
        pass

if _use_file_sink:
    try:
        _log_path = Path(os.environ["PODCODEX_DATA_DIR"]) / "logs" / "server.log"
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(_log_path),
            format=_LOG_FORMAT,
            level="DEBUG",
            rotation="10 MB",
            retention=3,
            enqueue=True,  # thread-safe; survives uvicorn's worker pool
            backtrace=True,
            diagnose=False,  # don't include local vars in tracebacks (size + privacy)
        )
        # Patch raw sys.stdout/stderr if missing (Windows --noconsole spawn
        # children get None for both). Any library that calls sys.stdout.write
        # directly — torch.hub progress, tqdm, faster-whisper — would crash
        # with AttributeError otherwise. Route to the same file as loguru;
        # interleaving is fine for diagnostic output.
        if sys.stdout is None or sys.stderr is None:
            _stdio_fp = open(_log_path, "a", buffering=1, encoding="utf-8")
            if sys.stdout is None:
                sys.stdout = _stdio_fp
            if sys.stderr is None:
                sys.stderr = _stdio_fp
        logger.info(
            "sidecar logging started — pid={}, frozen=True, data_dir={}",
            os.getpid(),
            os.environ["PODCODEX_DATA_DIR"],
        )
    except Exception as exc:  # noqa: BLE001 — must never crash app startup
        # Last-resort: write a single line to a fallback path so we have *some*
        # evidence of init failure. If this also fails, give up silently.
        try:
            fallback = Path.home() / "podcodex-startup-error.log"
            fallback.write_text(
                f"Failed to set up sidecar log file: {exc!r}\n", encoding="utf-8"
            )
        except Exception:
            pass


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru for a unified format."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)

for _name in (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "fastapi",
    "mcp",
    "FastMCP",
    "mcp.server.streamable_http_manager",
    "mcp.server.lowlevel",
):
    _lg = logging.getLogger(_name)
    _lg.handlers = [_InterceptHandler()]
    _lg.propagate = False

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("podcodex")
except PackageNotFoundError:  # editable install before metadata is registered
    __version__ = "0.0.0+unknown"
