"""Podcodex: Podcast transcription and intelligence."""

import logging
import os
import sys
from pathlib import Path

from loguru import logger

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<level>{message}</level>"
)

logger.remove()
logger.add(sys.stderr, format=_LOG_FORMAT, level="INFO")

# In the bundled sidecar (PyInstaller --onefile/--onedir), sys.stderr is
# unreliable on Windows --noconsole builds (no real console = NUL-backed
# stream). Add a file sink directly when the Tauri shell has provided
# PODCODEX_DATA_DIR so the user can actually see what happened. The dev
# path (uvicorn from .venv) doesn't have PODCODEX_DATA_DIR set so this
# is a no-op there; tty stderr stays the only sink.
if getattr(sys, "frozen", False) and os.environ.get("PODCODEX_DATA_DIR"):
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

__version__ = "0.1.0"
