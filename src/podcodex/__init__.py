"""Podcodex: Podcast transcription and intelligence."""

import logging
import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)


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
