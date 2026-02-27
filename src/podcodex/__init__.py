"""Podcodex: Podcast transcription and intelligence."""

from loguru import logger
import sys

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)

__version__ = "0.1.0"
