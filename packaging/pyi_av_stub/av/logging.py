"""Stub of ``av.logging`` — see ``av/__init__.py``.

torchvision.io.video calls ``av.logging.set_level(av.logging.ERROR)`` at
module import. Provide just those two so the import chain succeeds.
"""

from __future__ import annotations

ERROR = 16  # AV_LOG_ERROR


def set_level(_level: int | None = None) -> None:
    """No-op."""
