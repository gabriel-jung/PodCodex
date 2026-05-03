"""PyAV stub package — shadows real ``av`` during PyInstaller bundling.

Real PyAV ships GPL-built FFmpeg dylibs (libx264 / libx265); bundling
those contaminates the MIT release. See LICENSE_AUDIT.md.

Selected over real PyAV by ``--paths`` ordering in build_server.py.
Module-level ``__getattr__`` returns a sentinel so deep chains like
``av.video.frame.VideoFrame.pict_type`` resolve at import time without
ever pulling in real PyAV.
"""

from __future__ import annotations

from typing import Any

from . import logging  # noqa: F401 — re-export so ``av.logging`` works

__version__ = "0.0.0+podcodex-stub"


class _Anything:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, item: str) -> "_Anything":
        return _Anything(f"{self._name}.{item}")

    def __call__(self, *_args: Any, **_kwargs: Any) -> "_Anything":
        return _Anything(f"{self._name}()")

    def __iter__(self):
        return iter(())

    def __bool__(self) -> bool:
        return False

    def __int__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"<av-stub {self._name}>"


def __getattr__(name: str) -> Any:
    return _Anything(f"av.{name}")
