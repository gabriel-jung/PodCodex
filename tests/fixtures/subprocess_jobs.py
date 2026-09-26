"""Top-level functions used as subprocess entry points in tests.

Kept module-level so `multiprocessing.spawn` can re-import them.
"""

from __future__ import annotations

import time
from collections.abc import Callable


def add(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    a: int,
    b: int,
) -> int:
    progress_cb(0.0, "start")
    for i in range(3):
        if cancelled():
            return -1
        progress_cb((i + 1) / 3, f"tick {i}")
        time.sleep(0.02)
    progress_cb(1.0, "done")
    return a + b


def boom(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> None:
    raise ValueError("explode")


def slow(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> str:
    for _ in range(200):
        if cancelled():
            return "cancelled"
        time.sleep(0.05)
    return "done"


def big_result(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    size: int,
) -> str:
    return "x" * size


def big_boom(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    size: int,
) -> None:
    raise ValueError("explode " + "y" * size)


def omp_threads(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> str | None:
    import os

    return os.environ.get("OMP_NUM_THREADS")
