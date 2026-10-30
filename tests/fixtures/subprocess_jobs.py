"""Top-level functions used as subprocess entry points in tests.

Kept module-level so `multiprocessing.spawn` can re-import them.
"""

from __future__ import annotations

import time
from collections.abc import Callable


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
    progress_cb(0.0, "running")
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
    progress_cb(0.5, "halfway")
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


def stubborn(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> str:
    """Never checks ``cancelled``, like a Whisper pass mid-file."""
    time.sleep(30)
    return "done"


def raises_on_cancel(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> str:
    """Stops at its own cancel point by raising, like the index job."""
    from podcodex.core.cancel import TaskCancelled

    for _ in range(200):
        if cancelled():
            raise TaskCancelled("2 indexed")
        time.sleep(0.05)
    return "done"


def timed(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> tuple[float, float]:
    """Wall-clock span of the work, to tell overlapping runs apart."""
    start = time.time()
    time.sleep(0.5)
    return start, time.time()


def crashes_on_cancel(
    *,
    progress_cb: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> str:
    """Fails for real while a cancel is pending, like a CUDA OOM."""
    for _ in range(200):
        if cancelled():
            raise MemoryError("CUDA out of memory")
        time.sleep(0.05)
    return "done"
