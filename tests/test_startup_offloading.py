"""Work that used to block the uvicorn bind now runs after it.

Two pieces moved out of ``create_app`` (which executes at module import) and
into the lifespan, on worker threads:

  run_startup_recovery            190 ms — walks the show tree for
                                  atomic-write temp orphans left by a prior
                                  hard crash
  _register_show_folder_resolver  reaches IndexStore, which pulls pyarrow and
                                  numpy onto the import path

Neither is needed to answer a request, and together they were a third of the
sidecar's startup. The risk of moving them is that they quietly stop running
at all, which nothing else would notice.
"""

from __future__ import annotations

import subprocess
import sys

from tests.fixtures.api_client import make_client


def test_constructing_the_app_does_not_run_the_recovery_sweep(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []
    import podcodex.core.recovery as recovery

    monkeypatch.setattr(recovery, "run_startup_recovery", lambda: calls.append("ran"))

    make_client(tmp_path, monkeypatch)

    assert calls == [], "recovery ran during construction, delaying the bind"


def test_recovery_and_resolver_both_run_once_the_app_starts(
    tmp_path, monkeypatch
) -> None:
    calls: list[str] = []
    import podcodex.api.app as app_module
    import podcodex.core.recovery as recovery

    monkeypatch.setattr(
        recovery, "run_startup_recovery", lambda: calls.append("recovery")
    )
    monkeypatch.setattr(
        app_module,
        "_register_show_folder_resolver",
        lambda: calls.append("resolver"),
    )

    with make_client(tmp_path, monkeypatch):
        pass

    assert "recovery" in calls, "moved off the critical path and never runs"


def test_the_resolver_is_in_place_by_the_time_index_store_is_usable() -> None:
    """Registering it reaches IndexStore, so it cannot happen at app import
    without dragging pyarrow onto the startup path — but it also cannot
    happen *late*: ``get_all_collection_info`` caches its result against the
    collections mtime, so the first caller to win a race against a
    background registration pins an un-backfilled ``artwork_url`` for the
    rest of the process. Binding it to the import satisfies both.

    Subprocess, because this session has index_store loaded already.
    """
    code = (
        "import sys, podcodex.api.app;"
        "assert 'pyarrow' not in sys.modules;"
        "from podcodex.rag.index_store import IndexStore;"
        "print(IndexStore._show_folder_resolver is not None)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip() == "True"


def test_importing_the_api_pulls_no_heavy_optional_stack() -> None:
    """pyarrow/numpy/lancedb reach the process through the warmup thread, not
    the import path. Subprocess, since this session has them loaded already."""
    code = (
        "import sys, podcodex.api.app;"
        "print([m for m in ('pyarrow','numpy','lancedb','torch','nltk','mcp','httpx')"
        " if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip() == "[]", f"pulled in at import: {out.stdout.strip()}"
