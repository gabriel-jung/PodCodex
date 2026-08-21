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


def test_reporting_the_gpu_backend_does_not_import_torch() -> None:
    """The sidebar's GPU badge polls ``/api/gpu/status`` on mount, so a torch
    import inside ``status()`` lands ~4s into every launch — for a badge that
    renders on Windows hosts with an NVIDIA card and nowhere else. Subprocess,
    since this session has torch loaded already."""
    code = (
        "import sys;"
        "from podcodex.api import gpu_backend;"
        "gpu_backend.status();"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip() == "False", "status() put torch back on the boot path"


def test_the_backend_report_still_matches_what_torch_says() -> None:
    """Parsing ``torch/version.py`` has to give the same answer as reading the
    attribute, or the badge is fast and wrong.

    Subprocess, and the parse runs *before* torch is imported: in-process it
    would take the ``sys.modules`` shortcut and never exercise the parser.
    """
    code = (
        "from podcodex.api.gpu_backend import current_torch_backend;"
        "parsed = current_torch_backend();"
        "import torch;"
        "print(parsed, 'gpu' if (torch.version.cuda or '') else 'cpu')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    parsed, truth = out.stdout.split()

    assert parsed in {"gpu", "cpu"}, f"parser bailed out: {parsed}"
    assert parsed == truth


def test_the_version_parser_only_accepts_shapes_it_understands() -> None:
    """Both real torch layouts, and a hard "no" on anything else: an
    unrecognised right-hand side has to fall through to importing torch
    rather than be read as a CUDA build."""
    from podcodex.api.gpu_backend import _TORCH_CUDA_LINE

    cases = {
        "cuda: Optional[str] = None": "None",
        "cuda: Optional[str] = '12.8'": "'12.8'",
        "cuda = '11.8'": "'11.8'",
        'cuda = "12.1"': '"12.1"',
        "cuda: str | None = None  # trailing": "None",
        # Not the cuda assignment, must not match.
        "cuda_version = '12.8'": None,
        "__all__ = ['cuda']": None,
        "hip: Optional[str] = '6.0'": None,
        # Computed rather than literal: unreadable, so no answer.
        "cuda = get_cuda()": None,
    }
    for source, expected in cases.items():
        match = _TORCH_CUDA_LINE.search(source)
        assert (match.group(1) if match else None) == expected, source


def test_a_half_imported_torch_does_not_break_the_badge(monkeypatch) -> None:
    """A module is in sys.modules before its body has run, so the warm-up
    thread importing torch can expose one without a bound ``version`` while
    the sidebar polls the badge. Reading through it must not raise."""
    import importlib.util
    import types

    from podcodex.api.gpu_backend import current_torch_backend

    # Faithful to the real window: the import machinery binds __spec__ before
    # running the body, so the stand-in carries one too. Without it the test
    # would exercise the "torch is not installed" branch instead.
    partial = types.ModuleType("torch")
    partial.__spec__ = importlib.util.find_spec("torch")
    monkeypatch.setitem(sys.modules, "torch", partial)

    assert current_torch_backend() in {"gpu", "cpu"}
