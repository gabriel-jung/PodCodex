"""Tests for the deferred torch / transformers patch hook.

``bootstrap_for_bundled_sidecar`` no longer imports torch (~4 s) and
transformers (~3 s) at startup — nothing in the API needs them until a
search or a pipeline step runs. A ``sys.meta_path`` finder installs the
patches the moment either module actually executes.

The regression these tests exist for: transformers detects torch with a
bare ``importlib.util.find_spec("torch")`` and discards the spec. A hook
that consumed its callback at resolution time armed a trigger that never
fired, and torch stayed unpatched.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from podcodex import bootstrap


@pytest.fixture
def fake_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A throwaway importable module, so tests never load the real torch."""
    name = "podcodex_fake_ml_target"
    (tmp_path / f"{name}.py").write_text("VALUE = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield name
    sys.modules.pop(name, None)


@pytest.fixture
def armed(fake_module: str) -> Iterator[tuple[str, list[str], object]]:
    """Install a finder that fires on ``fake_module`` and records calls."""
    fired: list[str] = []
    finder = bootstrap._DeferredImportFinder()
    finder.register(fake_module, lambda: fired.append(fake_module))
    sys.meta_path.insert(0, finder)
    try:
        yield fake_module, fired, finder
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


def test_probing_for_the_module_does_not_consume_the_callback(
    armed: tuple[str, list[str], object],
) -> None:
    name, fired, _finder = armed

    spec = importlib.util.find_spec(name)

    assert spec is not None
    assert fired == [], "callback fired without the module ever executing"
    assert name not in sys.modules
    # Still armed: a later real import must get the patches.
    assert any(isinstance(f, bootstrap._DeferredImportFinder) for f in sys.meta_path)


def test_importing_the_module_fires_the_callback_once(
    armed: tuple[str, list[str], object],
) -> None:
    name, fired, finder = armed

    importlib.util.find_spec(name)  # probe first, as transformers does
    module = importlib.import_module(name)

    assert module.VALUE == 42
    assert fired == [name]
    assert finder._pending == {}
    # A second import must not re-fire: patches are install-once.
    del sys.modules[name]
    importlib.import_module(name)
    assert fired == [name]


def test_a_failing_callback_does_not_break_the_import(
    fake_module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> None:
        raise RuntimeError("patch exploded")

    finder = bootstrap._DeferredImportFinder()
    finder.register(fake_module, boom)
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module(fake_module)
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)

    # A broken patch must not evict the module: importlib drops any module
    # whose exec raises, which would force a full torch re-import on retry.
    assert module.VALUE == 42
    assert sys.modules[fake_module] is module


def test_arming_falls_back_to_eager_when_torch_is_already_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "torch", object())
    monkeypatch.setattr(bootstrap, "_install_ml_patches", lambda: calls.append("ml"))
    monkeypatch.setattr(
        bootstrap,
        "_install_eager_patches",
        lambda: calls.append("eager"),
    )
    before = list(sys.meta_path)

    bootstrap._arm_ml_patch_hook()

    assert calls == ["ml"], (
        "the ML half must run now, and only that half: the caller already "
        "installed the eager patches"
    )
    assert sys.meta_path == before, "no finder should be installed"


def test_registering_an_already_imported_module_runs_the_callback_now(
    fake_module: str,
) -> None:
    """The hook can only fire on a future import, so a module already in
    sys.modules has to be handled eagerly or the callback is lost. The nltk
    runtime hook depends on this: whether nltk is loaded yet is not
    something a caller should have to know."""
    importlib.import_module(fake_module)
    fired: list[str] = []

    bootstrap.defer_until_imported(fake_module, lambda: fired.append("now"))

    assert fired == ["now"]
