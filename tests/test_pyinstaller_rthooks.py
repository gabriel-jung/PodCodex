"""The two PyInstaller runtime hooks we override, and why.

PyInstaller auto-injects a runtime hook per collected package, and two of
them dominated the sidecar's startup by importing a package just to read
something small from it:

  pyi_rth_nltk        ``import nltk`` (1587 ms) for a data-path insertion
  pyi_rth_setuptools  ``import setuptools`` (136 ms) for a version number

Both are replaced through ``packaging/pyi_hooks/rthooks.dat``: PyInstaller
prepends ``--additional-hooks-dir`` ahead of the entry-point directories and
``_merge_rthooks`` keeps the first definition it finds for a module. Nothing
about that is checked at build time, so these tests are the guard — and the
end state has to stay identical, because both failure modes surface far from
here (inside whisperx alignment, or on a bare ``import distutils``).
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_DIR = REPO_ROOT / "packaging" / "pyi_hooks"


def _hook_path(name: str) -> Path:
    return HOOK_DIR / "rthooks" / name


@pytest.mark.parametrize(
    ("module", "hook_file"),
    [("nltk", "pyi_rth_nltk_lazy.py"), ("setuptools", "pyi_rth_setuptools_lite.py")],
)
def test_rthooks_dat_claims_the_module(module: str, hook_file: str) -> None:
    """PyInstaller reads this file with ``ast.literal_eval``; a typo is
    silent (it logs and moves on), leaving the slow hook in place."""
    mapping = ast.literal_eval((HOOK_DIR / "rthooks.dat").read_text())

    assert mapping.get(module) == [hook_file]
    assert _hook_path(hook_file).is_file(), (
        "PyInstaller asserts the referenced file exists"
    )


def _run_hook(hook_file: str) -> types.ModuleType:
    path = _hook_path(hook_file)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── nltk ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_meta_path():
    """The nltk hook registers a callback on the process-wide
    deferred-import finder. Snapshot and restore so a later test importing
    nltk for real does not trip a stale callback."""
    saved = list(sys.meta_path)
    yield
    sys.meta_path[:] = saved


@pytest.fixture
def meipass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand in for the frozen bundle's _internal directory."""
    (tmp_path / "nltk_data").mkdir()
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    return tmp_path


@pytest.fixture
def stub_nltk(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A stand-in for nltk, so the test never pays the 1.6 s import."""
    nltk = types.ModuleType("nltk")
    nltk.data = types.SimpleNamespace(path=["/preexisting/nltk_data"])
    monkeypatch.setitem(sys.modules, "nltk", nltk)
    return nltk


def test_nltk_hook_prepends_the_bundled_data_dir(
    meipass: Path, stub_nltk: types.ModuleType
) -> None:
    # nltk is already in sys.modules here, so defer_until_imported runs the
    # callback immediately — the same path a launch takes when something
    # upstream imported nltk before the hook registered.
    _run_hook("pyi_rth_nltk_lazy.py")

    assert stub_nltk.data.path[0] == str(meipass / "nltk_data"), (
        "whisperx alignment resolves models through nltk.data.path; the "
        "bundled directory has to come first, as the original hook had it"
    )
    assert "/preexisting/nltk_data" in stub_nltk.data.path, "existing paths kept"


def test_nltk_hook_does_not_import_nltk_when_it_is_absent(
    meipass: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the exercise: registering must not pull nltk in."""
    monkeypatch.delitem(sys.modules, "nltk", raising=False)

    _run_hook("pyi_rth_nltk_lazy.py")

    assert "nltk" not in sys.modules


# ── setuptools ──────────────────────────────────────────────────────────


def test_bundled_setuptools_is_new_enough_for_the_hardcoded_default() -> None:
    """The hook assumes "local", which is only setuptools >= 60's default.

    If the pin ever drops below that, the hook would install a shim that
    version does not want, so fail here rather than in a frozen build.
    """
    import setuptools

    major = int(setuptools.__version__.split(".")[0])

    assert major >= 60, (
        f"setuptools {setuptools.__version__} defaults to 'stdlib', but "
        f"pyi_rth_setuptools_lite.py hardcodes 'local'"
    )


def _run_hook_in_subprocess(tail: str, env: dict | None = None) -> str:
    """Subprocess, because setuptools is already imported in this session."""
    code = (
        f"import sys, runpy;"
        f"runpy.run_path({str(_hook_path('pyi_rth_setuptools_lite.py'))!r});"
        f"{tail}"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        **({"env": env} if env else {}),
    )
    return out.stdout.strip()


def test_setuptools_hook_does_not_import_setuptools() -> None:
    """The whole point: startup installs the shim and pays nothing for it."""
    assert _run_hook_in_subprocess("print('setuptools' in sys.modules)") == "False", (
        "the hook pulled setuptools in, which is the 136 ms it exists to avoid"
    )


def test_distutils_is_importable_after_the_setuptools_hook() -> None:
    """distutils left the stdlib in 3.12, so without the shim this fails.

    Importing it does pull setuptools in — that is the shim resolving the
    module, and it is exactly the cost being deferred: a launch that never
    touches distutils never pays it.
    """
    assert (
        _run_hook_in_subprocess("import distutils; print(distutils.__name__)")
        == "distutils"
    )


def test_env_override_still_suppresses_the_shim() -> None:
    """``SETUPTOOLS_USE_DISTUTILS=stdlib`` opted out before; it still does."""
    out = _run_hook_in_subprocess(
        "print(any(type(f).__name__ == 'DistutilsMetaFinder' for f in sys.meta_path))",
        env={"PATH": "/usr/bin:/bin", "SETUPTOOLS_USE_DISTUTILS": "stdlib"},
    )

    assert out == "False"
