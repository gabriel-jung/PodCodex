"""Guard the four version declarations that must move together.

WiX skips file replacement when the MSI version is unchanged, so a release
built from a tree where only one of these files was bumped ships a silently
broken upgrade. ``make bump`` keeps them in sync, but a hand edit to a single
file passes pre-commit (check-toml is syntax only), CI and the rest of the
suite. This test is the only thing that notices.

Declarations:

- ``pyproject.toml``            ``version = "X.Y.Z"``
- ``src-tauri/Cargo.toml``      ``version = "X.Y.Z"``
- ``src-tauri/Cargo.lock``      the ``podcodex-app`` package entry
- ``uv.lock``                   the ``podcodex`` package entry
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PYPROJECT = ROOT / "pyproject.toml"
CARGO_TOML = ROOT / "src-tauri" / "Cargo.toml"
CARGO_LOCK = ROOT / "src-tauri" / "Cargo.lock"
UV_LOCK = ROOT / "uv.lock"

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _top_level_version(path: Path) -> str:
    """Read the single top-level ``version = "X.Y.Z"`` line of a TOML file."""
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert matches, f"no version declaration in {path.name}"
    # The first match is the [project] / [package] one; both files declare
    # theirs before any dependency table.
    return matches[0]


def _locked_version(path: Path, package: str) -> str:
    """Read the ``version`` of one package entry in a Cargo/uv lockfile."""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf'^name = "{re.escape(package)}"\nversion = "([^"]+)"$',
        text,
        flags=re.MULTILINE,
    )
    assert match, f"no {package!r} package entry in {path.name}"
    return match.group(1)


@pytest.fixture(scope="module")
def declared_versions() -> dict[str, str]:
    return {
        "pyproject.toml": _top_level_version(PYPROJECT),
        "src-tauri/Cargo.toml": _top_level_version(CARGO_TOML),
        "src-tauri/Cargo.lock": _locked_version(CARGO_LOCK, "podcodex-app"),
        "uv.lock": _locked_version(UV_LOCK, "podcodex"),
    }


def test_every_declaration_is_a_plain_semver(declared_versions):
    for name, version in declared_versions.items():
        assert SEMVER_RE.match(version), f"{name} declares {version!r}, want X.Y.Z"


def test_all_four_declarations_agree(declared_versions):
    """Run ``make bump VERSION=X.Y.Z`` rather than editing one file by hand."""
    unique = set(declared_versions.values())
    assert len(unique) == 1, f"version drift: {declared_versions}"


def test_installed_metadata_matches_pyproject(declared_versions):
    """``__version__`` derives from package metadata, not a hardcoded string."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("podcodex")
    except PackageNotFoundError:  # pragma: no cover - not an editable install
        pytest.skip("podcodex is not installed in this environment")
    assert installed == declared_versions["pyproject.toml"]
