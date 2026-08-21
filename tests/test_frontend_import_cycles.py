"""Guard the one import cycle in the frontend API layer that bites at runtime.

``frontend/src/api/client.ts`` is a barrel: it re-exports every feature module
(``export * from "./health"`` and friends) while those modules import ``json``
and friends back out of it. That cycle is fine as long as the imported values
are only *called* later, from inside functions.

It stops being fine the moment a re-exported module reads a ``client.ts`` value
at **module scope**. ESM evaluates a module's dependencies before its own body,
so entering through the barrel evaluates the feature module first, and the
binding it reads is still in its temporal dead zone.

``health.ts`` did exactly that with ``BOOT_PATIENT_RETRY`` (spread into
``healthQueryOptions`` at module scope). It never surfaced in CI or in the
packaged app because Rollup hoists and reorders during the production build;
Vite serves native per-module ESM in dev, so it threw there and only there:

    Uncaught ReferenceError: can't access lexical declaration
    'BOOT_PATIENT_RETRY' before initialization

which rendered as a blank page under ``make dev-no-tauri``.

The fix was to move the retry policies into ``api/connection.ts``, a leaf that
imports nothing. These tests pin both halves of that arrangement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND_API = Path(__file__).resolve().parents[1] / "frontend" / "src" / "api"
CLIENT = FRONTEND_API / "client.ts"
CONNECTION = FRONTEND_API / "connection.ts"

# Read at module scope by their consumers, so they must never be reached
# through the barrel.
INIT_TIME_EXPORTS = ("BOOT_PATIENT_RETRY", "CONNECT_RETRY")


def _src(path: Path) -> str:
    assert path.exists(), f"missing {path}"
    return path.read_text(encoding="utf-8")


def _barrel_modules() -> list[Path]:
    """The feature modules ``client.ts`` re-exports, i.e. the cycle members."""
    names = re.findall(r'export \* from "\./([\w-]+)"', _src(CLIENT))
    assert names, "no `export * from` lines found; did client.ts stop being a barrel?"
    return [FRONTEND_API / f"{name}.ts" for name in names]


def test_connection_module_is_a_leaf():
    """It can only be a safe home for these values if it imports nothing."""
    imports = re.findall(r'^\s*import .*? from "([^"]+)";', _src(CONNECTION), re.M)
    assert imports == [], (
        f"api/connection.ts must stay a leaf, but it imports {imports}. "
        "Anything it pulls in joins the barrel cycle and the temporal-dead-zone "
        "error comes back."
    )


@pytest.mark.parametrize("module", _barrel_modules(), ids=lambda p: p.name)
def test_barrel_modules_do_not_take_init_time_values_from_the_barrel(module: Path):
    """A cycle member must import these from the leaf, never from ``./client``."""
    if not module.exists():  # a re-export of a module that lives elsewhere
        pytest.skip(f"{module.name} not found next to client.ts")
    for match in re.finditer(
        r'import\s*\{([^}]*)\}\s*from\s*"\./client";', _src(module)
    ):
        names = {n.strip() for n in match.group(1).split(",") if n.strip()}
        clashes = names.intersection(INIT_TIME_EXPORTS)
        assert not clashes, (
            f"{module.name} imports {sorted(clashes)} from the barrel it is itself "
            'part of. Import from "./connection" instead: if the value is read at '
            "module scope, going through client.ts is a temporal-dead-zone error "
            "under native ESM (dev only, invisible in the Rollup build)."
        )
