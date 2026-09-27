"""The base layers must not import the API package.

``core``, ``rag``, ``ingest``, ``bot``, ``cli`` and ``bundle`` run in
installs that have no fastapi: ``deploy/BOT.md``'s bot+rag+cpu, and every
spawned pipeline step worker. Reaching into ``podcodex.api.routes`` for a
domain helper pulls the whole route surface with it (``routes/__init__``
imports all nineteen modules eagerly), so ``podcodex-reindex`` (documented in
``deploy/SMOKE.md``) dies on ``ModuleNotFoundError: fastapi``.

A subprocess, not an in-process import: pytest has already imported the API
for other suites, so ``sys.modules`` here proves nothing.
"""

from __future__ import annotations

import subprocess
import sys

# Every module under these, walked in the child: a hand-picked list let new
# modules (and all of ingest and bot) through unchecked.
BASE_PACKAGES = [
    "podcodex.core",
    "podcodex.rag",
    "podcodex.ingest",
    "podcodex.bot",
    "podcodex.cli",
    "podcodex.bundle",
]

_SNIPPET = """
import importlib, pkgutil, sys
for name in {names!r}:
    pkg = importlib.import_module(name)
    for info in pkgutil.walk_packages(pkg.__path__, name + "."):
        importlib.import_module(info.name)
leaked = sorted(m for m in sys.modules if m.startswith("podcodex.api"))
if leaked:
    raise SystemExit("imported the API package: " + ", ".join(leaked))
"""


def test_base_layers_do_not_import_the_api_package():
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(names=BASE_PACKAGES)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_reindex_runs_without_fastapi():
    """The entry point `deploy/SMOKE.md` documents runs on an install that
    has no fastapi."""
    snippet = (
        "import sys;"
        "sys.modules['fastapi'] = None;"
        "import podcodex.rag.reindex as r;"
        "assert r.main"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
