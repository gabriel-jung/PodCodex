"""The base layers must not import the API package.

``core``, ``rag``, ``ingest``, ``bot`` and ``cli`` run in installs that have
no fastapi: ``deploy/BOT.md``'s bot+rag+cpu, and every spawned pipeline step
worker. Reaching into ``podcodex.api.routes`` for a domain helper used to
pull the whole route surface with it (``routes/__init__`` imports all
nineteen modules eagerly), which made ``podcodex-reindex`` — documented in
``deploy/SMOKE.md`` — die on ``ModuleNotFoundError: fastapi``.

A subprocess, not an in-process import: pytest has already imported the API
for other suites, so ``sys.modules`` here proves nothing.
"""

from __future__ import annotations

import subprocess
import sys

BASE_PACKAGES = [
    "podcodex.core.provenance",
    "podcodex.core.source",
    "podcodex.core.recovery",
    "podcodex.core.api_keys",
    "podcodex.core.transcribe_job",
    "podcodex.core.synthesize_job",
    "podcodex.core.delete_episode",
    "podcodex.rag.index_job",
    "podcodex.rag.reindex",
    "podcodex.cli.resolve",
]

_SNIPPET = """
import importlib, sys
for name in {names!r}:
    importlib.import_module(name)
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
    """The failure that made this a bug rather than a smell: the entry point
    `deploy/SMOKE.md` documents, on an install that has no fastapi."""
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
