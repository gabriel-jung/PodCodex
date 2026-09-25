"""Every route's subprocess entry point accepts the kwargs the route sends.

Entry functions are keyword-only and the routes build their kwargs dicts by
hand, so a renamed field used to fail only inside the spawned child, after
the task was accepted. `run_in_subprocess` now binds before spawning; this
checks every call site statically, since the pipeline extra (and so the
jobs themselves) never runs in CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from podcodex.api.subprocess_runner import _check_entry_signature

ROUTES = Path(__file__).resolve().parents[1] / "src" / "podcodex" / "api" / "routes"


def _call_sites() -> list[tuple[str, str, list[str]]]:
    sites = []
    for path in sorted(ROUTES.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            entry, kwargs = kw.get("entry_path"), kw.get("kwargs")
            if not isinstance(entry, ast.Constant) or not isinstance(kwargs, ast.Dict):
                continue
            keys = [k.value for k in kwargs.keys if isinstance(k, ast.Constant)]
            sites.append((f"{path.name}:{node.lineno}", entry.value, keys))
    return sites


SITES = _call_sites()


def test_every_entry_point_is_found():
    """Five call sites today; a refactor that hides them from this scan
    (kwargs built elsewhere) should be noticed, not silently skipped."""
    assert len(SITES) >= 5, SITES


@pytest.mark.parametrize("where,entry,keys", SITES, ids=[s[0] for s in SITES])
def test_route_kwargs_bind_to_the_entry(where, entry, keys):
    _check_entry_signature(entry, dict.fromkeys(keys))


def test_a_mismatch_is_refused_before_spawning():
    with pytest.raises(TypeError, match="transcribe_job:run"):
        _check_entry_signature(
            "podcodex.core.transcribe_job:run", {"audio_path": "x", "nope": 1}
        )
