"""Structural guards for two documented single-facility rules (CLAUDE.md).

* GPU detection and dtypes go through ``core/device.py``: no direct
  ``torch.cuda.is_available()`` or hard-coded half-precision dtypes elsewhere.
* ``PipelineDB`` shares one SQLite connection across FastAPI's threadpool, so
  every statement runs under ``self._lock`` (reads through ``_read``). A read
  outside the lock intermittently returned nothing when another thread
  committed mid-statement.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "podcodex"


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


# bootstrap.py maps dtype *names* to torch dtypes for a transformers patch;
# it picks nothing itself.
_DEVICE_ALLOWED = {"core/device.py", "bootstrap.py"}
_FORBIDDEN = {"torch.cuda.is_available", "torch.bfloat16", "torch.float16"}


def test_gpu_detection_and_dtypes_only_in_the_device_facility():
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in _DEVICE_ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and _dotted(node) in _FORBIDDEN:
                offenders.append(f"{rel}:{node.lineno} {_dotted(node)}")
            if (
                isinstance(node, ast.keyword)
                and node.arg == "compute_type"
                and isinstance(node.value, ast.Constant)
            ):
                offenders.append(f"{rel}:{node.value.lineno} compute_type literal")
    assert offenders == [], "use core/device.py instead:\n" + "\n".join(offenders)


def _under_lock(parents: list[ast.AST]) -> bool:
    for node in parents:
        if isinstance(node, ast.With):
            for item in node.items:
                if _dotted(item.context_expr) == "self._lock":
                    return True
    return False


def test_every_pipeline_db_statement_holds_the_lock():
    tree = ast.parse((SRC / "core" / "pipeline_db.py").read_text(encoding="utf-8"))
    (cls,) = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PipelineDB"
    ]
    # __init__ and migrations run before the instance is shared; _upsert is
    # only called by mark / mark_bulk, which hold the lock around it.
    exempt = {"__init__", "_run_migrations", "_upsert"}
    offenders = []

    def visit(node: ast.AST, parents: list[ast.AST], method: str) -> None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"execute", "executemany", "executescript"}
            and _dotted(node.func.value) == "self._conn"
            and method not in exempt
            and not _under_lock(parents)
        ):
            offenders.append(f"{method}:{node.lineno}")
        for child in ast.iter_child_nodes(node):
            visit(child, [*parents, node], method)

    for item in cls.body:
        if isinstance(item, ast.FunctionDef):
            visit(item, [], item.name)
    assert offenders == [], (
        "run these under `with self._lock` or via _read:\n" + "\n".join(offenders)
    )
