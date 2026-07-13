#!/usr/bin/env python3
"""Bump the app version in every file that must stay in sync.

WiX skips file replacement when the MSI version is unchanged, so a release
built without a bump silently ships a broken upgrade. This script rewrites
both version declarations; ``make bump`` then refreshes the lockfiles.

Rewritten here:
    pyproject.toml           version = "X.Y.Z"
    src-tauri/Cargo.toml     version = "X.Y.Z"

Refreshed by ``make bump`` afterwards:
    src-tauri/Cargo.lock     (cargo update --workspace)
    uv.lock                  (uv lock)

Usage::

    make bump VERSION=0.2.7
    # or: .venv/bin/python scripts/bump_version.py 0.2.7
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (path, pattern): pattern must match exactly one version declaration.
TARGETS: list[tuple[Path, str]] = [
    (ROOT / "pyproject.toml", r'^version = "(\d+\.\d+\.\d+)"$'),
    (ROOT / "src-tauri" / "Cargo.toml", r'^version = "(\d+\.\d+\.\d+)"$'),
]

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def main() -> None:
    if len(sys.argv) != 2 or not VERSION_RE.match(sys.argv[1]):
        print("usage: bump_version.py X.Y.Z", file=sys.stderr)
        sys.exit(2)
    version = sys.argv[1]

    for path, pattern in TARGETS:
        text = path.read_text(encoding="utf-8")
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if len(matches) != 1:
            print(
                f"ERROR: expected exactly one version line in {path.name}, "
                f"found {len(matches)}",
                file=sys.stderr,
            )
            sys.exit(1)
        new_text = re.sub(
            pattern, f'version = "{version}"', text, count=1, flags=re.MULTILINE
        )
        path.write_text(new_text, encoding="utf-8")
        print(f"{path.relative_to(ROOT)}: {matches[0]} -> {version}")


if __name__ == "__main__":
    main()
