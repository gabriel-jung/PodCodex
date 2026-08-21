"""Create placeholder sidecars so ``cargo tauri dev`` can compile.

Tauri requires every externalBin file *and* every declared resource path to
exist at build/dev time, even when the backend is started separately
(``PODCODEX_SKIP_BACKEND_SPAWN=1``). Rebuilding the ~1 GB PyInstaller bundle
on every dev iteration is unworkable, so we drop tiny stubs that Tauri can
checksum and copy.

Two shapes, because the two sidecars are bundled differently: ``yt-dlp`` is
an externalBin file (triple-suffixed), while the server is a PyInstaller
--onedir tree bundled as the ``binaries/server/`` resource.

The stubs are never actually run — ``PODCODEX_SKIP_BACKEND_SPAWN`` short-
circuits the spawn path in ``src-tauri/src/lib.rs``, and the frontend talks
directly to the user-spawned ``make dev-api`` uvicorn. If a stub is launched
by accident (misconfigured dev env) it exits with code 1 and a one-line
message so the failure is obvious.

Usage:
    .venv/bin/python scripts/setup_dev_sidecar.py
"""

from __future__ import annotations

import platform
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDECAR_DIR = REPO_ROOT / "src-tauri" / "binaries"

# Bytes; placeholder is < 1 KB. A real PyInstaller binary is hundreds of MB,
# so any file above this is assumed real and kept.
MIN_REAL_BINARY_SIZE = 100_000


def host_target_triple() -> str:
    try:
        out = subprocess.check_output(
            ["rustc", "--print", "host-tuple"], text=True
        ).strip()
        if out:
            return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        return (
            "aarch64-apple-darwin"
            if machine in {"arm64", "aarch64"}
            else "x86_64-apple-darwin"
        )
    if system == "Linux":
        return (
            "aarch64-unknown-linux-gnu"
            if machine in {"arm64", "aarch64"}
            else "x86_64-unknown-linux-gnu"
        )
    if system == "Windows":
        return "x86_64-pc-windows-msvc"
    raise RuntimeError(f"Unsupported host: {system}/{machine}")


def write_unix_stub(dest: Path) -> None:
    """POSIX shell stub. Exits 1 with a clear message if accidentally launched."""
    dest.write_text(
        "#!/bin/sh\n"
        "echo 'podcodex sidecar dev placeholder: backend should be started "
        "via make dev-api, not spawned by Tauri.' >&2\n"
        "exit 1\n"
    )
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_windows_stub(dest: Path) -> None:
    """Best-effort stub. Tauri's externalBin on Windows wants real .exe; if it
    rejects this, the user falls back to a real ``packaging/build_server.py``
    run. Documented in Makefile."""
    dest.write_bytes(b"MZ" + b"\x00" * 60 + b"PE\x00\x00")


def ensure_stub(dest: Path, triple: str) -> None:
    """Write a placeholder at ``dest`` unless a real build is already there.

    Size is the discriminator: a real PyInstaller output is hundreds of MB,
    a stub is under a kilobyte.
    """
    if dest.exists() and dest.stat().st_size > MIN_REAL_BINARY_SIZE:
        print(
            f"Real binary already present, leaving alone: {dest} "
            f"({dest.stat().st_size / 1024 / 1024:.1f} MB)"
        )
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    if "windows" in triple:
        write_windows_stub(dest)
        print(f"Wrote Windows placeholder: {dest}")
    else:
        write_unix_stub(dest)
        print(f"Wrote dev placeholder: {dest}")


def server_stub_path(triple: str) -> Path:
    """Where the --onedir server tree's executable lives.

    Also creates the ``_internal/`` sibling: Tauri checks that the declared
    ``binaries/server/`` resource path exists, and PyInstaller's real layout
    has the two side by side.
    """
    server_dir = SIDECAR_DIR / "server"
    internal = server_dir / "_internal"
    internal.mkdir(parents=True, exist_ok=True)
    (internal / ".keep").write_text(
        "Placeholder for the PyInstaller --onedir payload.\n"
    )
    return server_dir / (
        "podcodex-server.exe" if "windows" in triple else "podcodex-server"
    )


def main() -> int:
    triple = host_target_triple()
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)

    ensure_stub(server_stub_path(triple), triple)
    suffix = ".exe" if "windows" in triple else ""
    ensure_stub(SIDECAR_DIR / f"yt-dlp-{triple}{suffix}", triple)

    if "windows" in triple:
        print(
            "WARN: Windows placeholders may not satisfy Tauri's PE check. "
            "If `cargo tauri dev` rejects them, run "
            "`python packaging/build_server.py` and "
            "`python scripts/fetch_native_binaries.py` once."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
