"""Build the bundled podcodex-server binary.

Wraps PyInstaller around packaging/podcodex-server.spec (onefile mode), then
copies the resulting single-file executable to
``src-tauri/binaries/podcodex-server-<triple>``. Tauri's externalBin entry
ships it inside the .app. First launch each session pays a 10-30 s extraction
cost as PyInstaller unpacks libs to /tmp; subsequent launches in the same
boot are instant.

Usage:
    .venv/bin/python packaging/build_server.py            # build for current host
    .venv/bin/python packaging/build_server.py --clean    # wipe dist/ and build_work/ first
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGING_DIR = REPO_ROOT / "packaging"
SPEC_FILE = PACKAGING_DIR / "podcodex-server.spec"
DIST_DIR = PACKAGING_DIR / "dist"
WORK_DIR = PACKAGING_DIR / "build_work"
SIDECAR_OUT_DIR = REPO_ROOT / "src-tauri" / "binaries"


def host_target_triple() -> str:
    out = subprocess.check_output(["rustc", "--print", "host-tuple"], text=True).strip()
    if not out:
        raise RuntimeError("rustc --print host-tuple returned empty")
    return out


def is_windows(triple: str) -> bool:
    return "windows" in triple


def run_pyinstaller(clean: bool) -> Path:
    args = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC_FILE),
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(WORK_DIR),
        "--noconfirm",
    ]
    if clean:
        args.append("--clean")

    print(f"$ {' '.join(args)}")
    subprocess.check_call(args, cwd=PACKAGING_DIR)

    # Onefile output: dist/podcodex-server (or .exe on Windows).
    candidates = [DIST_DIR / "podcodex-server", DIST_DIR / "podcodex-server.exe"]
    for c in candidates:
        if c.is_file():
            return c
    raise RuntimeError(f"PyInstaller onefile output not found in {DIST_DIR}")


def copy_to_sidecar(source: Path, triple: str) -> None:
    """Place the built binary at ``src-tauri/binaries/podcodex-server-<triple>``.

    Tauri's externalBin convention requires the ``-<triple>`` suffix in the
    project tree; the .app bundler strips it on copy.
    """
    SIDECAR_OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if is_windows(triple) else ""
    dest = SIDECAR_OUT_DIR / f"podcodex-server-{triple}{suffix}"
    if dest.exists():
        dest.unlink()
    shutil.copy2(source, dest)
    dest.chmod(0o755)
    print(
        f"Sidecar binary written: {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean", action="store_true", help="wipe dist/ and build_work/ first"
    )
    args = parser.parse_args()

    if args.clean:
        for d in (DIST_DIR, WORK_DIR):
            if d.exists():
                shutil.rmtree(d)

    triple = host_target_triple()
    print(f"Target triple: {triple}")

    output = run_pyinstaller(clean=args.clean)
    copy_to_sidecar(output, triple)


if __name__ == "__main__":
    main()
