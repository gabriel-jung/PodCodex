"""Download yt-dlp static binary for the current host into
``src-tauri/binaries/`` so Tauri's externalBin discovery picks it up.

Run once before ``make bundle``. Re-run to refresh yt-dlp (gets bot-detection
patches every couple weeks).

ffmpeg is no longer fetched here — it ships inside the PyInstaller bundle
via the ``imageio-ffmpeg`` Python package.

Usage:
    python scripts/fetch_native_binaries.py             # current host only
    python scripts/fetch_native_binaries.py --all       # download for every supported triple
"""

from __future__ import annotations

import argparse
import platform
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDECAR_DIR = REPO_ROOT / "src-tauri" / "binaries"

# yt-dlp ships per-platform binaries from its GitHub releases page.
# "latest" tag redirects to the newest stable release.
YTDLP_SOURCES = {
    "aarch64-apple-darwin": "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos",
    "x86_64-apple-darwin": "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos",
    "x86_64-unknown-linux-gnu": "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux",
    "aarch64-unknown-linux-gnu": "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux_aarch64",
    "x86_64-pc-windows-msvc": "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe",
}


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


def is_windows(triple: str) -> bool:
    return "windows" in triple


def download(url: str) -> bytes:
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "podcodex-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def fetch_ytdlp(triple: str) -> None:
    if triple not in YTDLP_SOURCES:
        print(f"  ! no yt-dlp source for {triple}, skipping")
        return
    url = YTDLP_SOURCES[triple]
    print(f"yt-dlp -> {triple}")
    payload = download(url)

    ext = ".exe" if is_windows(triple) else ""
    dest = SIDECAR_DIR / f"yt-dlp-{triple}{ext}"
    dest.write_bytes(payload)
    if not is_windows(triple):
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  wrote {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true", help="download for every supported triple"
    )
    args = parser.parse_args()

    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    triples = list(YTDLP_SOURCES.keys()) if args.all else [host_target_triple()]

    failures: list[str] = []
    for t in triples:
        try:
            fetch_ytdlp(t)
        except Exception as e:
            msg = f"yt-dlp/{t}: {e}"
            print(f"  FAIL {msg}")
            failures.append(msg)

    # In single-host mode the binary is required for `cargo tauri build` —
    # tolerate-and-skip would just defer the failure to the Tauri build step.
    # In --all mode partial success is fine (cross-arch fetches are best-effort).
    if failures and not args.all:
        print(f"\nFailed to fetch {len(failures)} required binary(s):", file=sys.stderr)
        for m in failures:
            print(f"  - {m}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
