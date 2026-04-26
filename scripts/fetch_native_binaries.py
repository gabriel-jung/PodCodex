"""Download ffmpeg + yt-dlp static binaries for the current host into
``src-tauri/binaries/`` so Tauri's externalBin discovery picks them up.

Run once before ``make bundle``. Re-run to refresh yt-dlp (gets bot-detection
patches every couple weeks). ffmpeg rarely needs updating — pin a known-good
version and forget.

Usage:
    python scripts/fetch_native_binaries.py             # current host only
    python scripts/fetch_native_binaries.py --all       # download for every supported triple
    python scripts/fetch_native_binaries.py --ffmpeg-only
    python scripts/fetch_native_binaries.py --yt-dlp-only
"""

from __future__ import annotations

import argparse
import io
import platform
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDECAR_DIR = REPO_ROOT / "src-tauri" / "binaries"

# Pin ffmpeg version. Update by bumping these URLs together.
FFMPEG_SOURCES = {
    "aarch64-apple-darwin": (
        "https://www.osxexperts.net/ffmpeg711arm.zip",
        "ffmpeg",  # path inside the zip
    ),
    "x86_64-apple-darwin": (
        "https://www.osxexperts.net/ffmpeg711intel.zip",
        "ffmpeg",
    ),
    "x86_64-unknown-linux-gnu": (
        "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
        None,  # auto-detect: extract any "ffmpeg" file from the tar
    ),
    "aarch64-unknown-linux-gnu": (
        "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz",
        None,
    ),
    "x86_64-pc-windows-msvc": (
        "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
        None,  # auto-detect: extract bin/ffmpeg.exe
    ),
}

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


def extract_ffmpeg(payload: bytes, archive_url: str, hint: str | None) -> bytes:
    """Pull the ffmpeg binary out of the downloaded archive."""
    if archive_url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = zf.namelist()
            if hint and hint in names:
                return zf.read(hint)
            candidates = [
                n
                for n in names
                if n.endswith("/ffmpeg")
                or n.endswith("/ffmpeg.exe")
                or n in {"ffmpeg", "ffmpeg.exe"}
            ]
            if not candidates:
                raise RuntimeError(f"no ffmpeg binary in zip: {names[:10]}")
            # Prefer bin/ffmpeg.exe on Windows essentials build.
            candidates.sort(key=lambda n: (0 if "/bin/" in n else 1, len(n)))
            return zf.read(candidates[0])
    if archive_url.endswith(".tar.xz") or archive_url.endswith(".tar.gz"):
        mode = "r:xz" if archive_url.endswith(".xz") else "r:gz"
        with tarfile.open(fileobj=io.BytesIO(payload), mode=mode) as tf:
            for member in tf.getmembers():
                if member.isfile() and member.name.rsplit("/", 1)[-1] == "ffmpeg":
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    return f.read()
        raise RuntimeError("no ffmpeg in tarball")
    raise RuntimeError(f"unknown archive format: {archive_url}")


def fetch_ffmpeg(triple: str) -> None:
    if triple not in FFMPEG_SOURCES:
        print(f"  ! no ffmpeg source for {triple}, skipping")
        return
    url, hint = FFMPEG_SOURCES[triple]
    print(f"ffmpeg → {triple}")
    payload = download(url)
    binary = extract_ffmpeg(payload, url, hint)

    ext = ".exe" if is_windows(triple) else ""
    dest = SIDECAR_DIR / f"ffmpeg-{triple}{ext}"
    dest.write_bytes(binary)
    if not is_windows(triple):
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  wrote {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def fetch_ytdlp(triple: str) -> None:
    if triple not in YTDLP_SOURCES:
        print(f"  ! no yt-dlp source for {triple}, skipping")
        return
    url = YTDLP_SOURCES[triple]
    print(f"yt-dlp → {triple}")
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
    parser.add_argument("--ffmpeg-only", action="store_true")
    parser.add_argument("--yt-dlp-only", action="store_true")
    args = parser.parse_args()

    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    triples = list(FFMPEG_SOURCES.keys()) if args.all else [host_target_triple()]

    for t in triples:
        if not args.yt_dlp_only:
            try:
                fetch_ffmpeg(t)
            except Exception as e:
                print(f"  FAIL ffmpeg/{t}: {e}")
        if not args.ffmpeg_only:
            try:
                fetch_ytdlp(t)
            except Exception as e:
                print(f"  FAIL yt-dlp/{t}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
