"""GPU backend service — detect NVIDIA, download/install/activate the
optional CUDA bundle on hosts that have a supported GPU.

Bundle layout (mirrors VoiceBox's split, see ``packaging/package_gpu.py``):

    <data_dir>/backends/gpu/
        podcodex-server-gpu     — the GPU sidecar binary
        _internal/              — PyInstaller onedir tree (torch lib, etc.)
            torch/lib/libcudnn.so.9    ← from cuda-libs.tar.gz
            torch/lib/libtorch_cpu.so  ← from server-core.tar.gz
        cuda-libs.json          — manifest copied from the download
        activated               — empty marker; presence = use this backend

Both archives extract to the SAME install dir so the binary's RPATH
(``$ORIGIN/_internal/torch/lib`` etc.) reunites with its libs. Splitting
into subdirs would break runtime resolution.

The Tauri shell (M.5) reads the ``activated`` marker on startup to decide
which sidecar to spawn — bundled CPU sidecar or the extracted GPU server.

Dev mode (uvicorn from .venv): the backend is whatever's in the venv.
This module's status report still works (so the UI can render correctly),
but ``download``/``activate`` raise ``DevModeError`` — gated by the same
guard so the dev-no-tauri pipeline keeps working untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

from loguru import logger

from podcodex.core.app_paths import data_dir, running_in_bundle


class DevModeError(RuntimeError):
    """Raised when an install/activate operation is attempted in dev mode."""


@dataclass(frozen=True)
class GPUInfo:
    name: str
    vram_mb: int


# ── Filesystem layout ───────────────────────────────────────────────────


def gpu_install_dir() -> Path:
    return data_dir() / "backends" / "gpu"


def _manifest_path() -> Path:
    return gpu_install_dir() / "cuda-libs.json"


def _activated_marker() -> Path:
    return gpu_install_dir() / "activated"


# ── Detection ───────────────────────────────────────────────────────────


def detect_nvidia_gpu() -> GPUInfo | None:
    """Run ``nvidia-smi`` to detect the first NVIDIA GPU, if any.

    Returns None when nvidia-smi is missing, no GPU is present, or the
    driver isn't responding. We don't bundle pynvml — shelling out keeps
    the bundle small and matches what users would check at the terminal.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    first_line = result.stdout.strip().splitlines()
    if not first_line:
        return None
    parts = [p.strip() for p in first_line[0].split(",")]
    if len(parts) < 2:
        return None
    try:
        vram_mb = int(parts[1])
    except ValueError:
        return None
    return GPUInfo(name=parts[0], vram_mb=vram_mb)


def current_torch_backend() -> str:
    """Report the active torch backend: ``"gpu"``, ``"cpu"``, or ``"missing"``."""
    try:
        import torch
    except ImportError:
        return "missing"
    return "gpu" if (torch.version.cuda or "") else "cpu"


# ── Install state ───────────────────────────────────────────────────────


def installed_manifest() -> dict | None:
    """Return the installed cuda-libs manifest dict, or None if not installed."""
    p = _manifest_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _gpu_binary_path() -> Path | None:
    """Resolve the installed GPU sidecar binary, .exe-suffixed on Windows."""
    bare = gpu_install_dir() / "podcodex-server-gpu"
    if bare.is_file():
        return bare
    exe = gpu_install_dir() / "podcodex-server-gpu.exe"
    if exe.is_file():
        return exe
    return None


def installed_server_core_version() -> str | None:
    """Run ``<gpu-binary> --version`` to read the installed server-core version.

    Returns None when the binary is missing, --version isn't supported (older
    install pre-M.8), or the subprocess errors out. The launcher uses the
    same probe pattern from Rust — see ``src-tauri/src/lib.rs::probe_sidecar_version``.
    """
    binary = _gpu_binary_path()
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(gpu_install_dir()),
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    if len(parts) < 2:
        return None
    return parts[-1]


def is_gpu_activated() -> bool:
    """True when the activated marker is present AND the install is intact."""
    if not _activated_marker().is_file():
        return False
    return installed_manifest() is not None


def _app_version() -> str:
    from podcodex import __version__

    return __version__


def status() -> dict[str, Any]:
    """Synchronous status report — safe to call from any context, dev or bundle."""
    gpu = detect_nvidia_gpu()
    manifest = installed_manifest()
    return {
        "mode": "bundle" if running_in_bundle() else "dev",
        "current_backend": current_torch_backend(),
        "gpu_detected": gpu is not None,
        "gpu_name": gpu.name if gpu else None,
        "vram_mb": gpu.vram_mb if gpu else None,
        "installed_version": manifest.get("version") if manifest else None,
        "installed_server_version": installed_server_core_version(),
        "app_version": _app_version(),
        "activated": is_gpu_activated(),
        "install_dir": str(gpu_install_dir()),
    }


# ── Download + install ──────────────────────────────────────────────────


def _ensure_bundle_mode() -> None:
    if not running_in_bundle():
        raise DevModeError(
            "GPU backend management is only available in the packaged desktop "
            "app. In dev mode, the pipeline uses whatever torch is installed "
            "in your venv directly."
        )


def _stream_download(
    url: str,
    dest: Path,
    progress_cb: Callable[[float, str], None],
    cancel_event: threading.Event | None,
    progress_start: float,
    progress_end: float,
    label: str,
) -> None:
    """Download *url* to *dest* with chunked progress reporting and
    cooperative cancellation. Raises if the connection fails, the
    transfer is cancelled, or no bytes arrive."""
    progress_cb(progress_start, f"Connecting to {label}…")
    req = urllib.request.Request(url, headers={"User-Agent": "podcodex-gpu/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        chunk_size = 1 << 20  # 1 MiB
        read = 0
        with open(dest, "wb") as f:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Download cancelled")
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total > 0:
                    span = progress_end - progress_start
                    frac = progress_start + span * (read / total)
                    mb_done = read / 1024**2
                    mb_total = total / 1024**2
                    progress_cb(frac, f"{label}: {mb_done:.0f} / {mb_total:.0f} MB")
                else:
                    mb_done = read / 1024**2
                    progress_cb(progress_start, f"{label}: {mb_done:.0f} MB")
    if dest.stat().st_size == 0:
        raise RuntimeError(f"{label} download produced an empty file")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _extract_tar_gz(archive: Path, dest: Path, label: str, progress_cb, frac: float) -> None:
    """Extract *archive* into *dest* (which must already exist). Does not
    wipe the dest — both archives extract into the same install root."""
    progress_cb(frac, f"Extracting {label}…")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")  # filter=data avoids tar exploits


def _resolve_artifact_url(manifest_url: str, archive_name: str) -> str:
    """Resolve an archive name relative to the manifest URL.

    Manifests live alongside their archives at the same release URL, so a
    relative resolution against the manifest's directory is enough.
    """
    base = manifest_url.rsplit("/", 1)[0] + "/"
    return urljoin(base, archive_name)


def _fetch_text(url: str, *, timeout: int = 15) -> str:
    """Fetch a URL and return the body as text."""
    req = urllib.request.Request(url, headers={"User-Agent": "podcodex-gpu/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _try_fetch_sha256(url: str) -> str | None:
    """Fetch a ``<archive>.sha256`` sidecar file. Returns the digest string
    or None if the sidecar isn't published (older release, network glitch)."""
    try:
        text = _fetch_text(url, timeout=10)
    except Exception:
        return None
    return text.strip().split()[0] if text.strip() else None


def download_and_install(
    progress_cb: Callable[[float, str], None],
    manifest_url: str,
) -> dict:
    """Download manifest + archives, sha256-verify, extract, write marker.

    Selectively re-downloads only what's stale:
      - Server core: checked against the running app's __version__ via
        ``<binary> --version``. Re-downloaded on every app release.
      - Cuda-libs: checked against the manifest's ``version`` field
        (e.g. ``cu128-v1``). Re-downloaded only on toolkit / torch major
        bumps.

    Designed to be submitted to ``task_manager`` — ``progress_cb`` carries
    a ``cancel_event`` attribute set by ``TaskInfo``.
    """
    _ensure_bundle_mode()
    if not manifest_url:
        raise ValueError("Manifest URL is empty — set PODCODEX_GPU_MANIFEST_URL or pass via API.")

    cancel_event: threading.Event | None = getattr(progress_cb, "cancel_event", None)
    install_dir = gpu_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)

    progress_cb(0.0, "Fetching manifest…")
    manifest = json.loads(_fetch_text(manifest_url))
    cuda_archive_name = manifest.get("archive")
    cuda_sha = manifest.get("sha256")
    cuda_libs_version = manifest.get("version")
    if not cuda_archive_name or not cuda_sha or not cuda_libs_version:
        raise RuntimeError(
            f"Manifest missing required fields (archive, sha256, version): {manifest}"
        )

    target_app_version = _app_version()
    needs_server = installed_server_core_version() != target_app_version
    installed = installed_manifest()
    needs_libs = installed is None or installed.get("version") != cuda_libs_version

    if not needs_server and not needs_libs:
        progress_cb(1.0, "Already up to date.")
        return {
            "installed_version": cuda_libs_version,
            "server_version": target_app_version,
            "skipped": True,
        }

    # The server-core archive name is stable across releases (only its
    # *contents* change with each app version). The cuda-libs archive name
    # carries the version tag so the URL is unique per toolkit bump.
    server_archive_name = "server-core.tar.gz"
    server_url = _resolve_artifact_url(manifest_url, server_archive_name)
    cuda_url = _resolve_artifact_url(manifest_url, cuda_archive_name)

    # Allocate progress budget proportionally to what we're actually pulling.
    if needs_server and needs_libs:
        server_band = (0.02, 0.15)
        cuda_band = (0.15, 0.85)
    elif needs_server:
        server_band = (0.02, 0.85)
        cuda_band = None
    else:
        server_band = None
        cuda_band = (0.02, 0.85)

    with tempfile.TemporaryDirectory(prefix="podcodex-gpu-dl-") as tmp_str:
        tmp = Path(tmp_str)

        if needs_server:
            assert server_band is not None
            logger.info(
                "GPU server-core: installed={}, target={} → downloading",
                installed_server_core_version(),
                target_app_version,
            )
            server_tar = tmp / server_archive_name
            _stream_download(
                server_url, server_tar, progress_cb, cancel_event,
                progress_start=server_band[0], progress_end=server_band[1] - 0.02,
                label="server core",
            )
            # Verify against sidecar .sha256 if published; warn otherwise.
            expected = _try_fetch_sha256(server_url + ".sha256")
            if expected:
                progress_cb(server_band[1] - 0.01, "Verifying server core hash…")
                actual = _sha256(server_tar)
                if actual != expected:
                    raise RuntimeError(
                        f"server-core sha256 mismatch — expected {expected[:16]}…, got {actual[:16]}…"
                    )
            else:
                logger.warning("server-core.tar.gz.sha256 not published; skipping integrity check")
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Install cancelled before extraction")
            _extract_tar_gz(server_tar, install_dir, "server core", progress_cb, server_band[1])

        if needs_libs:
            assert cuda_band is not None
            logger.info(
                "GPU cuda-libs: installed={}, target={} → downloading",
                installed.get("version") if installed else "<none>",
                cuda_libs_version,
            )
            cuda_tar = tmp / cuda_archive_name
            _stream_download(
                cuda_url, cuda_tar, progress_cb, cancel_event,
                progress_start=cuda_band[0], progress_end=cuda_band[1] - 0.02,
                label="cuda libs",
            )
            progress_cb(cuda_band[1] - 0.01, "Verifying cuda-libs hash…")
            actual = _sha256(cuda_tar)
            if actual != cuda_sha:
                raise RuntimeError(
                    f"cuda-libs sha256 mismatch — expected {cuda_sha[:16]}…, got {actual[:16]}…"
                )
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Install cancelled before extraction")
            _extract_tar_gz(cuda_tar, install_dir, "cuda libs", progress_cb, cuda_band[1])

    _manifest_path().write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    progress_cb(1.0, "Installed. Activate to switch the sidecar to GPU.")
    return {
        "installed_version": cuda_libs_version,
        "server_version": target_app_version,
        "downloaded_server": needs_server,
        "downloaded_libs": needs_libs,
    }


# ── Activation ──────────────────────────────────────────────────────────


def activate() -> None:
    """Mark the GPU backend as the one to spawn on next sidecar restart."""
    _ensure_bundle_mode()
    if installed_manifest() is None:
        raise RuntimeError("No GPU backend installed — call download first.")
    _activated_marker().touch()
    logger.info("GPU backend activated at {}", gpu_install_dir())


def deactivate() -> None:
    """Stop spawning the GPU backend; revert to the bundled CPU sidecar."""
    _ensure_bundle_mode()
    marker = _activated_marker()
    if marker.is_file():
        marker.unlink()
    logger.info("GPU backend deactivated")


def uninstall() -> None:
    """Remove the on-disk install entirely. Idempotent."""
    _ensure_bundle_mode()
    install_dir = gpu_install_dir()
    if install_dir.is_dir():
        shutil.rmtree(install_dir)
    logger.info("GPU backend install removed: {}", install_dir)


# ── Manifest URL discovery ──────────────────────────────────────────────

# GitHub release pattern: ``v<version>/cuda-libs.json``. The CI workflow
# (``.github/workflows/release.yml``) uploads ``cuda-libs.json`` alongside
# the MSI/DMG to the matching release tag. Override with the env var when
# building from a fork or hosting GPU artifacts elsewhere.
# GitHub "latest release" redirect. Resolves to the most recent non-draft
# release tag — works for stable (v0.1.0) and pre-release (v0.1.0-rc.1)
# tags as long as the draft has been published. Pinning to v{__version__}
# would 404 during rc-tag iteration where pyproject says 0.1.0 but the
# release is at v0.1.0-rc.1. The manifest's own torch_compat field is the
# version guard for ABI mismatches, not the URL.
_DEFAULT_LATEST_MANIFEST_URL = (
    "https://github.com/gabriel-jung/PodCodex/releases/latest/download/cuda-libs.json"
)


def default_manifest_url() -> str:
    """Manifest URL the GPU download button hits when no override is set.

    Override with ``PODCODEX_GPU_MANIFEST_URL`` for forks or to point at a
    specific (non-latest) release.
    """
    override = os.environ.get("PODCODEX_GPU_MANIFEST_URL", "").strip()
    return override or _DEFAULT_LATEST_MANIFEST_URL


# Back-compat alias — older route handlers reference the old name.
manifest_url_from_env = default_manifest_url
