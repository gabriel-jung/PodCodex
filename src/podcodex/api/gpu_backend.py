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
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
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


# CI publishes GPU archives for Windows only. Linux is buildable from
# source via ``make dev`` but no MSI/release flow exists yet, so a manifest
# fetch on Linux would resolve to Windows binaries that wouldn't run.
# macOS uses Metal/MPS via the bundled CPU sidecar's torch — no separate
# CUDA install is meaningful there.
_GPU_SUPPORTED_PLATFORMS = ("win32",)


def gpu_supported_on_platform() -> bool:
    """Whether the GPU backend installer can produce a working install on this OS."""
    return sys.platform in _GPU_SUPPORTED_PLATFORMS


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


def _installing_marker() -> Path:
    """Present while archives are being extracted over the install dir.

    Extraction writes into the live tree, so a kill or crash part way leaves
    old and new files mixed. While this marker exists the install counts as
    absent (no manifest, no server version) and the activated marker is
    lifted, so neither the launcher nor a re-download trusts that tree.
    """
    return gpu_install_dir() / ".installing"


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


# ``cuda: Optional[str] = '12.8'`` in torch/version.py, or ``cuda = None``.
# The annotation is absent on older torch, hence the optional group. Only
# those two right-hand sides are accepted: anything else means the layout
# changed, and falling through to the import is better than guessing.
_TORCH_CUDA_LINE = re.compile(
    r"^cuda\s*(?::[^=]*)?=\s*(None|'[^']*'|\"[^\"]*\")\s*(?:#.*)?$",
    re.MULTILINE,
)


def current_torch_backend() -> str:
    """Report the active torch backend: ``"gpu"``, ``"cpu"``, or ``"missing"``.

    Read out of ``torch/version.py`` rather than by importing torch: this is
    called by ``status()``, which the sidebar's GPU badge polls on mount, and
    importing torch there would put ~4s of ML import back on every launch for
    a badge that is usually not even rendered. ``version.py`` is a flat file
    of literals, so parsing it is equivalent to reading the attributes.

    The distribution version cannot answer this: the default PyPI wheel is
    the CUDA build on Linux and Windows and still reports a bare ``2.8.0``.
    """
    # Already paid for elsewhere in this process, so ask it directly. Via
    # getattr: a module lands in sys.modules *before* its body runs, so a
    # concurrent import (the search warm-up thread) can expose a torch whose
    # ``version`` submodule is not bound yet. Falling through to the parse
    # is correct there; raising AttributeError at the badge is not.
    torch = sys.modules.get("torch")
    version = getattr(torch, "version", None)
    if version is not None:
        return "gpu" if (getattr(version, "cuda", None) or "") else "cpu"

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return "missing"
    if spec is None or not spec.origin:
        return "missing"

    # Present on disk in the frozen sidecar too: hooks-contrib's torch hook
    # sets ``module_collection_mode = "pyz+py"``, so the sources land beside
    # the archive copy (the same reason ``inspect.getsource`` works there).
    try:
        source = (Path(spec.origin).parent / "version.py").read_text(encoding="utf-8")
    except OSError:
        source = ""
    if match := _TORCH_CUDA_LINE.search(source):
        return "cpu" if match.group(1) == "None" else "gpu"

    # Unrecognised layout. Correctness wins over the import cost here; the
    # regression test above keeps this from becoming the normal path.
    logger.debug("torch/version.py unreadable, falling back to importing torch")
    import torch

    return "gpu" if (torch.version.cuda or "") else "cpu"


# ── Install state ───────────────────────────────────────────────────────


def installed_manifest() -> dict | None:
    """Return the installed cuda-libs manifest dict, or None if not installed."""
    if _installing_marker().exists():
        return None
    p = _manifest_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _gpu_binary_path(root: Path) -> Path | None:
    """Resolve the GPU sidecar binary under *root*, .exe-suffixed on Windows."""
    bare = root / "podcodex-server-gpu"
    if bare.is_file():
        return bare
    exe = root / "podcodex-server-gpu.exe"
    if exe.is_file():
        return exe
    return None


def installed_server_core_version() -> str | None:
    """Run ``<gpu-binary> --version`` to read the installed server-core version.

    Returns None when the binary is missing, --version isn't supported (older
    install pre-M.8), or the subprocess errors out. The launcher uses the
    same probe pattern from Rust — see ``src-tauri/src/lib.rs::probe_sidecar_version``.
    """
    if _installing_marker().exists():
        return None
    return _probe_server_core_version(gpu_install_dir())


def _probe_server_core_version(root: Path) -> str | None:
    """``--version`` of the GPU binary under *root*, whatever the install state."""
    binary = _gpu_binary_path(root)
    if binary is None:
        return None
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(root),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # OSError covers a binary that cannot be executed (permissions, a
        # wrong-platform file), not just a missing one.
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


def _needs_update(installed_server_version: str | None) -> bool:
    """True when GPU is installed but the server-core version trails the app.

    This is the silent-regression path: user updates the MSI, the
    ``<data_dir>/backends/gpu/`` install survives but its
    ``podcodex-server-gpu --version`` reports the OLD app version. The
    Tauri shell falls back to the bundled CPU sidecar without telling
    anyone — transcription quietly becomes 10× slower until the user
    notices and re-downloads.
    """
    if installed_server_version is None:
        return False
    return installed_server_version != _app_version()


def status() -> dict[str, Any]:
    """Synchronous status report — safe to call from any context, dev or bundle."""
    gpu = detect_nvidia_gpu()
    manifest = installed_manifest()
    # Probing the GPU sidecar's --version spawns a PyInstaller'd Python and
    # is the slowest part of this call (~2-4s on Windows). Compute once and
    # reuse for the needs_update check below.
    server_version = installed_server_core_version()
    return {
        "mode": "bundle" if running_in_bundle() else "dev",
        "current_backend": current_torch_backend(),
        "gpu_detected": gpu is not None,
        "gpu_name": gpu.name if gpu else None,
        "vram_mb": gpu.vram_mb if gpu else None,
        "installed_version": manifest.get("version") if manifest else None,
        "installed_server_version": server_version,
        "app_version": _app_version(),
        "activated": is_gpu_activated(),
        "install_dir": str(gpu_install_dir()),
        "platform_supported": gpu_supported_on_platform(),
        "needs_update": _needs_update(server_version),
    }


# ── Download + install ──────────────────────────────────────────────────


def _ensure_bundle_mode() -> None:
    if not running_in_bundle():
        raise DevModeError(
            "GPU backend management is only available in the packaged desktop "
            "app. In dev mode, the pipeline uses whatever torch is installed "
            "in your venv directly."
        )


def _ensure_platform_supported() -> None:
    if not gpu_supported_on_platform():
        raise RuntimeError(
            f"GPU backend is not available on {sys.platform}. "
            "Only Windows is currently supported. macOS uses the bundled "
            "CPU sidecar (with Apple's MPS via torch); Linux users build "
            "from source via `make dev`."
        )


_DOWNLOAD_ATTEMPTS = 4
_DOWNLOAD_BACKOFF_CAP_S = 30.0


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
    cooperative cancellation. Raises if the connection keeps failing, the
    transfer is cancelled, or no bytes arrive.

    A dropped connection or a stalled read is retried with capped
    exponential backoff, resuming with a ``Range`` request from what is
    already on disk (GitHub release assets honour it), so one blip no longer
    restarts a multi-GB download from zero.
    """
    import http.client
    import urllib.error

    progress_cb(progress_start, f"Connecting to {label}…")
    dest.unlink(missing_ok=True)
    # Consecutive attempts that moved no bytes; one that resumed and made
    # progress resets it, so a long download on a link that drops every few
    # hundred MB still finishes.
    stalled = 0
    while True:
        before = dest.stat().st_size if dest.is_file() else 0
        try:
            _download_once(
                url,
                dest,
                progress_cb,
                cancel_event,
                progress_start,
                progress_end,
                label,
            )
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # nothing left past what we have: complete
                break
            if exc.code < 500:
                raise
            reason: object = exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
        ) as exc:
            reason = exc
        after = dest.stat().st_size if dest.is_file() else 0
        stalled = 0 if after > before else stalled + 1
        if stalled >= _DOWNLOAD_ATTEMPTS:
            raise RuntimeError(
                f"{label} download kept failing ({reason}); "
                "check the connection and try again."
            )
        delay = min(_DOWNLOAD_BACKOFF_CAP_S, 2.0 ** max(stalled, 1))
        logger.warning(
            "{} download interrupted ({}), retrying in {:.0f}s", label, reason, delay
        )
        progress_cb(progress_start, f"{label}: connection lost, retrying…")
        if (cancel_event or threading.Event()).wait(delay):
            raise RuntimeError("Download cancelled")
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"{label} download produced an empty file")


def _download_once(
    url: str,
    dest: Path,
    progress_cb: Callable[[float, str], None],
    cancel_event: threading.Event | None,
    progress_start: float,
    progress_end: float,
    label: str,
) -> None:
    """One transfer, resuming after whatever *dest* already holds."""
    have = dest.stat().st_size if dest.is_file() else 0
    headers = {"User-Agent": "podcodex-gpu/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        resumed = have and resp.status == 206
        if not resumed:
            have = 0  # the server sent the whole file again
        total = int(resp.headers.get("Content-Length") or 0) + have
        chunk_size = 1 << 20  # 1 MiB
        read = have
        with open(dest, "ab" if resumed else "wb") as f:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Download cancelled")
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                mb_done = read / 1024**2
                if total > 0:
                    span = progress_end - progress_start
                    frac = progress_start + span * (read / total)
                    progress_cb(
                        frac, f"{label}: {mb_done:.0f} / {total / 1024**2:.0f} MB"
                    )
                else:
                    progress_cb(progress_start, f"{label}: {mb_done:.0f} MB")
        if total > 0 and read < total:
            # A short body: raised as the retryable error it is.
            import http.client

            raise http.client.IncompleteRead(b"", total - read)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _extract_tar_gz(
    archive: Path, dest: Path, label: str, progress_cb, frac: float
) -> None:
    """Extract *archive* into *dest* (which must already exist). Does not
    wipe the dest — both archives extract into the same install root."""
    progress_cb(frac, f"Extracting {label}…")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")  # filter=data avoids tar exploits


def _purge_stale_dist_info(install_dir: Path) -> None:
    """Remove orphan ``podcodex-*.dist-info/`` directories before re-extracting
    server-core.

    PyInstaller --copy-metadata bakes ``_internal/podcodex-X.Y.Z.dist-info/``
    into the bundle, with the *version* embedded in the directory name. A
    plain tar.extractall over an existing install lays down the new
    dist-info alongside the old one — both survive, since they have
    different names. ``importlib.metadata.version("podcodex")`` then
    iterates ``_internal`` and returns whichever it finds first (in
    practice, the alphabetically earlier — i.e. older — version), so
    ``podcodex-server --version`` keeps reporting the pre-update value
    and the Settings UI loops on "out of date".

    Sweep them out before the new tar lands. Other dist-info dirs (torch,
    transformers, …) don't have the same effect — none of those package
    versions are read back through ``--version`` — so we limit the
    cleanup to the one that actually breaks UX.
    """
    internal = install_dir / "_internal"
    if not internal.is_dir():
        return
    for entry in internal.glob("podcodex-*.dist-info"):
        if entry.is_dir():
            try:
                shutil.rmtree(entry)
                logger.info("Removed stale podcodex dist-info: {}", entry.name)
            except OSError as exc:  # noqa: PERF203 — log per-entry errors
                logger.warning("Could not remove {}: {!r}", entry, exc)


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
    a ``cancel_event`` attribute set by ``TaskInfo``. Holds the install lock
    for the whole run, so activate and uninstall refuse meanwhile.
    """
    with _INSTALL_LOCK:
        return _download_and_install_locked(progress_cb, manifest_url)


def _download_and_install_locked(
    progress_cb: Callable[[float, str], None], manifest_url: str
) -> dict:
    _ensure_bundle_mode()
    _ensure_platform_supported()
    if not manifest_url:
        raise ValueError(
            "Manifest URL is empty. Set PODCODEX_GPU_MANIFEST_URL to override "
            "the default release manifest."
        )

    cancel_event: threading.Event | None = getattr(progress_cb, "cancel_event", None)
    install_dir = gpu_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)

    progress_cb(0.0, "Fetching manifest…")
    manifest, manifest_url = _fetch_manifest(manifest_url)
    cuda_archive_name = manifest.get("archive")
    cuda_sha = manifest.get("sha256")
    cuda_libs_version = manifest.get("version")
    # server-core.tar.gz is the archive that becomes the executed sidecar, so
    # its digest is required, not optional: the hash used to come from a
    # ``.sha256`` sidecar fetch whose 404 (stale mirror, network blip) silently
    # downgraded the check on the one archive that carries code.
    server_sha = manifest.get("server_sha256")
    if not cuda_archive_name or not cuda_sha or not cuda_libs_version or not server_sha:
        raise RuntimeError(
            "Manifest missing required fields "
            f"(archive, sha256, server_sha256, version): {manifest}"
        )

    target_app_version = _app_version()
    # A manifest stamped for another app release carries a server core the
    # launcher refuses (it only runs a GPU sidecar of its own version).
    # Installing it anyway left the app on CPU and every retry re-downloading
    # the same archive; say so instead.
    manifest_server = manifest.get("server_version")
    if manifest_server and manifest_server != target_app_version:
        raise RuntimeError(
            f"The GPU download available is for PodCodex {manifest_server}, "
            f"but this app is {target_app_version}. Update the app, or wait "
            "for this release's GPU build to be published."
        )
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

    was_activated = _activated_marker().is_file()
    # Beside the install dir, not in the system temp dir: the verified server
    # core is moved in from here, which is a rename only on the same volume.
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".gpu-dl-", dir=install_dir.parent
    ) as tmp_str:
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
                server_url,
                server_tar,
                progress_cb,
                cancel_event,
                progress_start=server_band[0],
                progress_end=server_band[1] - 0.02,
                label="server core",
            )
            progress_cb(server_band[1] - 0.01, "Verifying server core hash…")
            actual = _sha256(server_tar)
            if actual != server_sha:
                raise RuntimeError(
                    f"server-core sha256 mismatch: expected {server_sha[:16]}…, got {actual[:16]}…"
                )
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Install cancelled before extraction")
            # Extracted and version-checked in a staging dir first: a server
            # core of another version (the launcher will not run it) is
            # refused before it overwrites a working install.
            staging = tmp / "server-core"
            staging.mkdir()
            _extract_tar_gz(
                server_tar, staging, "server core", progress_cb, server_band[1]
            )
            extracted = _probe_server_core_version(staging)
            if extracted != target_app_version:
                raise RuntimeError(
                    f"The downloaded server core reports version {extracted}, "
                    f"but this app is {target_app_version}; the current GPU "
                    "backend was left as it is."
                )
            _begin_extraction()
            _purge_stale_dist_info(install_dir)
            _merge_tree(staging, install_dir)

        if needs_libs:
            assert cuda_band is not None
            logger.info(
                "GPU cuda-libs: installed={}, target={} → downloading",
                installed.get("version") if installed else "<none>",
                cuda_libs_version,
            )
            cuda_tar = tmp / cuda_archive_name
            _stream_download(
                cuda_url,
                cuda_tar,
                progress_cb,
                cancel_event,
                progress_start=cuda_band[0],
                progress_end=cuda_band[1] - 0.02,
                label="cuda libs",
            )
            progress_cb(cuda_band[1] - 0.01, "Verifying cuda-libs hash…")
            actual = _sha256(cuda_tar)
            if actual != cuda_sha:
                raise RuntimeError(
                    f"cuda-libs sha256 mismatch: expected {cuda_sha[:16]}…, got {actual[:16]}…"
                )
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Install cancelled before extraction")
            _begin_extraction()
            _extract_tar_gz(
                cuda_tar, install_dir, "cuda libs", progress_cb, cuda_band[1]
            )

    # Manifest last and atomic, then the install is trusted again.
    from podcodex.core._utils import write_json_atomic

    write_json_atomic(_manifest_path(), manifest)
    if was_activated:
        _activated_marker().touch()
    _installing_marker().unlink(missing_ok=True)
    progress_cb(1.0, "Installed. Activate to switch the sidecar to GPU.")
    return {
        "installed_version": cuda_libs_version,
        "server_version": target_app_version,
        "downloaded_server": needs_server,
        "downloaded_libs": needs_libs,
    }


# ── Activation ──────────────────────────────────────────────────────────


def _merge_tree(src: Path, dest: Path) -> None:
    """Move every entry of *src* into *dest*, replacing what is there.

    The same result as extracting the archive over *dest*, from a tree that
    was already verified; entries *dest* has and *src* lacks (the cuda libs)
    stay.
    """
    for entry in src.iterdir():
        target = dest / entry.name
        if entry.is_dir() and target.is_dir():
            _merge_tree(entry, target)
            continue
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        shutil.move(str(entry), str(target))


def _begin_extraction() -> None:
    """Mark the tree untrusted (and not launchable) before writing into it."""
    _installing_marker().touch()
    _activated_marker().unlink(missing_ok=True)


# Held by download_and_install for its whole run and taken by activate and
# uninstall without waiting: one guard in the backend for "nothing touches
# the install dir while it is being written", whichever caller.
_INSTALL_LOCK = threading.Lock()


def _refuse_while_installing() -> None:
    if _INSTALL_LOCK.locked():
        raise RuntimeError(
            "The GPU backend is being downloaded; wait for it to finish."
        )


def activate() -> None:
    """Mark the GPU backend as the one to spawn on next sidecar restart."""
    _ensure_bundle_mode()
    _refuse_while_installing()
    if _installing_marker().exists():
        raise RuntimeError(
            "The GPU backend install did not finish. Download it again first."
        )
    if installed_manifest() is None:
        raise RuntimeError("No GPU backend installed. Call download first.")
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
    if not _INSTALL_LOCK.acquire(blocking=False):
        raise RuntimeError(
            "The GPU backend is being downloaded; wait for it to finish."
        )
    try:
        _uninstall_locked()
    finally:
        _INSTALL_LOCK.release()


def _uninstall_locked() -> None:
    install_dir = gpu_install_dir()
    if install_dir.is_dir():
        shutil.rmtree(install_dir)
    logger.info("GPU backend install removed: {}", install_dir)


# ── Manifest URL discovery ──────────────────────────────────────────────

# The CI workflow (``.github/workflows/release.yml``) uploads
# ``cuda-libs.json`` beside the MSI/DMG on each release tag. The app's own
# tag is tried first, because the launcher only runs a GPU server core of the
# app's exact version and "latest" is the newest *stable* release: a beta
# build, or an install one release behind, pulled a server core it then
# refused. A prerelease tag (``vX.Y.Z-beta.N``) is not derivable from the
# version, so "latest" stays as the fallback, and the manifest's
# ``server_version`` (checked in ``download_and_install``) refuses a mismatch
# there with a sentence instead of installing it.
_RELEASES = "https://github.com/gabriel-jung/PodCodex/releases"
_DEFAULT_LATEST_MANIFEST_URL = f"{_RELEASES}/latest/download/cuda-libs.json"


def _tagged_manifest_url() -> str:
    return f"{_RELEASES}/download/v{_app_version()}/cuda-libs.json"


def default_manifest_url() -> str:
    """Manifest URL the GPU download button hits when no override is set.

    Override with ``PODCODEX_GPU_MANIFEST_URL`` for forks or to point at a
    specific release.
    """
    override = os.environ.get("PODCODEX_GPU_MANIFEST_URL", "").strip()
    return override or _tagged_manifest_url()


def _fetch_manifest(manifest_url: str) -> tuple[dict, str]:
    """The manifest and the URL it came from (archives resolve against it).

    Falls back from the app's own tag to "latest" only for the default URL,
    and only on a 404 (no GPU build on that tag); an override is used as is.
    """
    import urllib.error

    try:
        return _load_manifest(manifest_url), manifest_url
    except urllib.error.HTTPError as exc:
        if exc.code != 404 or manifest_url != _tagged_manifest_url():
            raise _manifest_error(exc) from exc
    try:
        return _load_manifest(
            _DEFAULT_LATEST_MANIFEST_URL
        ), _DEFAULT_LATEST_MANIFEST_URL
    except urllib.error.HTTPError as exc:
        raise _manifest_error(exc) from exc


def _manifest_error(exc) -> RuntimeError:
    if exc.code == 404:
        return RuntimeError("No GPU build is published for this release yet.")
    return RuntimeError(f"Could not fetch the GPU manifest: {exc}")


def _load_manifest(url: str) -> dict:
    """Fetch and parse one manifest; HTTP errors propagate as they are."""
    import urllib.error

    try:
        data = json.loads(_fetch_text(url))
    except urllib.error.HTTPError:
        raise
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the GPU download server: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("The GPU manifest is not valid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError("The GPU manifest has an unexpected shape")
    return data
