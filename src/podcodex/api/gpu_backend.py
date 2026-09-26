"""GPU backend service — detect NVIDIA, download/install/activate the
optional CUDA bundle on hosts that have a supported GPU.

Layout (archives from ``packaging/package_gpu.py``):

    <data_dir>/backends/gpu/
        activated               : empty marker; presence = use this backend
        current                 : name of the install folder to run
        v-<app>-<id>/           : one complete install
            podcodex-server-gpu : the GPU sidecar binary
            _internal/          : PyInstaller onedir tree
                torch/lib/torch_cuda.dll   <- torch-runtime part
                torch/lib/cudnn64_9.dll    <- cuda-libs part
            cuda-libs.json      : the release manifest it was installed from,
                                  plus the files each part laid down

All parts extract into ONE folder so the binary's RPATH
(``$ORIGIN/_internal/torch/lib`` etc.) reunites with its libs.

An install is built in a staging folder beside the live one and switched to
by rewriting ``current``, so a failed, cancelled or killed install leaves the
working one untouched, and an update never writes over files a running GPU
sidecar has loaded (Windows refuses that). Parts whose tag did not change are
hard-linked from the live install instead of downloaded. The switch takes
effect on the next launch, the only time backends switch anyway.

Before ``current`` existed the install lived flat in ``backends/gpu/``
itself; that layout is still read (no ``current`` file) until the next
install replaces it.

The Tauri shell reads ``activated`` and ``current`` on startup to decide
which sidecar to spawn (``locate_gpu_sidecar`` in ``src-tauri/src/lib.rs``).

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
import uuid
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
    """Root of the GPU backend: install folders, ``current``, ``activated``."""
    return data_dir() / "backends" / "gpu"


_MANIFEST_NAME = "cuda-libs.json"
_POINTER_NAME = "current"
_ACTIVATED_NAME = "activated"
_INSTALL_PREFIX = "v-"


def _valid_install_name(name: str) -> bool:
    """Same rule as ``valid_install_name`` in ``src-tauri/src/lib.rs``."""
    return (
        name.startswith(_INSTALL_PREFIX)
        and "/" not in name
        and "\\" not in name
        and ".." not in name
    )


def active_install_dir() -> Path:
    """The folder the launcher runs: the one ``current`` names, else the root
    (the legacy flat layout)."""
    root = gpu_install_dir()
    try:
        name = (root / _POINTER_NAME).read_text(encoding="utf-8").strip()
    except OSError:
        return root
    if _valid_install_name(name) and (root / name).is_dir():
        return root / name
    return root


def _manifest_path() -> Path:
    return active_install_dir() / _MANIFEST_NAME


def _activated_marker() -> Path:
    return gpu_install_dir() / _ACTIVATED_NAME


def _installing_marker() -> Path:
    """Legacy flat layout only: present while archives were being extracted
    over the live tree, which a kill part way left mixed. Such a tree counts
    as absent. Current installs never write into the live folder."""
    return gpu_install_dir() / ".installing"


def _legacy_install_interrupted() -> bool:
    return active_install_dir() == gpu_install_dir() and _installing_marker().exists()


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
    """Return the active install's manifest dict, or None if not installed."""
    if _legacy_install_interrupted():
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
    if _legacy_install_interrupted():
        return None
    installed = installed_manifest()
    if _recorded_files(installed, "server-core") is not None:
        # Built by this installer, which probed the extracted core before
        # switching to it, so its recorded tag is that version. Spawning the
        # binary costs 2-4 s on Windows, on every GPU panel status call.
        return _installed_part_tags(installed).get("server-core") or None
    return _probe_server_core_version(active_install_dir())


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
        "install_dir": str(active_install_dir()),
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
) -> list[str]:
    """Extract *archive* into *dest* (which must already exist) and return
    the relative paths of the files it laid down."""
    progress_cb(frac, f"Extracting {label}…")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest, filter="data")  # filter=data avoids tar exploits
        return sorted(m.name for m in tar.getmembers() if m.isfile())


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


@dataclass(frozen=True)
class _Part:
    """One release archive: ``server-core``, ``torch-runtime`` or ``cuda-libs``."""

    name: str
    tag: str
    archive: str
    sha256: str


# Rough download sizes, to split the progress bar before sizes are known.
_PART_WEIGHT = {"server-core": 1, "torch-runtime": 5, "cuda-libs": 12}


def _manifest_parts(manifest: dict) -> list[_Part]:
    """The parts a release manifest ships.

    Schema 2 lists them. The two-archive schema before it is read as a
    server core plus cuda libs, so an install from an older manifest still
    knows what it holds.
    """
    raw = manifest.get("parts")
    if raw is None:
        cuda_archive = manifest.get("archive")
        cuda_sha = manifest.get("sha256")
        cuda_tag = manifest.get("version")
        # server-core.tar.gz becomes the executed sidecar, so its digest is
        # required, not optional.
        server_sha = manifest.get("server_sha256")
        if not cuda_archive or not cuda_sha or not cuda_tag or not server_sha:
            raise RuntimeError(
                "Manifest missing required fields "
                f"(archive, sha256, server_sha256, version): {manifest}"
            )
        return _checked_names(
            [
                _Part(
                    "server-core",
                    manifest.get("server_version") or "",
                    "server-core.tar.gz",
                    server_sha,
                ),
                _Part("cuda-libs", cuda_tag, cuda_archive, cuda_sha),
            ]
        )
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"Manifest has no parts: {manifest}")
    parts: list[_Part] = []
    for entry in raw:
        fields = ("name", "tag", "archive", "sha256")
        if not isinstance(entry, dict) or not all(
            isinstance(entry.get(k), str) and entry.get(k) for k in fields
        ):
            raise RuntimeError(
                f"Manifest part missing a field (name, tag, archive, sha256): {entry}"
            )
        parts.append(_Part(*(entry[k] for k in fields)))
    if "server-core" not in {p.name for p in parts}:
        raise RuntimeError("Manifest has no server-core part")
    return _checked_names(parts)


def _checked_names(parts: list[_Part]) -> list[_Part]:
    """Refuse a part whose name or archive is not a plain file name.

    Both are joined onto the download folder (and the archive onto the
    release URL) before any hash is checked, and the manifest can come from
    an overridden URL or a mirror.
    """
    from podcodex.core._utils import bad_path_component

    for part in parts:
        for value in (part.name, part.archive):
            if bad_path_component(value):
                raise RuntimeError(f"Unsafe name in manifest: {value!r}")
    return parts


def _installed_part_tags(manifest: dict | None) -> dict[str, str]:
    """Tag of each part the active install holds, from its recorded manifest."""
    if manifest is None:
        return {}
    try:
        return {p.name: p.tag for p in _manifest_parts(manifest)}
    except RuntimeError:
        return {}


def download_and_install(
    progress_cb: Callable[[float, str], None],
    manifest_url: str,
) -> dict:
    """Download the parts that changed, build a new install beside the live
    one, then switch ``current`` to it.

    A part is reused (hard-linked from the live install) when its tag
    matches: the server core by ``<binary> --version`` against the app
    version, torch runtime and cuda libs by the content tag in the manifest.
    A typical app update therefore downloads only the server core.

    Designed to be submitted to ``task_manager``: ``progress_cb`` carries a
    ``cancel_event`` attribute set by ``TaskInfo``. Holds the install lock
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
    root = gpu_install_dir()
    root.mkdir(parents=True, exist_ok=True)

    progress_cb(0.0, "Fetching manifest…")
    manifest, manifest_url = _fetch_manifest(manifest_url)
    parts = _manifest_parts(manifest)

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

    current = active_install_dir()
    installed = installed_manifest()
    installed_tags = _installed_part_tags(installed)
    server_current = installed_server_core_version() == target_app_version

    def _reusable(part: _Part) -> bool:
        same = (
            server_current
            if part.name == "server-core"
            else installed_tags.get(part.name) == part.tag
        )
        # Still whole on disk too: a file quarantined by an antivirus or
        # deleted by hand would otherwise fail every later update.
        return same and _part_intact(current, installed, part)

    reused = [p for p in parts if _reusable(p)]
    needed = [p for p in parts if p not in reused]
    if not needed:
        progress_cb(1.0, "Already up to date.")
        return {
            "installed_version": manifest.get("version"),
            "server_version": target_app_version,
            "skipped": True,
        }

    bands = _progress_bands(needed)
    final = root / f"{_INSTALL_PREFIX}{target_app_version}-{uuid.uuid4().hex[:8]}"
    # Beside the live install (same volume), so hard links work and the
    # finished tree moves into place with a rename.
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    scratch = Path(tempfile.mkdtemp(prefix=".dl-", dir=root))
    try:
        # Downloaded parts go in first and reused files only fill the gaps, so
        # nothing is ever moved over a hard link: removing a link to a DLL a
        # running GPU sidecar has loaded fails on Windows.
        files: dict[str, list[str]] = {}
        for part in needed:
            start, end = bands[part.name]
            label = part.name.replace("-", " ")
            logger.info(
                "GPU {}: installed={}, target={} → downloading",
                part.name,
                installed_tags.get(part.name, "<none>"),
                part.tag,
            )
            archive = scratch / part.archive
            _stream_download(
                _resolve_artifact_url(manifest_url, part.archive),
                archive,
                progress_cb,
                cancel_event,
                progress_start=start,
                progress_end=end - 0.02,
                label=label,
            )
            progress_cb(end - 0.01, f"Verifying {label} hash…")
            actual = _sha256(archive)
            if actual != part.sha256:
                raise RuntimeError(
                    f"{part.name} sha256 mismatch: expected "
                    f"{part.sha256[:16]}…, got {actual[:16]}…"
                )
            _raise_if_cancelled(cancel_event)
            # Extracted on its own, then moved in: tarfile overwrites in
            # place, which would write through a hard link into the live
            # install. Moving replaces the link instead.
            unpacked = scratch / f"{part.name}-files"
            unpacked.mkdir()
            extracted_files = _extract_tar_gz(
                archive, unpacked, label, progress_cb, end - 0.01
            )
            archive.unlink()
            if part.name == "server-core":
                # A server core of another version (the launcher will not run
                # it) is refused before it becomes an install.
                extracted = _probe_server_core_version(unpacked)
                if extracted != target_app_version:
                    raise RuntimeError(
                        f"The downloaded server core reports version {extracted}, "
                        f"but this app is {target_app_version}; the current GPU "
                        "backend was left as it is."
                    )
            files[part.name] = extracted_files
            _merge_tree(unpacked, staging)

        if reused:
            progress_cb(0.9, "Reusing the parts that did not change…")
            files.update(
                _carry_over(
                    current,
                    staging,
                    reused,
                    installed,
                    server_replaced=any(p.name == "server-core" for p in needed),
                )
            )

        from podcodex.core._utils import write_json_atomic

        write_json_atomic(
            staging / _MANIFEST_NAME, {**manifest, "installed_files": files}
        )
        _raise_if_cancelled(cancel_event)
        progress_cb(0.95, "Switching to the new install…")
        os.replace(staging, final)
        _write_pointer(final.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)

    _prune_old_installs(keep=final.name)
    progress_cb(1.0, "Installed. Activate (or restart) to run on the GPU.")
    return {
        "installed_version": manifest.get("version"),
        "server_version": target_app_version,
        "downloaded": [p.name for p in needed],
        "reused": [p.name for p in reused],
    }


def _progress_bands(needed: list[_Part]) -> dict[str, tuple[float, float]]:
    """Split 0.02..0.90 of the bar across the parts, by rough download size."""
    total = sum(_PART_WEIGHT.get(p.name, 1) for p in needed)
    bands: dict[str, tuple[float, float]] = {}
    pos = 0.02
    for part in needed:
        width = 0.88 * _PART_WEIGHT.get(part.name, 1) / total
        bands[part.name] = (pos, pos + width)
        pos += width
    return bands


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        from podcodex.api.tasks import TaskCancelled

        raise TaskCancelled("the current GPU backend is unchanged")


# Entries of the root that belong to the layout, never to an install.
def _is_layout_entry(name: str) -> bool:
    return (
        name in (_POINTER_NAME, _ACTIVATED_NAME)
        or name.startswith(".")
        or name.startswith(_INSTALL_PREFIX)
    )


def _carry_over(
    current: Path,
    staging: Path,
    reused: list[_Part],
    installed: dict | None,
    *,
    server_replaced: bool,
) -> dict[str, list[str]]:
    """Hard-link the reused parts' files from the live install into *staging*,
    which already holds the downloaded parts.

    Returns the file list of each carried part. An install from before file
    lists were recorded (the legacy flat layout, or a two-archive manifest)
    has no per-part split on disk, so its whole tree is carried, minus what
    the downloaded parts already laid down. Minus, too, its
    ``_internal/podcodex-X.Y.Z.dist-info`` when a new server core came in:
    the version is in the directory name, so the old one would sit beside
    the new, and ``importlib.metadata`` would report whichever it found
    first (in practice the older), looping the panel on "out of date".
    """
    listed = {p.name: _recorded_files(installed, p.name) for p in reused}
    if all(rels is not None for rels in listed.values()):
        for rels in listed.values():
            for rel in rels:
                _link_file(current / rel, staging / rel)
        return listed

    legacy_root = current == gpu_install_dir()
    for entry in current.iterdir():
        if entry.name == _MANIFEST_NAME or (
            legacy_root and _is_layout_entry(entry.name)
        ):
            continue
        for f in entry.rglob("*") if entry.is_dir() else [entry]:
            if not f.is_file():
                continue
            rel = f.relative_to(current)
            if (staging / rel).exists() or (
                server_replaced
                and rel.parts[:1] == ("_internal",)
                and len(rel.parts) > 1
                and rel.parts[1].startswith("podcodex-")
                and rel.parts[1].endswith(".dist-info")
            ):
                continue
            _link_file(f, staging / rel)
    return {}


def _recorded_files(installed: dict | None, part: str) -> list[str] | None:
    """The files the active install recorded for *part*, or None when it
    recorded none (an install older than file lists)."""
    listed = ((installed or {}).get("installed_files") or {}).get(part)
    if not isinstance(listed, list):
        return None
    return [r for r in listed if isinstance(r, str) and ".." not in r]


def _part_intact(current: Path, installed: dict | None, part: _Part) -> bool:
    """Whether every file the install recorded for *part* is still there.

    True when no file list was recorded (a legacy install): its carry-over
    walks whatever is on disk and cannot miss a file.
    """
    listed = _recorded_files(installed, part.name)
    return listed is None or all((current / r).is_file() for r in listed)


def _link_file(src: Path, dest: Path) -> None:
    """Hard link *src* at *dest*, copying when the volume refuses links."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _write_pointer(name: str) -> None:
    """Point ``current`` at *name*, atomically: the launcher reads it at startup."""
    from podcodex.core._utils import atomic_write

    atomic_write(
        gpu_install_dir() / _POINTER_NAME,
        lambda tmp: tmp.write_text(name, encoding="utf-8"),
        tag="current",
    )


def _running_from(path: Path) -> bool:
    """Whether this process's executable lives under *path* (a GPU sidecar
    running out of that install, whose unloaded files must not be deleted)."""
    try:
        exe_dir = Path(sys.executable).resolve().parent
        return exe_dir == path.resolve() or exe_dir.is_relative_to(path.resolve())
    except OSError:
        return False


def _prune_old_installs(keep: str) -> None:
    """Delete every install but *keep*, except one this process runs from.

    The running GPU sidecar has loaded some of its files and will import
    others later, so its install is only removed by the next install after a
    restart. Anything Windows still refuses (another process holding a file)
    is left for next time too.
    """
    root = gpu_install_dir()
    legacy_running = _running_from(root) and not any(
        _running_from(root / e.name)
        for e in root.iterdir()
        if e.name.startswith(_INSTALL_PREFIX)
    )
    for entry in root.iterdir():
        if entry.name in (keep, _POINTER_NAME, _ACTIVATED_NAME):
            continue
        if entry.name.startswith(_INSTALL_PREFIX):
            if _running_from(entry):
                continue
        elif legacy_running:
            continue  # the flat legacy install is the one running
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError as exc:
            logger.info("Could not remove old GPU install entry {}: {!r}", entry, exc)


def _merge_tree(src: Path, dest: Path) -> None:
    """Move every entry of *src* into *dest*, replacing what is there.

    Replacing unlinks the target before the move, so a hard-linked file
    shared with the live install is detached rather than overwritten.
    Entries *dest* has and *src* lacks (the other parts) stay.
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


# ── Activation ──────────────────────────────────────────────────────────


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
    if _legacy_install_interrupted():
        raise RuntimeError(
            "The GPU backend install did not finish. Download it again first."
        )
    if installed_manifest() is None:
        raise RuntimeError("No GPU backend installed. Call download first.")
    _activated_marker().touch()
    logger.info("GPU backend activated at {}", active_install_dir())


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
