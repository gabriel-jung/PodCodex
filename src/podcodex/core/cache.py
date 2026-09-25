"""
podcodex.core.cache — Model cache management.

Centralises all ML model downloads into a PodCodex-controlled directory
instead of scattering across ~/.cache/huggingface/.

Resolution order:
  1. ``PODCODEX_CACHE_DIR`` env var (explicit override)
  2. ``app_paths.data_dir() / "models"`` (the one definition of the data
     dir: ``PODCODEX_DATA_DIR`` when the Tauri shell sets it, otherwise
     the platform app-data directory)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from podcodex.core.app_paths import data_dir


def get_cache_dir() -> Path:
    """Return (and create) the PodCodex model cache directory.

    Everything but the explicit override goes through
    ``app_paths.data_dir()``, so the models tree sits next to the index
    and the logs no matter who started the process. The bundled sidecar
    inherits ``PODCODEX_DATA_DIR`` from the Tauri shell; the bot, the
    MCP server and dev checkouts get the platform app-data dir instead
    of a separate ``~/.podcodex``. A second definition of the data dir
    here is what used to send the bot's HF cache to a path on no compose
    volume, so BGE-M3 was re-downloaded on every container recreate.
    """
    explicit = os.environ.get("PODCODEX_CACHE_DIR", "").strip()
    path = Path(explicit) if explicit else data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_hf_cache_dir() -> Path:
    """Return (and create) the HuggingFace root inside the PodCodex cache.

    ``HF_HOME`` points here; see :func:`wire_model_caches` for the layout.
    """
    hf_dir = get_cache_dir() / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    return hf_dir


def get_hf_hub_dir() -> Path:
    """The hub snapshot store (``HF_HUB_CACHE``), for loaders taking a cache dir.

    Passing it explicitly holds even in a process whose huggingface_hub was
    imported before the env vars were set (a script, a test).
    """
    return get_hf_cache_dir() / "hub"


def wire_model_caches() -> None:
    """Point every ML cache env var at the PodCodex model cache.

    The single setter for the cache layout, run first by every
    ``bootstrap_for_*()`` so the vars are in place before torch,
    transformers or huggingface_hub is imported (they read the vars once, at
    import). Every process shares one tree, dev and bot included, so the
    in-app model list and delete see every download. Values already in the
    environment win: the Tauri shell presets the hub and torch vars.

    ``HF_HOME`` is ``<hf>``; ``HF_HUB_CACHE`` and ``TRANSFORMERS_CACHE`` are
    both ``<hf>/hub``. ``transformers.utils.hub.cached_file`` and qwen_tts's
    ``from_pretrained`` look up via ``TRANSFORMERS_CACHE`` and fall back to
    ``HF_HOME/transformers`` (a stub dir distinct from where
    ``snapshot_download`` writes), so the hub and transformers vars must
    agree or the loader and downloader halves split-brain.
    """
    models_dir = get_cache_dir()
    hub = str(get_hf_hub_dir())
    os.environ.setdefault("HF_HOME", str(get_hf_cache_dir()))
    os.environ.setdefault("HF_HUB_CACHE", hub)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub)
    os.environ.setdefault("TORCH_HOME", str(models_dir / "torch"))
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME", str(models_dir / "sentence-transformers")
    )

    # Cap HF Hub network calls so a flaky uplink (VPN, captive portal,
    # huggingface.co outage) can't stall a cached-model load past 10s. The
    # default urllib3 read-timeout is unset, so a half-broken TCP can hang
    # the whole pipeline at startup.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")

    # Hard offline opt-in: when the user knows every model is cached, this
    # bypasses the etag round-trip entirely. Maps to both HF Hub and the
    # transformers-side flag because the two libraries gate on different vars.
    if os.environ.get("PODCODEX_HF_OFFLINE", "").strip() in {"1", "true", "yes"}:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def list_cached_models() -> list[dict]:
    """List models currently in the cache with disk usage info.

    Scans both layouts HF Hub uses depending on which entrypoint downloaded:
      - ``<hf_root>/hub/models--<org>--<name>``  (when ``HF_HOME`` is set;
        used by pyannote.audio.Pipeline.from_pretrained, etc.)
      - ``<hf_root>/models--<org>--<name>``      (when ``cache_dir=`` is
        passed directly; used by faster-whisper's WhisperModel)
    """
    hf_root = get_hf_cache_dir()
    candidates = [hf_root / "hub", hf_root]

    models: list[dict] = []
    seen: set[str] = set()
    for parent in candidates:
        if not parent.is_dir():
            continue
        for entry in sorted(parent.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            if not entry.name.startswith("models--"):
                continue
            if entry.name in seen:
                continue
            seen.add(entry.name)
            blobs_dir = entry / "blobs"
            if blobs_dir.is_dir():
                size = sum(
                    f.stat().st_size for f in blobs_dir.rglob("*") if f.is_file()
                )
            else:
                size = sum(
                    f.stat().st_size
                    for f in entry.rglob("*")
                    if f.is_file() and not f.is_symlink()
                )
            name = entry.name.replace("models--", "").replace("--", "/")
            models.append(
                {
                    "id": entry.name,
                    "name": name,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 1),
                    "path": str(entry),
                }
            )
    return models


def delete_cached_model(model_id: str) -> bool:
    """Delete a model from the cache by its directory name. Returns True if deleted.

    *model_id* comes from a request, and ``parent / ".."`` passed the old
    ``target.parent == parent`` check, which would have removed the whole
    HuggingFace cache. Only a single ``models--*`` component is accepted.
    """
    from podcodex.core._utils import bad_path_component

    if bad_path_component(model_id) or not model_id.startswith("models--"):
        return False
    hf_root = get_hf_cache_dir()
    deleted = False
    # Both layouts: a model fetched once through each kind of loader has a
    # copy in each, and removing one left the other on disk.
    for parent in (hf_root / "hub", hf_root):
        target = parent / model_id
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
            deleted = True
    return deleted


def get_vram_status() -> dict | None:
    """GPU VRAM info for the Settings widget, or None without CUDA.

    Runs in the API process, so it reads device properties and this
    process's allocator counters only: probing free memory would create a
    CUDA context and hold VRAM for the server's lifetime. ``free_mb`` is
    therefore total minus what this process reserved, not device-wide.
    """
    from podcodex.core.device import vram_total_bytes

    total = vram_total_bytes()
    if total is None:
        return None
    try:
        import torch

        used = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        return {
            "total_mb": round(total / (1024 * 1024)),
            "used_mb": round(used / (1024 * 1024)),
            "reserved_mb": round(reserved / (1024 * 1024)),
            "free_mb": round((total - reserved) / (1024 * 1024)),
            "device": torch.cuda.get_device_name(0),
        }
    except Exception:  # noqa: BLE001
        return None
