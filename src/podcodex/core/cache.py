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
    """Return the HuggingFace-style cache dir inside the PodCodex cache.

    Sets ``HF_HOME``, ``HF_HUB_CACHE`` and ``TRANSFORMERS_CACHE`` to the
    same ``<hf_dir>/hub`` location so libraries that don't accept an
    explicit ``cache_dir`` parameter still resolve to PodCodex's
    controlled directory. Pyannote (diarization) and BGEM3 (embeddings)
    only respect ``HF_HOME``; ``transformers.utils.hub.cached_file`` and
    qwen_tts's ``from_pretrained`` look up via ``TRANSFORMERS_CACHE`` and
    fall back to ``HF_HOME/transformers`` (a stub dir distinct from where
    ``snapshot_download`` actually writes), so all three must be aligned
    or the loader/downloader halves split-brain. Call this function
    early in any pipeline path that loads models.
    """
    hf_dir = get_cache_dir() / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    hub_cache = str(hf_dir / "hub")
    os.environ.setdefault("HF_HOME", str(hf_dir))
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hub_cache)
    return hf_dir


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
    """Delete a model from the cache by its directory name. Returns True if deleted."""
    hf_root = get_hf_cache_dir()
    for parent in (hf_root / "hub", hf_root):
        target = parent / model_id
        if target.exists() and target.is_dir() and target.parent == parent:
            shutil.rmtree(target)
            return True
    return False


def get_vram_status() -> dict | None:
    """Return GPU VRAM info if torch + CUDA available, else None."""
    from podcodex.core.device import cuda_available

    if not cuda_available():
        return None
    try:
        import torch

        total = torch.cuda.get_device_properties(0).total_mem
        used = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        return {
            "total_mb": round(total / (1024 * 1024)),
            "used_mb": round(used / (1024 * 1024)),
            "reserved_mb": round(reserved / (1024 * 1024)),
            "free_mb": round((total - reserved) / (1024 * 1024)),
            "device": torch.cuda.get_device_name(0),
        }
    except Exception:
        return None
