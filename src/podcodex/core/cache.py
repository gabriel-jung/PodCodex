"""
podcodex.core.cache — Model cache management.

Centralises all ML model downloads into a PodCodex-controlled directory
instead of scattering across ~/.cache/huggingface/.

Default location: ``~/.podcodex/models/``
Override via ``PODCODEX_CACHE_DIR`` env var.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def get_cache_dir() -> Path:
    """Return (and create) the PodCodex model cache directory."""
    path = Path(
        os.environ.get("PODCODEX_CACHE_DIR", str(Path.home() / ".podcodex" / "models"))
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_hf_cache_dir() -> Path:
    """Return the HuggingFace-style cache dir inside the PodCodex cache.

    Also sets ``HF_HOME`` so libraries that don't accept an explicit
    ``cache_dir`` parameter still download into PodCodex's controlled
    directory.  This covers pyannote (diarization) and BGEM3 (embeddings)
    which only respect the env var.  Call this function early in any
    pipeline path that loads models.
    """
    hf_dir = get_cache_dir() / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_dir))
    return hf_dir


def list_cached_models() -> list[dict]:
    """List models currently in the cache with disk usage info."""
    hf_hub = get_hf_cache_dir() / "hub"
    if not hf_hub.exists():
        return []

    models: list[dict] = []
    for entry in sorted(hf_hub.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
        # Extract a readable model name from the directory name
        # HF hub dirs look like "models--org--name"
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
    hf_hub = get_hf_cache_dir() / "hub"
    target = hf_hub / model_id
    if target.exists() and target.is_dir() and target.parent == hf_hub:
        shutil.rmtree(target)
        return True
    return False


def get_vram_status() -> dict | None:
    """Return GPU VRAM info if torch + CUDA available, else None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
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
