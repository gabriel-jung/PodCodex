"""
podcodex.core.cache — Model cache management.

Centralises all ML model downloads into a PodCodex-controlled directory
instead of scattering across ~/.cache/huggingface/.

Resolution order:
  1. ``PODCODEX_CACHE_DIR`` env var (explicit override)
  2. ``PODCODEX_DATA_DIR/models`` (Tauri shell sets PODCODEX_DATA_DIR)
  3. ``~/.podcodex/models`` (dev fallback)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def get_cache_dir() -> Path:
    """Return (and create) the PodCodex model cache directory.

    The bundled sidecar inherits ``PODCODEX_DATA_DIR`` from the Tauri
    shell pointing at the OS-native app data dir (e.g.
    ``%APPDATA%\\podcodex`` on Windows). Without this fall-through,
    models would land under ``~/.podcodex/`` even though Tauri set
    ``HF_HOME`` to ``<data_dir>/models/huggingface`` — split-brain
    cache where HF Hub and our code disagree on where models live.
    """
    explicit = os.environ.get("PODCODEX_CACHE_DIR", "").strip()
    if explicit:
        path = Path(explicit)
    else:
        data_dir = os.environ.get("PODCODEX_DATA_DIR", "").strip()
        if data_dir:
            path = Path(data_dir) / "models"
        else:
            path = Path.home() / ".podcodex" / "models"
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
                size = sum(f.stat().st_size for f in blobs_dir.rglob("*") if f.is_file())
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
