"""Model cache management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from podcodex.core.cache import (
    delete_cached_model,
    get_cache_dir,
    get_vram_status,
    list_cached_models,
)

router = APIRouter()


@router.get("")
def get_models():
    """List cached models with disk usage and optional VRAM info."""
    return {
        "models": list_cached_models(),
        "cache_dir": str(get_cache_dir()),
        "vram": get_vram_status(),
    }


@router.delete("/{model_id}")
def remove_model(model_id: str):
    """Delete a cached model by ID."""
    if not delete_cached_model(model_id):
        raise HTTPException(404, "Model not found in cache")
    return {"status": "deleted", "model_id": model_id}
