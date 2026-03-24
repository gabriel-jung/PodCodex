"""Health check endpoint — used by Tauri to know when the backend is ready."""

from __future__ import annotations

import importlib.util

from fastapi import APIRouter

router = APIRouter(tags=["health"])


def _has_extra(package: str) -> bool:
    """Check if an optional dependency is installed."""
    return importlib.util.find_spec(package) is not None


@router.get("/api/health")
async def health() -> dict:
    """Return API status and detected capabilities."""
    return {
        "status": "ok",
        "capabilities": {
            "pipeline": _has_extra("whisperx"),
            "rag": _has_extra("qdrant_client"),
            "bot": _has_extra("discord"),
            "ingest": _has_extra("feedparser"),
        },
    }
