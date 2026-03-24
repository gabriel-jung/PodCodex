"""Filesystem browsing routes — list directories and audio files."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter()

_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".flac"}


@router.get("/api/fs/list")
async def list_directory(
    path: str = Query(default="~", description="Directory to list"),
    show_files: bool = Query(
        default=False, description="Include audio files in listing"
    ),
) -> dict:
    """List subdirectories (and optionally audio files) in the given path."""
    target = Path(path).expanduser().resolve()
    if not target.is_dir():
        return {
            "path": str(target),
            "parent": str(target.parent),
            "dirs": [],
            "files": [],
            "error": "Not a directory",
        }

    dirs: list[dict] = []
    files: list[dict] = []
    try:
        for item in sorted(target.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                is_show = (item / "show.toml").exists()
                has_audio = any(
                    f.suffix.lower() in _AUDIO_EXTS
                    for f in item.iterdir()
                    if f.is_file()
                )
                dirs.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "is_show": is_show,
                        "has_audio": has_audio,
                    }
                )
            elif show_files and item.is_file() and item.suffix.lower() in _AUDIO_EXTS:
                files.append(
                    {
                        "name": item.name,
                        "path": str(item),
                    }
                )
    except PermissionError:
        return {
            "path": str(target),
            "parent": str(target.parent),
            "dirs": [],
            "files": [],
            "error": "Permission denied",
        }

    return {
        "path": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "dirs": dirs,
        "files": files,
        "error": None,
    }
