"""Filesystem browsing routes — list directories and audio files."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from podcodex.api.routes._helpers import AUDIO_EXTS

router = APIRouter()

# Non-audio auxiliary files only; audio has its own delete endpoint.
_DELETABLE_EXTS = {".vtt", ".srt", ".json", ".txt", ".info.json"}


@router.get("/list")
async def list_directory(
    path: str = Query(default="~", description="Directory to list"),
    show_files: bool = Query(
        default=False, description="Include matching files in listing"
    ),
    extensions: str = Query(
        default="",
        description="Comma-separated extensions to include when show_files=true (e.g. 'podcodex'). "
        "Empty = audio files only (default).",
    ),
) -> dict:
    """List subdirectories (and optionally files) in the given path."""
    if extensions.strip():
        ext_filter = {
            f".{e.strip().lstrip('.').lower()}"
            for e in extensions.split(",")
            if e.strip()
        }
    else:
        ext_filter = AUDIO_EXTS
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
        entries = sorted(target.iterdir())
    except PermissionError:
        return {
            "path": str(target),
            "parent": str(target.parent),
            "dirs": [],
            "files": [],
            "error": "Permission denied",
        }

    for item in entries:
        try:
            if item.name.startswith("."):
                continue
            if item.is_dir():
                from podcodex.ingest.show import SHOW_META_FILENAME

                is_show = (item / SHOW_META_FILENAME).exists()
                try:
                    has_audio = any(
                        f.suffix.lower() in AUDIO_EXTS
                        for f in item.iterdir()
                        if f.is_file()
                    )
                except PermissionError:
                    has_audio = False
                dirs.append(
                    {
                        "name": item.name,
                        "path": str(item),
                        "is_show": is_show,
                        "has_audio": has_audio,
                    }
                )
            elif show_files and item.is_file() and item.suffix.lower() in ext_filter:
                files.append(
                    {
                        "name": item.name,
                        "path": str(item),
                    }
                )
        except PermissionError:
            continue

    return {
        "path": str(target),
        "parent": str(target.parent) if target.parent != target else None,
        "dirs": dirs,
        "files": files,
        "error": None,
    }


@router.post("/mkdir")
async def make_directory(
    path: str = Query(..., description="Parent directory"),
    name: str = Query(..., description="New folder name"),
) -> dict:
    """Create a new subdirectory inside the given path."""
    parent = Path(path).expanduser().resolve()
    if not parent.is_dir():
        return {"path": None, "error": "Parent directory does not exist"}
    # Prevent path traversal
    if "/" in name or "\\" in name or name in (".", ".."):
        return {"path": None, "error": "Invalid folder name"}
    target = parent / name
    if target.exists():
        return {"path": str(target), "error": "Folder already exists"}
    try:
        target.mkdir(parents=False)
    except PermissionError:
        return {"path": None, "error": "Permission denied"}
    return {"path": str(target), "error": None}


@router.delete("/file")
async def delete_file(
    path: str = Query(..., description="Absolute path to the file to delete"),
) -> dict:
    """Delete a non-audio auxiliary file (subtitles, metadata, etc).

    Safety: only removes files whose suffix is in a small allow-list, and
    only when the file lives inside a folder that looks like a show folder
    (contains ``show.toml``) to prevent arbitrary filesystem writes.
    """
    from podcodex.ingest.show import SHOW_META_FILENAME

    p = Path(path).expanduser().resolve()
    if p.suffix.lower() not in _DELETABLE_EXTS:
        raise HTTPException(400, f"Refusing to delete file type: {p.suffix}")

    parent = p.parent
    while True:
        if (parent / SHOW_META_FILENAME).exists():
            break
        if parent.parent == parent:
            raise HTTPException(400, "File is not inside a known show folder")
        parent = parent.parent

    try:
        p.unlink()
    except FileNotFoundError as e:
        raise HTTPException(404, f"File not found: {path}") from e
    return {"status": "deleted", "path": str(p)}


@router.post("/open")
async def open_folder(
    path: str = Query(..., description="Folder to open in the OS file manager"),
) -> dict:
    """Open a folder in the OS file manager (Finder, Explorer, etc.)."""
    target = Path(path).expanduser().resolve()
    if not target.is_dir():
        return {"error": "Not a directory"}
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", str(target)])
        elif system == "Windows":
            subprocess.Popen(["explorer", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception as exc:
        return {"error": str(exc)}
    return {"error": None}
