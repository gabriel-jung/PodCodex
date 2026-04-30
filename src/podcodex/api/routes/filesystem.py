"""Filesystem browsing routes — list directories and audio files."""

from __future__ import annotations

import os
import platform
import stat
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from podcodex.api.routes._helpers import AUDIO_EXTS

router = APIRouter()

# Non-audio auxiliary files only; audio has its own delete endpoint.
_DELETABLE_EXTS = {".vtt", ".srt", ".json", ".txt", ".info.json"}

# APFS surfaces system pseudo-volumes under /Volumes that users never browse.
_MAC_SKIP_VOLUMES = frozenset(
    {"Recovery", "Preboot", "Update", "VM", "xarts", "iSCPreboot", "Hardware"}
)

# WSL2 ships internal-only mounts at /mnt/wsl and /mnt/wslg (WSL plumbing
# + WSLg X-server runtime).
_LINUX_SKIP_MOUNTS = frozenset({"wsl", "wslg"})


def _is_hidden_st(name: str, st: os.stat_result | None) -> bool:
    """True for entries Finder/Explorer would hide, given a pre-fetched stat.

    Splits the name+stat-based check out of :func:`_is_hidden` so callers
    that already hold a ``stat_result`` (e.g. the directory listing loop
    that needs ``S_ISDIR``) avoid a second syscall. Covers dot-prefix
    (Unix), macOS ``UF_HIDDEN``, Windows ``FILE_ATTRIBUTE_HIDDEN``.
    """
    if name.startswith("."):
        return True
    if st is None:
        return False
    if hasattr(st, "st_flags") and st.st_flags & getattr(stat, "UF_HIDDEN", 0):
        return True
    if hasattr(st, "st_file_attributes") and st.st_file_attributes & getattr(
        stat, "FILE_ATTRIBUTE_HIDDEN", 0
    ):
        return True
    return False


def _is_hidden(item: Path) -> bool:
    """True for entries Finder/Explorer would hide. Issues one ``lstat``."""
    try:
        return _is_hidden_st(item.name, item.lstat())
    except OSError:
        return item.name.startswith(".")


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
        entries = sorted(target.iterdir(), key=lambda p: p.name.casefold())
    except PermissionError:
        return {
            "path": str(target),
            "parent": str(target.parent),
            "dirs": [],
            "files": [],
            "error": "Permission denied",
        }

    from podcodex.ingest.show import SHOW_META_FILENAME

    for item in entries:
        try:
            try:
                st = item.lstat()
            except OSError:
                continue
            if _is_hidden_st(item.name, st):
                continue
            if stat.S_ISDIR(st.st_mode):
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
            elif (
                show_files
                and stat.S_ISREG(st.st_mode)
                and item.suffix.lower() in ext_filter
            ):
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


@router.get("/drives")
async def list_drives() -> dict:
    """Enumerate filesystem roots for the FolderPicker's quick-access list.

    Returns whatever's appropriate for the host OS:
      - Windows native: probe A:..Z:, return present drive letters.
      - Linux (incl. WSL2): WSL drive bridges under /mnt/<letter> if any,
        plus /media/<user>/<vol> mounts.
      - macOS: entries under /Volumes (one per mounted volume).
    """
    drives: list[dict] = []
    system = platform.system()

    if system == "Windows":
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            root = Path(f"{letter}:\\")
            if root.exists():
                drives.append({"label": f"Drive {letter}", "path": str(root)})
    elif system == "Darwin":
        volumes = Path("/Volumes")
        if volumes.is_dir():
            for v in sorted(volumes.iterdir(), key=lambda p: p.name.casefold()):
                if not v.is_dir():
                    continue
                if _is_hidden(v) or v.name.startswith("com.apple."):
                    continue
                if v.name in _MAC_SKIP_VOLUMES:
                    continue
                # Skip the boot-disk symlink ("Macintosh HD" → "/") — Home
                # already covers that tree. ``os.readlink`` reads only the
                # link target without traversing it, so a stalled volume
                # cannot block the event loop here.
                try:
                    if os.readlink(v) == "/":
                        continue
                except OSError:
                    pass
                drives.append({"label": v.name, "path": str(v)})
    else:
        # Linux + WSL2. /mnt/* covers WSL2 drive bridges (any letter
        # 'c'-'z') plus arbitrary non-letter mounts (/mnt/data, /mnt/nas).
        # /media/<user>/<vol> is the typical udisks/GVfs automount location
        # on systemd desktops. /run/media/<user>/<vol> is the modern variant
        # on Fedora-derived distros.
        mnt = Path("/mnt")
        if mnt.is_dir():
            try:
                for entry in sorted(mnt.iterdir(), key=lambda p: p.name.casefold()):
                    if not entry.is_dir():
                        continue
                    if entry.name in _LINUX_SKIP_MOUNTS:
                        continue
                    if len(entry.name) == 1 and entry.name.isalpha():
                        label = f"Drive {entry.name.upper()}"
                    else:
                        label = entry.name
                    drives.append({"label": label, "path": str(entry)})
            except PermissionError:
                pass
        for media_root in (Path("/media"), Path("/run/media")):
            if not media_root.is_dir():
                continue
            try:
                for user_dir in sorted(
                    media_root.iterdir(), key=lambda p: p.name.casefold()
                ):
                    if not user_dir.is_dir():
                        continue
                    for vol in sorted(
                        user_dir.iterdir(), key=lambda p: p.name.casefold()
                    ):
                        if vol.is_dir():
                            drives.append({"label": vol.name, "path": str(vol)})
            except PermissionError:
                pass

    return {"drives": drives}


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
