"""
Shared utilities for podcodex Streamlit tabs.
"""

import re


def normalize_path(path: str) -> str:
    """Normalize a user-provided path for use on WSL.

    Handles:
    - Shell-escaped characters (e.g. /mnt/d/My\ Folder\&\ Stuff -> /mnt/d/My Folder& Stuff)
    - Windows paths (e.g. C:\\Users\\gabriel -> /mnt/c/Users/gabriel)
    - Surrounding quotes
    """
    path = path.strip().strip("'\"")

    # Windows path: convert to WSL /mnt/<drive>/...
    win_match = re.match(r"^([A-Za-z]):[/\\](.*)", path)
    if win_match:
        drive = win_match.group(1).lower()
        rest = win_match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    # Unescape shell escape sequences (backslash followed by any character)
    path = re.sub(r"\\(.)", r"\1", path)

    return path


def fmt_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS (e.g. 4356 → '1:12:36')."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"
