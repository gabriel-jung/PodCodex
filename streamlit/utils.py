"""
Shared utilities for podcodex Streamlit tabs.
"""


def fmt_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS (e.g. 4356 → '1:12:36')."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"
