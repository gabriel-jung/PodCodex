"""
podcodex.ingest.show — Show-level metadata stored as ``show.toml``.

Each show folder may contain a ``show.toml`` with the canonical show name,
an optional RSS feed URL, a speaker roster, and a primary language.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SHOW_META_FILENAME = "show.toml"


@dataclass
class ShowMeta:
    """Show-level metadata persisted in ``show.toml``."""

    name: str
    rss_url: str = ""
    speakers: list[str] = field(default_factory=list)
    language: str = ""


def load_show_meta(show_folder: Path) -> ShowMeta | None:
    """Read ``show.toml`` from *show_folder*. Returns None if the file is missing."""
    path = Path(show_folder) / SHOW_META_FILENAME
    if not path.exists():
        return None
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return ShowMeta(
        name=raw.get("name", ""),
        rss_url=raw.get("rss_url", ""),
        speakers=raw.get("speakers", []),
        language=raw.get("language", ""),
    )


def _toml_string(s: str) -> str:
    """Escape a string for TOML double-quoted format."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def save_show_meta(show_folder: Path, meta: ShowMeta) -> Path:
    """Write ``show.toml`` to *show_folder*. Creates the directory if needed."""
    folder = Path(show_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SHOW_META_FILENAME

    lines: list[str] = [f'name = "{_toml_string(meta.name)}"']
    if meta.rss_url:
        lines.append(f'rss_url = "{_toml_string(meta.rss_url)}"')
    if meta.language:
        lines.append(f'language = "{_toml_string(meta.language)}"')
    if meta.speakers:
        items = ", ".join(f'"{_toml_string(s)}"' for s in meta.speakers)
        lines.append(f"speakers = [{items}]")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
