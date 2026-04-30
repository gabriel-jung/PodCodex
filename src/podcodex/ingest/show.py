"""
podcodex.ingest.show — Show-level metadata stored as ``show.toml``.

Each show folder may contain a ``show.toml`` with the canonical show name,
an optional RSS feed URL, a speaker roster, and a primary language.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

SHOW_META_FILENAME = "show.toml"


@dataclass
class PipelineDefaults:
    """Expected pipeline settings for a show — used to detect outdated runs.

    Fields default to empty/None so "unset" can be distinguished from an
    explicit user choice — callers merging these into effective defaults
    only override when a value is actually set.
    """

    # Transcribe
    model_size: str = ""
    diarize: bool | None = None
    # Correct / Translate (LLM)
    llm_mode: str = ""  # "ollama" | "api"
    llm_provider: str = ""  # "openai", "anthropic", etc.
    llm_model: str = ""
    # Translate
    target_lang: str = ""


@dataclass
class ShowMeta:
    """Show-level metadata persisted in ``show.toml``."""

    name: str
    rss_url: str = ""
    youtube_url: str = ""
    speakers: list[str] = field(default_factory=list)
    language: str = ""
    artwork_url: str = ""
    pipeline: PipelineDefaults = field(default_factory=PipelineDefaults)


# Re-parsing TOML for every show on every render was the single biggest
# HomePage stall — mtime-keyed so save_show_meta auto-invalidates.
_SHOW_META_CACHE: dict[str, tuple[float, ShowMeta | None]] = {}


def load_show_meta(show_folder: Path) -> ShowMeta | None:
    """Read ``show.toml`` from *show_folder*. Returns None if the file is missing."""
    path = Path(show_folder) / SHOW_META_FILENAME
    cache_key = str(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _SHOW_META_CACHE.pop(cache_key, None)
        return None
    except OSError:
        return None

    cached = _SHOW_META_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        logger.warning(f"Invalid show.toml, skipping: {path} ({exc})")
        _SHOW_META_CACHE[cache_key] = (mtime, None)
        return None
    pipe_raw = raw.get("pipeline", {})
    pipeline = PipelineDefaults(
        model_size=pipe_raw.get("model_size", ""),
        diarize=pipe_raw.get("diarize"),
        llm_mode=pipe_raw.get("llm_mode", ""),
        llm_provider=pipe_raw.get("llm_provider", ""),
        llm_model=pipe_raw.get("llm_model", ""),
        target_lang=pipe_raw.get("target_lang", ""),
    )
    meta = ShowMeta(
        name=raw.get("name", ""),
        rss_url=raw.get("rss_url", ""),
        youtube_url=raw.get("youtube_url", ""),
        speakers=raw.get("speakers", []),
        language=raw.get("language", ""),
        artwork_url=raw.get("artwork_url", ""),
        pipeline=pipeline,
    )
    _SHOW_META_CACHE[cache_key] = (mtime, meta)
    return meta


def show_display(folder: Path) -> str:
    """Human-readable show name: ``show.toml.name`` if present, else folder basename."""
    folder = Path(folder)
    meta = load_show_meta(folder)
    return (meta.name if meta else None) or folder.name


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
    if meta.youtube_url:
        lines.append(f'youtube_url = "{_toml_string(meta.youtube_url)}"')
    if meta.language:
        lines.append(f'language = "{_toml_string(meta.language)}"')
    if meta.artwork_url:
        lines.append(f'artwork_url = "{_toml_string(meta.artwork_url)}"')
    if meta.speakers:
        items = ", ".join(f'"{_toml_string(s)}"' for s in meta.speakers)
        lines.append(f"speakers = [{items}]")

    # Pipeline defaults section
    p = meta.pipeline
    pipe_lines: list[str] = []
    if p.model_size:
        pipe_lines.append(f'model_size = "{_toml_string(p.model_size)}"')
    if p.diarize is not None:
        pipe_lines.append(f"diarize = {'true' if p.diarize else 'false'}")
    if p.llm_mode:
        pipe_lines.append(f'llm_mode = "{_toml_string(p.llm_mode)}"')
    if p.llm_provider:
        pipe_lines.append(f'llm_provider = "{_toml_string(p.llm_provider)}"')
    if p.llm_model:
        pipe_lines.append(f'llm_model = "{_toml_string(p.llm_model)}"')
    if p.target_lang:
        pipe_lines.append(f'target_lang = "{_toml_string(p.target_lang)}"')
    if pipe_lines:
        lines.append("")
        lines.append("[pipeline]")
        lines.extend(pipe_lines)

    from podcodex.core._utils import atomic_write

    body = "\n".join(lines) + "\n"
    atomic_write(
        path,
        lambda p: p.write_text(body, encoding="utf-8"),
        suffix=".toml",
    )
    return path
