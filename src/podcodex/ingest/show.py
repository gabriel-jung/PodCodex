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
    num_speakers: str = ""  # expected speaker count; "" = auto-detect
    # Correct / Translate (LLM)
    llm_mode: str = ""  # "ollama" | "api" | "manual"
    llm_provider_profile: str = ""  # name of a profile from the catalog
    llm_key_name: str = ""  # name of an entry in the api key pool
    # Model name keyed by mode ("ollama"/"api"/"manual"). Empty entry / missing
    # key means "inherit the app default for that mode". Stored as a per-mode
    # dict so a value set under one mode never leaks into another.
    llm_models_by_mode: dict[str, str] = field(default_factory=dict)
    # Max audio duration per LLM batch in minutes. ``None`` = inherit the app
    # default (currently 15). Overridden per-episode in the panel as a count.
    llm_batch_minutes: float | None = None
    context: str = ""  # show description fed to the LLM for accuracy
    # Translate
    target_lang: str = ""
    # RAG indexing: embedding model + chunker this show is indexed under.
    # Drives the index UI defaults and which collection the MCP server queries.
    rag_model: str = ""  # key from rag.defaults.MODELS
    rag_chunker: str = ""  # key from rag.defaults.CHUNKING_STRATEGIES


@dataclass
class ShowMeta:
    """Show-level metadata persisted in ``show.toml``."""

    name: str
    rss_url: str = ""
    youtube_url: str = ""
    speakers: list[str] = field(default_factory=list)
    language: str = ""
    artwork_url: str = ""
    # Regex (one capture group) applied to the episode title at index time to
    # extract a broadcast/diffusion number. Empty = no extraction for this show.
    broadcast_number_pattern: str = ""
    # Raw diarisation label -> canonical speaker, applied at read time (e.g.
    # ``{"Raf": "Rafik"}``). Retroactive: no reindex needed.
    speaker_aliases: dict[str, str] = field(default_factory=dict)
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
        num_speakers=pipe_raw.get("num_speakers", ""),
        llm_mode=pipe_raw.get("llm_mode", ""),
        llm_provider_profile=pipe_raw.get("llm_provider_profile", ""),
        llm_key_name=pipe_raw.get("llm_key_name", ""),
        llm_models_by_mode={
            str(k): str(v)
            for k, v in (pipe_raw.get("llm_models_by_mode") or {}).items()
            if v
        },
        llm_batch_minutes=pipe_raw.get("llm_batch_minutes"),
        context=pipe_raw.get("context", ""),
        target_lang=pipe_raw.get("target_lang", ""),
        rag_model=pipe_raw.get("rag_model", ""),
        rag_chunker=pipe_raw.get("rag_chunker", ""),
    )
    meta = ShowMeta(
        name=raw.get("name", ""),
        rss_url=raw.get("rss_url", ""),
        youtube_url=raw.get("youtube_url", ""),
        speakers=raw.get("speakers", []),
        language=raw.get("language", ""),
        artwork_url=raw.get("artwork_url", ""),
        broadcast_number_pattern=raw.get("broadcast_number_pattern", ""),
        speaker_aliases={
            str(k): str(v) for k, v in (raw.get("speaker_aliases") or {}).items() if v
        },
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
    if meta.broadcast_number_pattern:
        lines.append(
            f'broadcast_number_pattern = "{_toml_string(meta.broadcast_number_pattern)}"'
        )

    # Pipeline defaults section
    p = meta.pipeline
    pipe_lines: list[str] = []
    if p.model_size:
        pipe_lines.append(f'model_size = "{_toml_string(p.model_size)}"')
    if p.diarize is not None:
        pipe_lines.append(f"diarize = {'true' if p.diarize else 'false'}")
    if p.num_speakers:
        pipe_lines.append(f'num_speakers = "{_toml_string(p.num_speakers)}"')
    if p.llm_mode:
        pipe_lines.append(f'llm_mode = "{_toml_string(p.llm_mode)}"')
    if p.llm_provider_profile:
        pipe_lines.append(
            f'llm_provider_profile = "{_toml_string(p.llm_provider_profile)}"'
        )
    if p.llm_key_name:
        pipe_lines.append(f'llm_key_name = "{_toml_string(p.llm_key_name)}"')
    models_by_mode = {k: v for k, v in (p.llm_models_by_mode or {}).items() if k and v}
    if models_by_mode:
        # Quote both key and value so non-bare-key mode names (hyphen, dot,
        # whitespace) stay valid TOML across future mode additions.
        entries = ", ".join(
            f'"{_toml_string(k)}" = "{_toml_string(v)}"'
            for k, v in sorted(models_by_mode.items())
        )
        pipe_lines.append(f"llm_models_by_mode = {{ {entries} }}")
    if p.llm_batch_minutes is not None and p.llm_batch_minutes > 0:
        # `repr` keeps the decimal point so the value round-trips as TOML
        # float instead of degrading to int for whole numbers.
        pipe_lines.append(f"llm_batch_minutes = {float(p.llm_batch_minutes)!r}")
    if p.context:
        pipe_lines.append(f'context = "{_toml_string(p.context)}"')
    if p.target_lang:
        pipe_lines.append(f'target_lang = "{_toml_string(p.target_lang)}"')
    if p.rag_model:
        pipe_lines.append(f'rag_model = "{_toml_string(p.rag_model)}"')
    if p.rag_chunker:
        pipe_lines.append(f'rag_chunker = "{_toml_string(p.rag_chunker)}"')
    if pipe_lines:
        lines.append("")
        lines.append("[pipeline]")
        lines.extend(pipe_lines)

    # Speaker alias table. Emitted after [pipeline] so a TOML table header does
    # not swallow the pipeline scalar keys.
    if meta.speaker_aliases:
        lines.append("")
        lines.append("[speaker_aliases]")
        for raw_label, canonical in meta.speaker_aliases.items():
            lines.append(f'"{_toml_string(raw_label)}" = "{_toml_string(canonical)}"')

    from podcodex.core._utils import atomic_write

    body = "\n".join(lines) + "\n"
    atomic_write(
        path,
        lambda p: p.write_text(body, encoding="utf-8"),
        suffix=".toml",
    )
    return path
