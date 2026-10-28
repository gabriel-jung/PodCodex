"""
podcodex.ingest.show — Show-level metadata stored as ``show.toml``.

Each show folder may contain a ``show.toml`` with the show's stable id, its
display name, an optional RSS feed URL, a speaker roster, and a primary language.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
import uuid
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
    """Show-level metadata persisted in ``show.toml``.

    ``id`` is the show's stable identity: minted once, never rewritten, and
    the key every other store uses (LanceDB collections, bot passwords, RAG
    preferences). ``name`` is a pure display label that users may change at
    will; ``show_display`` is its only reader.
    """

    id: str = ""
    name: str = ""
    rss_url: str = ""
    youtube_url: str = ""
    speakers: list[str] = field(default_factory=list)
    language: str = ""
    artwork_url: str = ""
    # Regex (one capture group) applied to the episode title at index time to
    # extract a broadcast/diffusion number. Empty = no extraction for this show.
    broadcast_number_pattern: str = ""
    pipeline: PipelineDefaults = field(default_factory=PipelineDefaults)


class ShowMetaUnreadableError(OSError):
    """``show.toml`` exists but does not parse.

    An ``OSError`` so callers that already skip unwritable folders skip this
    one too. Raised instead of minting over the file: a fresh ``show.toml``
    would discard the existing id, which orphans every collection keyed on it.
    """


# Re-parsing TOML for every show on every render was the single biggest
# HomePage stall — mtime-keyed so save_show_meta auto-invalidates.
_SHOW_META_CACHE: dict[str, tuple[float, ShowMeta | None]] = {}


def _copy_meta(meta: ShowMeta | None) -> ShowMeta | None:
    """A copy safe to mutate: the scalars plus fresh containers.

    Field by field rather than ``deepcopy``: this runs on every cache hit,
    the hot path the cache exists for.
    """
    if meta is None:
        return None
    pipeline = dataclasses.replace(
        meta.pipeline, llm_models_by_mode=dict(meta.pipeline.llm_models_by_mode)
    )
    return dataclasses.replace(meta, speakers=list(meta.speakers), pipeline=pipeline)


def _as_str_list(value) -> list[str]:
    """A speaker roster from TOML: a hand-written ``speakers = "Alice"`` is
    one speaker, not five letters (``list("Alice")``)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


_ID_LINE = re.compile(r'^\s*id\s*=\s*"([^"\\\n]+)"', re.MULTILINE)


def salvage_show_id(folder: Path) -> str:
    """The id from a ``show.toml`` that no longer parses, or "".

    The id is written first, on a line of its own, so it usually survives
    whatever broke the rest of the file. Only for cleanup paths (deleting a
    show must still find its collections); everything else refuses the file.
    """
    try:
        text = (Path(folder) / SHOW_META_FILENAME).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return ""
    m = _ID_LINE.search(text)
    return m.group(1).strip() if m else ""


def load_show_meta(show_folder: Path) -> ShowMeta | None:
    """Read ``show.toml`` from *show_folder*.

    Returns None if the file is missing or does not parse. The result is a
    copy, so a caller mutating it without saving cannot change what the next
    reader sees.
    """
    path = Path(show_folder) / SHOW_META_FILENAME
    cache_key = str(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _SHOW_META_CACHE.pop(cache_key, None)
        return None
    except OSError:
        return None

    from podcodex.core._utils import mtime_settled

    cached = _SHOW_META_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return _copy_meta(cached[1])

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as exc:
        logger.warning(f"Invalid show.toml, skipping: {path} ({exc})")
        if mtime_settled(mtime):
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
        id=str(raw.get("id", "") or "").strip(),
        name=raw.get("name", ""),
        rss_url=raw.get("rss_url", ""),
        youtube_url=raw.get("youtube_url", ""),
        speakers=_as_str_list(raw.get("speakers")),
        language=raw.get("language", ""),
        artwork_url=raw.get("artwork_url", ""),
        broadcast_number_pattern=raw.get("broadcast_number_pattern", ""),
        pipeline=pipeline,
    )
    if mtime_settled(mtime):
        _SHOW_META_CACHE[cache_key] = (mtime, _copy_meta(meta))
    return meta


def is_feed_backed(show_folder: Path, meta: ShowMeta | None = None) -> bool:
    """True when a show's episode list comes from a feed (RSS or YouTube).

    Single definition of the rule, shared by the import endpoint and the
    ``accepts_imports`` flag both pickers gate on: three separate encodings
    of "is this a local show" previously disagreed, so the UI could offer a
    destination the server then rejected. A cached feed counts even without
    a URL (the URL may have been cleared), and a URL counts even without a
    cache (the first refresh will populate it).

    Pass ``meta`` when the caller already loaded it to skip a re-read.
    """
    show_folder = Path(show_folder)
    if meta is None:
        meta = load_show_meta(show_folder)
    if meta and (meta.rss_url or meta.youtube_url):
        return True
    return (show_folder / ".feed_cache.json").exists()


def show_display(folder: Path) -> str:
    """Human-readable show name: ``show.toml.name`` if present, else folder basename."""
    folder = Path(folder)
    meta = load_show_meta(folder)
    return (meta.name if meta else None) or folder.name


def show_id(folder: Path) -> str:
    """Stable id from ``show.toml``, or "" when the show has never been minted."""
    meta = load_show_meta(Path(folder))
    return meta.id if meta else ""


# Keeps ids readable in ``ls`` and log lines without letting a long folder
# name dominate the collection table name.
_MAX_SLUG_LEN = 40


def _slug(text: str) -> str:
    """Lowercase, collapse anything else to underscores, bound the length.

    Built on the same normalizer collection names use, deliberately: an id is
    embedded verbatim in a collection name, so it has to be a fixed point of
    that function. If the two drifted, two distinct ids could normalize onto
    one table.
    """
    from podcodex.rag.store import _normalize_show

    return _normalize_show(text)[:_MAX_SLUG_LEN].strip("_") or "show"


def _mint_id(name: str) -> str:
    """Build a fresh show id from a display name."""
    return f"{_slug(name)}_{uuid.uuid4().hex[:8]}"


def ensure_show_id(folder: Path) -> str:
    """Return the show's stable id, minting and persisting one if absent.

    The id is ``{slug}_{8 hex}``: the slug is frozen from whatever the show
    is called at mint time and never updated afterwards, so it stays readable
    without ever being authoritative. The random suffix is what makes the id
    unique, because two shows can genuinely carry the same name and a
    mint-time uniqueness check cannot see a show being minted on another
    machine.

    Creates ``show.toml`` when the folder has none. Idempotent.

    Raises:
        ShowMetaUnreadableError: ``show.toml`` exists but does not parse.
    """
    folder = Path(folder)
    meta = load_show_meta(folder)
    if meta and meta.id:
        return meta.id
    if meta is None:
        meta = ShowMeta(name=folder.name)  # save_show_meta refuses an unreadable file
    save_show_meta(folder, meta)  # mints on write
    logger.info(f"Minted show id {meta.id!r} for {folder}")
    return meta.id


_TOML_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\t": "\\t", "\r": "\\r"}


def _toml_string(s: str) -> str:
    """Escape a string for TOML double-quoted format.

    TOML forbids every control character but tab in a basic string, so a CR
    from a pasted feed title would otherwise write a file that no longer
    parses.
    """
    return "".join(
        _TOML_ESCAPES.get(c)
        or (f"\\u{ord(c):04X}" if ord(c) < 0x20 or ord(c) == 0x7F else c)
        for c in s
    )


def save_show_meta(show_folder: Path, meta: ShowMeta, *, remint: bool = False) -> Path:
    """Write ``show.toml`` to *show_folder*. Creates the directory if needed.

    Mints an ``id`` when the metadata has none, so identity is established
    the moment a show first exists on disk. Minting anywhere else made it a
    side effect of whichever reader touched the show first, and a show
    created and indexed before any of them ran got collections with no id,
    which a later rename then orphaned.

    *remint* (bundle import only) writes a fresh id over whatever the file
    holds, for an imported copy of a show whose carried id is taken.

    *remint* (bundle import only) writes a fresh id over whatever the file
    holds, for an imported copy of a show whose carried id is taken.

    Raises:
        ShowMetaUnreadableError: *meta* has no id and an existing
            ``show.toml`` does not parse; minting would discard its id.
        OSError: *meta* has no id but the existing file holds one (the
            caller's read of it failed a moment ago), or it cannot be read.
        OSError: *meta* has no id but the existing file holds one (the
            caller's read of it failed a moment ago), or it cannot be read.
    """
    folder = Path(show_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SHOW_META_FILENAME
    if not meta.id and path.exists() and not remint:
        _refuse_minting_over(path)
    # Assigned back only once the write lands, so a failed write cannot leave
    # an id on the caller's object that is not on disk.
    show_id = meta.id or _mint_id(meta.name or folder.name)
    body = render_show_toml(meta, show_id)

    from podcodex.core._utils import atomic_write

    atomic_write(
        path,
        lambda p: p.write_text(body, encoding="utf-8"),
        suffix=".toml",
    )
    meta.id = show_id
    return path


def _refuse_minting_over(path: Path) -> None:
    """Raise unless the existing *path* parses and holds no id to lose.

    The caller has a ShowMeta without an id, often a fresh one it fell back
    to because ``load_show_meta`` returned None; writing it would replace
    whatever id the file holds. A file that does not parse is refused for
    good (``ShowMetaUnreadableError``). One that cannot be read right now (a
    sync client or antivirus holding a lock) is a plain ``OSError``: the file
    may be fine, and telling the user to remove it could cost them the id.
    """
    try:
        existing = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ShowMetaUnreadableError(
            f"{path} exists but cannot be read ({exc}); fix or remove it "
            "before this show can be used"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ShowMetaUnreadableError(
            f"{path} is not valid UTF-8; fix or remove it before this show can be used"
        ) from exc
    if str(existing.get("id") or "").strip():
        # The file parses and carries an id, so the caller's id-less meta
        # comes from a read that failed (a lock now gone). Writing it would
        # mint over the id and fall back on every other field.
        raise OSError(f"{path} could not be read just now; try again")
    if str(existing.get("id") or "").strip():
        # The file parses and carries an id, so the caller's id-less meta
        # comes from a read that failed (a lock now gone). Writing it would
        # mint over the id and fall back on every other field.
        raise OSError(f"{path} could not be read just now; try again")


def render_show_toml(meta: ShowMeta, show_id: str) -> str:
    """The ``show.toml`` text for *meta* under *show_id*, without writing it."""
    lines: list[str] = []
    # Written first: it is the show's identity, and a human opening the file
    # should see it before the fields they are allowed to edit.
    lines.append(f'id = "{_toml_string(show_id)}"')
    lines.append(f'name = "{_toml_string(meta.name)}"')
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
    return "\n".join(lines) + "\n"
