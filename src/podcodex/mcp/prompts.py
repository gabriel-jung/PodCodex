"""User-managed MCP prompt templates.

Prompts are named templates with slots (``{topic}``, ``{show}``...) that
MCP clients surface in their slash menu. Unlike tools, the user invokes
them explicitly — they don't add context until picked.

Storage: ``~/.config/podcodex/mcp_prompts.json`` (global, not per-show).
Five built-ins are seeded on first load and cannot be deleted, only
toggled off. User-defined prompts are fully editable.

Live re-registration: after any CRUD mutation ``reregister_all`` clears
FastMCP's internal prompt dict and re-adds every enabled entry. New
prompts become callable without restarting the Python process; Claude
Desktop still needs its own restart to re-handshake the catalog.
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import Prompt
from mcp.server.fastmcp.prompts.base import PromptArgument

from podcodex.core._utils import write_json_atomic


SlotType = Literal["string", "enum", "int", "bool"]

PROMPTS_PATH = Path.home() / ".config" / "podcodex" / "mcp_prompts.json"

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{1,39}$")
_TEMPLATE_SLOT_RE = re.compile(r"\{(\w+)\}")

# Tool names already registered — user prompt ids must not collide.
_RESERVED_IDS = frozenset({"search", "exact", "list_shows", "get_context"})


@dataclass
class SlotDef:
    name: str
    type: SlotType = "string"
    required: bool = True
    default: str | None = None
    options: list[str] = field(default_factory=list)


@dataclass
class PromptDef:
    id: str
    name: str
    title: str
    description: str
    template: str
    slots: list[SlotDef] = field(default_factory=list)
    enabled: bool = True
    is_builtin: bool = False


# ── Validation ──────────────────────────────────────────────────────────


class PromptValidationError(ValueError):
    """Raised when a PromptDef fails validation."""


def validate_prompt(p: PromptDef) -> None:
    """Raise ``PromptValidationError`` if the definition is malformed.

    Checks:
      - slug shape (``^[a-z][a-z0-9_-]{1,39}$``)
      - no collision with built-in tool names
      - template references only declared slots (undeclared ``{foo}`` rejected)
      - slot names are unique and slug-shaped
    """
    if not _SLUG_RE.match(p.id):
        raise PromptValidationError(
            f"Invalid id {p.id!r}. Must be lowercase, start with a letter, "
            "2-40 chars of [a-z0-9_-]."
        )
    if p.id in _RESERVED_IDS:
        raise PromptValidationError(f"Id {p.id!r} collides with a built-in tool name.")
    slot_names = [s.name for s in p.slots]
    if len(slot_names) != len(set(slot_names)):
        raise PromptValidationError("Duplicate slot names.")
    for s in p.slots:
        if not _SLUG_RE.match(s.name):
            raise PromptValidationError(f"Invalid slot name {s.name!r}.")
    declared = set(slot_names)
    used = set(_TEMPLATE_SLOT_RE.findall(p.template))
    undeclared = used - declared
    if undeclared:
        raise PromptValidationError(
            f"Template uses undeclared slots: {sorted(undeclared)}."
        )


# ── Storage ─────────────────────────────────────────────────────────────


def _prompts_path() -> Path:
    """Indirection for tests — overridden via monkeypatch."""
    return PROMPTS_PATH


def load_prompts() -> list[PromptDef]:
    """Read prompts from disk, seeding built-ins on first run.

    Built-ins are merged in on every load — if the user deleted a
    built-in from the file by hand it is re-added. The user's own
    ``enabled`` override on a built-in is respected if present.
    """
    path = _prompts_path()
    stored: list[PromptDef] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = []
    except (OSError, json.JSONDecodeError):
        logger.exception(f"prompts: failed to parse {path}; ignoring file")
        raw = []
    for item in raw:
        try:
            stored.append(_from_dict(item))
        except Exception:
            logger.exception(f"prompts: skipping malformed entry {item!r}")

    # Merge built-ins: add any missing, preserve user 'enabled' flag if set.
    existing_ids = {p.id for p in stored}
    out = list(stored)
    for builtin in _builtin_prompts():
        if builtin.id not in existing_ids:
            out.append(builtin)
    # Stable UI ordering: built-ins first (alphabetical), then user prompts.
    out.sort(key=lambda p: (not p.is_builtin, p.id))
    return out


def save_prompts(prompts: list[PromptDef]) -> None:
    """Persist prompts to disk atomically."""
    write_json_atomic(
        _prompts_path(),
        [_to_dict(p) for p in prompts],
        prefix=".prompts_",
    )


def _to_dict(p: PromptDef) -> dict:
    d = asdict(p)
    d["slots"] = [asdict(s) for s in p.slots]
    return d


def _from_dict(d: dict) -> PromptDef:
    slots = [SlotDef(**s) for s in d.get("slots") or []]
    return PromptDef(
        id=d["id"],
        name=d.get("name", d["id"]),
        title=d.get("title", d["id"]),
        description=d.get("description", ""),
        template=d["template"],
        slots=slots,
        enabled=bool(d.get("enabled", True)),
        is_builtin=bool(d.get("is_builtin", False)),
    )


# ── Built-in templates ─────────────────────────────────────────────────


_CITATION_FORMAT = (
    "Always cite using the chunk's `episode_title` field (the human "
    "readable title), not the raw `episode` stem, and the `start` "
    "timestamp formatted MM:SS. Citation shape: "
    "`[Show • Episode title @ MM:SS]`."
)

_SOURCE_LABELS = (
    "Mark general-knowledge claims outside these transcripts with "
    "`(outside the transcripts)`. Mark your own interpretation or "
    "synthesis with `(inference)`. The user is paying for analysis, not "
    "a transcript dump — they already have a frontend for raw search "
    "results."
)


def _builtin_prompts() -> list[PromptDef]:
    return [
        PromptDef(
            id="brief",
            name="brief",
            title="Synthesised brief on a topic",
            description="Research {topic} across the indexed podcasts and return a themed brief.",
            template=(
                "Produce a synthesised brief on {topic}, not a transcript "
                "dump. Do the research work: call `search` across every "
                "indexed show, issue follow-up queries if the first pass is "
                "thin, and expand the most relevant hits with `get_context` "
                "before drawing conclusions.\n\n"
                "Structure your answer as 3-5 themes YOU identify from the "
                "material. For each theme:\n"
                "  - State the insight in one sentence in your own words.\n"
                "  - Back it with one short quote or close paraphrase, "
                "cited.\n"
                "  - Note tensions, caveats, or contradictions if they "
                "surface.\n\n"
                "End with a short 'so what' paragraph: what the material "
                "reveals, or where it falls short. Skip exhaustive "
                "coverage — pick the interesting angles.\n\n"
                f"{_CITATION_FORMAT} {_SOURCE_LABELS}"
            ),
            slots=[SlotDef(name="topic", required=True)],
            is_builtin=True,
        ),
        PromptDef(
            id="speaker",
            name="speaker",
            title="Speaker profile",
            description="Build an analytical profile of {name} from across the index.",
            template=(
                "Build a profile of {name} from the indexed podcasts. Start "
                "with `speaker_stats` scoped to any show they appear in to "
                "get airtime context, then `search` with the speaker filter "
                "set to their name; expand standout hits with "
                "`get_context`.\n\n"
                "Organise by THEME — their recurring concerns, stances, "
                "rhetorical moves — not by episode. Include:\n"
                "  - A one-line characterisation.\n"
                "  - Recurring themes (2-4), each with a cited example.\n"
                "  - Notable tensions or evolutions across episodes.\n"
                "  - A short closing observation about what's distinctive.\n\n"
                f"{_CITATION_FORMAT} {_SOURCE_LABELS}"
            ),
            slots=[SlotDef(name="name", required=True)],
            is_builtin=True,
        ),
        PromptDef(
            id="quote",
            name="quote",
            title="Verify a quote",
            description="Check whether {phrase} appears in the transcripts and report context.",
            template=(
                'Verify the phrase "{phrase}" in the indexed transcripts. '
                "Call `exact` first for literal matches; if it returns "
                "nothing, fall back to `search` and report the closest "
                "wording and how it diverges. For each hit expand with "
                "`get_context` and quote the surrounding 1-2 sentences so "
                "the user can judge whether it really matches their "
                "intent.\n\n"
                "Present findings as a short list (one bullet per hit). "
                "Include a verdict at the top: confirmed / partial match / "
                "not found.\n\n"
                f"{_CITATION_FORMAT} {_SOURCE_LABELS}"
            ),
            slots=[SlotDef(name="phrase", required=True)],
            is_builtin=True,
        ),
        PromptDef(
            id="compare",
            name="compare",
            title="Compare shows on a topic",
            description="Analyse how each indexed show treats {topic} and what it reveals.",
            template=(
                "Compare how each indexed show discusses {topic}. Start "
                "with `list_shows`, then `search` per show, expanding the "
                "strongest hits with `get_context`. Produce an analytical "
                "comparison, not a per-show dump.\n\n"
                "Structure:\n"
                "  - A one-paragraph synthesis of the key axis of "
                "difference.\n"
                "  - Per show: a one-sentence characterisation of its "
                "angle, plus one or two cited passages that illustrate.\n"
                "  - A closing paragraph on what the contrast reveals "
                "about the shows themselves (tone, depth, audience).\n\n"
                "Shows with no relevant hits should be named explicitly, "
                "not omitted.\n\n"
                f"{_CITATION_FORMAT} {_SOURCE_LABELS}"
            ),
            slots=[SlotDef(name="topic", required=True)],
            is_builtin=True,
        ),
        PromptDef(
            id="timeline",
            name="timeline",
            title="Topic timeline with analysis",
            description="Chronological evolution of {topic} with commentary on shifts.",
            template=(
                "Build a chronological view of how {topic} surfaces across "
                "the transcripts. Use `search` broadly, expand with "
                "`get_context` where framing or tone matters. Order entries "
                "by episode (use the `start` field for intra-episode "
                "ordering).\n\n"
                "For each entry: citation, one-sentence summary of how "
                "the topic is framed at that moment, and a note when the "
                "framing shifts relative to earlier entries. End with a "
                "short commentary on the trajectory — does the topic "
                "deepen, fragment, recur identically?\n\n"
                f"{_CITATION_FORMAT} {_SOURCE_LABELS}"
            ),
            slots=[SlotDef(name="topic", required=True)],
            is_builtin=True,
        ),
    ]


# ── Registration with FastMCP ──────────────────────────────────────────


def _build_prompt(pdef: PromptDef) -> Prompt:
    """Wrap a ``PromptDef`` as a FastMCP ``Prompt`` object."""
    # The FastMCP prompt fn renders the template with provided kwargs.
    template = pdef.template

    pid = pdef.id

    def fn(**kwargs) -> str:
        try:
            return template.format(**kwargs)
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"Prompt {pid!r} missing required slot {missing!r}"
            ) from exc

    # Name the function so FastMCP derives a sensible ident for it.
    fn.__name__ = pdef.id

    arguments = [
        PromptArgument(
            name=slot.name,
            description=f"{slot.type} slot",
            required=slot.required,
        )
        for slot in pdef.slots
    ]

    return Prompt(
        name=pdef.id,
        title=pdef.title,
        description=pdef.description,
        arguments=arguments,
        fn=fn,
    )


def reregister_all(mcp: FastMCP, prompts: list[PromptDef] | None = None) -> None:
    """Clear FastMCP's live prompt registry and re-add every enabled prompt.

    Relies on ``mcp._prompt_manager._prompts`` being a mutable dict — an
    internal detail of the mcp SDK. If a future release reshapes it the
    clear is skipped (stale prompts linger until process restart), and a
    warning is logged so the divergence is visible.
    """
    if prompts is None:
        prompts = load_prompts()
    mgr = mcp._prompt_manager
    store = getattr(mgr, "_prompts", None)
    if isinstance(store, dict):
        store.clear()
    else:
        logger.warning(
            "mcp SDK no longer exposes _prompt_manager._prompts as a dict; "
            "prompt edits won't take effect until PodCodex restarts."
        )
    for pdef in prompts:
        if not pdef.enabled:
            continue
        mgr.add_prompt(_build_prompt(pdef))


# ── Live reload (file watcher) ─────────────────────────────────────────
#
# When the stdio subprocess is running inside Claude Desktop, edits made
# from the desktop app's Settings → Claude Desktop → Prompts UI land on
# disk but the subprocess doesn't know. A cheap mtime poll + session
# notification lets the `/` slash menu update without restarting Claude
# Desktop — the subprocess still has to be the one spawned by Claude,
# which it is.

_active_session = None  # type: ignore[var-annotated]  # ServerSession when a client is connected


def _install_session_capture(mcp: FastMCP) -> None:
    """Monkey-patch the low-level dispatcher so we keep a reference to the
    current session. Needed to push ``prompts/list_changed`` notifications
    from the file watcher, which runs outside any request's context.

    Reaches into ``mcp._mcp_server._handle_request`` — internal to the
    SDK. If a future release renames or seals these, live-reload silently
    degrades to "restart PodCodex to apply" and a warning is logged.
    """
    try:
        server = mcp._mcp_server
        if getattr(server, "_podcodex_session_capture_installed", False):
            return
        original = server._handle_request
    except AttributeError:
        logger.warning(
            "mcp SDK no longer exposes _mcp_server._handle_request; live "
            "prompt reload notifications disabled."
        )
        return

    async def _capture(message, req, session, lifespan_context, raise_exceptions):
        global _active_session
        _active_session = session
        return await original(message, req, session, lifespan_context, raise_exceptions)

    server._handle_request = _capture  # type: ignore[method-assign]
    server._podcodex_session_capture_installed = True  # type: ignore[attr-defined]


async def _watch_prompts_file(mcp: FastMCP, interval: float = 2.0) -> None:
    """Poll the prompts JSON file; on mtime change, reregister + notify.

    Errors are logged and swallowed so a transient disk glitch doesn't
    tear the stdio server down.
    """
    import anyio

    path = _prompts_path()
    last_mtime: float | None = None
    try:
        last_mtime = path.stat().st_mtime
    except FileNotFoundError:
        pass

    while True:
        await anyio.sleep(interval)
        try:
            current = path.stat().st_mtime
        except FileNotFoundError:
            current = None
        if current == last_mtime:
            continue
        last_mtime = current
        try:
            reregister_all(mcp)
            logger.info("prompts: reloaded from disk")
        except Exception:
            logger.exception("prompts: reload failed")
            continue
        session = _active_session
        if session is not None:
            try:
                await session.send_prompt_list_changed()
            except Exception:
                logger.exception("prompts: notification failed")


@asynccontextmanager
async def live_reload_lifespan(mcp: FastMCP):
    """FastMCP lifespan that installs session capture + a prompt watcher.

    Assign via ``mcp.settings.lifespan = live_reload_lifespan`` before
    ``mcp.run()`` (the stdio entrypoint sets this in ``main()``).
    """
    import anyio

    global _active_session
    _install_session_capture(mcp)
    async with anyio.create_task_group() as tg:
        tg.start_soon(_watch_prompts_file, mcp)
        try:
            yield {}
        finally:
            tg.cancel_scope.cancel()
            _active_session = None
