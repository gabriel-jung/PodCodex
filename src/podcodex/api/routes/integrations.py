"""Integrations — toggles that expose PodCodex to external desktop apps.

Currently just Claude Desktop. The toggle writes (or removes) a stdio
``mcpServers.podcodex`` entry pointing at ``.venv/bin/podcodex-mcp`` in
``claude_desktop_config.json`` and preserves every other key. Claude
Desktop spawns the binary as a subprocess on startup.

(The API backend also mounts an HTTP MCP endpoint at ``/mcp`` for
Claude Code and other MCP clients that support HTTP transport. Claude
Desktop itself only reads stdio entries from ``claude_desktop_config``.)
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel

from podcodex.core._utils import write_json_atomic

router = APIRouter()


# ── Path resolution ─────────────────────────────────────────────────────


def _claude_config_path() -> Path:
    """Locate Claude Desktop's config file per-OS.

    macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json
    Linux:   ~/.config/Claude/claude_desktop_config.json
    Windows: %APPDATA%\\Claude\\claude_desktop_config.json
    """
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if system == "Windows":
        appdata = os.environ.get("APPDATA") or str(Path.home())
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def _claude_installed(config_path: Path) -> bool:
    """Heuristic: the Claude config directory exists."""
    return config_path.parent.exists()


# ── Config I/O ──────────────────────────────────────────────────────────


def _read_config(path: Path) -> dict:
    """Load the Claude config, returning an empty dict if missing or empty.

    Raises HTTPException(422) if the file exists but isn't valid JSON —
    the user's own edits should never be clobbered silently.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise HTTPException(500, f"Cannot read Claude config: {exc}") from exc
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            422,
            f"Claude Desktop config is not valid JSON ({exc.msg} at line "
            f"{exc.lineno}). Fix it manually at {path} and retry.",
        ) from exc


def _podcodex_mcp_path() -> str:
    """Resolve the ``podcodex-mcp`` stdio binary's absolute path.

    Tries PATH first (covers `make dev` where the venv is on PATH), then
    falls back to the directory of the running Python interpreter — that
    is the venv's ``bin``/``Scripts`` folder when the API itself runs
    inside it, which is the common case.
    """
    found = shutil.which("podcodex-mcp")
    if found:
        return str(Path(found).resolve())
    suffix = ".exe" if platform.system() == "Windows" else ""
    fallback = Path(sys.executable).parent / f"podcodex-mcp{suffix}"
    return str(fallback.resolve())


def _entry() -> dict:
    """The mcpServers.podcodex entry Claude Desktop should see.

    Claude Desktop's ``claude_desktop_config.json`` only accepts stdio
    entries — an HTTP shape is rejected with "invalid MCP server
    configuration". So we point it at the stdio binary and let Claude
    spawn it as a subprocess (the HTTP endpoint on the API backend is
    for other clients like Claude Code).
    """
    return {"command": _podcodex_mcp_path()}


def _is_enabled(cfg: dict) -> bool:
    """True if the config has a podcodex stdio entry pointing at our binary."""
    entry = (cfg.get("mcpServers") or {}).get("podcodex")
    if not isinstance(entry, dict):
        return False
    command = str(entry.get("command", ""))
    return command.endswith("podcodex-mcp") or command.endswith("podcodex-mcp.exe")


# ── Response model ─────────────────────────────────────────────────────


class ClaudeDesktopStatus(BaseModel):
    enabled: bool
    config_path: str
    command_path: str
    claude_desktop_installed: bool
    mcp_available: bool
    needs_restart_hint: str


def _status(request: Request) -> ClaudeDesktopStatus:
    config_path = _claude_config_path()
    mcp_available = bool(getattr(request.app.state, "mcp_available", False))
    try:
        cfg = _read_config(config_path)
    except HTTPException as exc:
        # Report status without failing the GET — surface the parse error to UI.
        logger.warning(f"integrations: config parse error: {exc.detail}")
        cfg = {}
    return ClaudeDesktopStatus(
        enabled=_is_enabled(cfg) if mcp_available else False,
        config_path=str(config_path),
        command_path=_podcodex_mcp_path(),
        claude_desktop_installed=_claude_installed(config_path),
        mcp_available=mcp_available,
        needs_restart_hint=(
            "Quit Claude Desktop fully (Cmd+Q on macOS, Alt+F4 elsewhere) "
            "and reopen it. Claude Desktop reads its MCP config only at "
            "startup."
        ),
    )


# ── Routes ──────────────────────────────────────────────────────────────


@router.get("/claude-desktop", response_model=ClaudeDesktopStatus)
async def get_claude_desktop_status(request: Request) -> ClaudeDesktopStatus:
    """Current toggle state + resolved paths/endpoint for the UI."""
    return _status(request)


@router.post("/claude-desktop/enable", response_model=ClaudeDesktopStatus)
async def enable_claude_desktop(request: Request) -> ClaudeDesktopStatus:
    """Write the mcpServers.podcodex entry, preserving every other key."""
    if not getattr(request.app.state, "mcp_available", False):
        raise HTTPException(
            503,
            "MCP extra is not installed on the server. Run "
            "`uv sync --extra desktop` and restart PodCodex.",
        )
    path = _claude_config_path()
    cfg = _read_config(path)
    servers = dict(cfg.get("mcpServers") or {})
    servers["podcodex"] = _entry()
    cfg["mcpServers"] = servers
    write_json_atomic(path, cfg, prefix=".claude_cfg_")
    logger.info(f"integrations: enabled podcodex entry in {path}")
    return _status(request)


@router.post("/claude-desktop/disable", response_model=ClaudeDesktopStatus)
async def disable_claude_desktop(request: Request) -> ClaudeDesktopStatus:
    """Remove only the podcodex entry. Every other key is preserved."""
    path = _claude_config_path()
    cfg = _read_config(path)
    servers = dict(cfg.get("mcpServers") or {})
    if "podcodex" in servers:
        servers.pop("podcodex")
        if servers:
            cfg["mcpServers"] = servers
        else:
            cfg.pop("mcpServers", None)
        write_json_atomic(path, cfg, prefix=".claude_cfg_")
        logger.info(f"integrations: removed podcodex entry from {path}")
    return _status(request)
