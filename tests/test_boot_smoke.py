"""Boot-smoke wiring tests: the app factory builds, every router registers,
and the optional surfaces (bot, MCP) import cleanly.

CI skips the ``pipeline`` extra, so transcribe/synthesize execution has no
integration coverage there. These tests guard the wiring layer that unit
tests miss: router include-order regressions, import-time failures in
optional extras, and accidentally dropped route modules.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from podcodex.api.app import create_app


@pytest.fixture
def app(tmp_path, monkeypatch):
    """App built against an isolated config file (same pattern as test_api)."""
    from podcodex.core import app_config as app_config_mod

    monkeypatch.setattr(app_config_mod, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(app_config_mod, "_LOAD_CACHE", None)
    return create_app()


# One representative path per route module in src/podcodex/api/routes/.
# A missing entry means the router was never included in create_app().
REPRESENTATIVE_PATHS = [
    "/api/audio/clip",  # audio
    "/api/batch/start",  # batch
    "/api/bundle/preview",  # bundle
    "/api/config",  # config
    "/api/correct/start",  # correct
    "/api/episodes/{show}",  # episodes
    "/api/export/srt",  # export
    "/api/fs/list",  # filesystem
    "/api/gpu/status",  # gpu
    "/api/health",  # health
    "/api/index/start",  # index
    "/api/integrations/claude-desktop",  # integrations
    "/api/models",  # models
    "/api/podcasts/search",  # rss
    "/api/search/query",  # search
    "/api/shows/",  # shows
    "/api/synthesize/generate",  # synthesize
    "/api/transcribe/start",  # transcribe
    "/api/translate/start",  # translate
    "/api/ws",  # ws
    "/api/shows/{show_folder:path}/youtube/download",  # youtube
    "/mcp",  # MCP server mount
]


def test_every_router_registers(app):
    paths = {r.path for r in app.routes}
    missing = [p for p in REPRESENTATIVE_PATHS if p not in paths]
    assert not missing, f"routes missing from app wiring: {missing}"


def test_route_count_sane(app):
    # 131 unique paths as of 2026-07; a hard floor catches a router include
    # silently vanishing without breaking on every intentional route removal.
    paths = {r.path for r in app.routes}
    assert len(paths) >= 120, f"only {len(paths)} unique paths registered"


def test_api_does_not_import_discord():
    """The API must import cleanly without the ``bot`` extra installed.

    The packaged sidecar is built without ``--extra bot`` (see release.yml),
    so any module the API pulls in transitively must not import discord.py,
    or the shipped desktop backend dies at boot with ModuleNotFoundError.
    Runs in a subprocess: the in-process sys.modules is already polluted by
    the bot tests.
    """
    code = (
        "import sys, podcodex.api.app; sys.exit(1 if 'discord' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, (
        "podcodex.api.app transitively imports discord; move the shared code "
        "out of any module that imports discord (see core/show_passwords.py)"
    )


def test_bot_imports():
    import podcodex.bot.announce  # noqa: F401
    import podcodex.bot.bot  # noqa: F401


def test_mcp_imports():
    import podcodex.mcp.prompts  # noqa: F401
    import podcodex.mcp.server  # noqa: F401
