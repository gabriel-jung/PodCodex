"""The MCP surface is built on first request, not at startup.

``mcp.server.fastmcp`` costs ~0.75 s to import (most of it jsonschema
loading its rfc3987_syntax format checker), which measured as 45% of
``podcodex.api.app``'s import time. Nothing reaches /mcp during an app
launch — only Claude Desktop or Claude Code connecting does — so paying it
at startup delayed every launch for a surface most sessions never touch.

That ``podcodex.api.app`` does not pull mcp in is asserted once, with the
rest of the deferred stack, in ``tests/test_startup_offloading.py``.
"""

from __future__ import annotations

import sys

import pytest

from tests.fixtures.api_client import make_client


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Entered as a context manager, unlike most API tests, because the
    lazy mount registers the sub-app's lifespan onto a stack the *parent*
    lifespan owns. A TestClient that never starts up has no such stack. An
    ASGI server never routes a request before startup completes, so this is
    the harness matching production, not a workaround."""
    with make_client(tmp_path, monkeypatch) as c:
        yield c


def test_mcp_is_advertised_without_being_imported(tmp_path, monkeypatch) -> None:
    """Availability is answered by find_spec, which resolves without exec.

    Deliberately does not start the app: the flag must be right from
    construction, since /api/integrations reads it to decide what to show.
    """
    app = make_client(tmp_path, monkeypatch).app
    assert app.state.mcp_available is True


def test_mcp_builds_on_first_request_and_is_reused(client) -> None:
    """One test, not two: ``podcodex.mcp.server.mcp`` is a module singleton
    whose streamable-http session manager refuses to start twice in a
    process, so only the first app built in a test session can host it.
    Splitting this would make the second test depend on execution order."""
    # An empty POST is a protocol error for streamable-http, which is fine:
    # what matters is that the sub-app answered at all rather than the
    # parent returning 404 for an unmounted path.
    response = client.post("/mcp", json={})

    assert response.status_code != 404
    assert response.status_code != 503, "sub-app failed to build on first request"
    assert "mcp.server.fastmcp" in sys.modules

    mount = client.app.state.mcp_mount
    built = mount.app
    assert built is not None

    client.post("/mcp", json={})

    assert mount.app is built, "sub-app rebuilt; the 0.75 s import would repeat"
