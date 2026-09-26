"""System routes: device override and extras install/remove."""

from __future__ import annotations

import os

import pytest

from tests.fixtures.api_client import make_client
from tests.fixtures.tasks import active_task


@pytest.fixture
def client(tmp_path, monkeypatch):
    return make_client(tmp_path, monkeypatch)


def test_auto_does_not_undo_the_kernel_guard(client, monkeypatch):
    """The guard demoted this process to CPU; "auto" must not re-enable CUDA."""
    import podcodex.core.device as device

    monkeypatch.setattr(device, "_kernel_guard_error", RuntimeError("no sm_61"))
    monkeypatch.setenv("PODCODEX_DEVICE", "cpu")
    r = client.post("/api/system/device", json={"override": "auto"})
    assert r.status_code == 200, r.text
    assert os.environ["PODCODEX_DEVICE"] == "cpu"


class _FakeProc:
    stdout: list[str] = []
    returncode = 0

    def wait(self):
        return 0


def _capture_uv(monkeypatch):
    from podcodex.api.routes import health

    calls: list[list[str]] = []

    def popen(cmd, **_k):
        calls.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(health.subprocess, "Popen", popen)
    monkeypatch.setattr(health, "_invalidate_capabilities", lambda: None)
    return health, calls


def test_install_is_inexact_so_unknown_extras_survive(monkeypatch):
    health, calls = _capture_uv(monkeypatch)
    health._run_uv(health._uv_cmd("sync", "--inexact"), lambda *_a: None, "x")
    assert "--inexact" in calls[0]


def test_install_and_remove_never_run_an_exact_sync(client, monkeypatch):
    """An exact sync dropped the torch variant, the dev group and any extra a
    capability probe missed; remove uninstalls the extra's own packages."""
    from podcodex.api.routes import health

    ran: list[list[str]] = []
    monkeypatch.setattr(health, "_run_uv", lambda cmd, *_a: ran.append(cmd) or {})
    monkeypatch.setattr(health, "_removal_plan", lambda _e: (["yt-dlp"], []))
    for route in ("install-extra", "remove-extra"):
        r = client.post(f"/api/system/{route}", json={"extra": "youtube"})
        assert r.status_code == 200, r.text
        _wait_task(r.json()["task_id"])
    install, remove = ran
    assert "sync" in install and "--inexact" in install
    assert "pip" in remove and remove[-1] == "yt-dlp" and "sync" not in remove


def _wait_task(task_id, timeout=5.0):
    import time

    from podcodex.api.tasks import task_manager

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = task_manager.get(task_id)
        if info is not None and info.finished_at is not None:
            return info
        time.sleep(0.02)
    raise AssertionError("task never finished")


def test_removal_plan_keeps_what_others_still_need():
    from podcodex.api.routes.health import _removal_plan

    assert _removal_plan("youtube") == (["yt-dlp"], [])
    # pyarrow and httpx are listed by pipeline but lancedb and mcp need them.
    pipeline, _ = _removal_plan("pipeline")
    assert "pyarrow" not in pipeline and "httpx" not in pipeline
    # soundfile is also in the dev group.
    assert "soundfile" not in pipeline


def test_removing_an_extra_another_includes_says_so():
    """mcp includes rag: removing rag alone removes nothing, and says why
    instead of reporting a success that changed nothing."""
    from podcodex.api.routes.health import _removal_plan

    names, keepers = _removal_plan("rag")
    assert names == []
    assert "mcp" in keepers and "gpu" not in keepers


def test_two_extras_operations_do_not_overlap(client):
    from podcodex.api.routes import health

    with active_task(health._EXTRAS_LOCK_KEY, "install_rag") as info:
        info.step, info.subject = "install", "rag"
        r = client.post("/api/system/install-extra", json={"extra": "bot"})
        assert r.status_code == 409, r.text
        assert "rag install" in r.json()["detail"]
        # The same click again reconnects to its own run.
        r = client.post("/api/system/install-extra", json={"extra": "rag"})
        assert r.json()["task_id"] == "install_rag"


def test_claude_desktop_entry_from_the_gpu_sidecar(tmp_path, monkeypatch):
    from podcodex.api.routes import integrations

    cpu = tmp_path / "podcodex-server"
    cpu.write_text("")
    monkeypatch.setenv("PODCODEX_CPU_SERVER", str(cpu))
    # The entry points at the bundled CPU binary, not the GPU one.
    assert integrations._bundled_server_path() == str(cpu.resolve())
    # An entry an older GPU sidecar wrote still reads as enabled.
    cfg = {
        "mcpServers": {
            integrations._SERVER_KEY: {
                "command": "/data/backends/gpu/podcodex-server-gpu",
                "args": ["--mcp"],
            }
        }
    }
    assert integrations._is_enabled(cfg)


def test_docs_are_off_in_the_shipped_app(tmp_path, monkeypatch):
    """/docs and /openapi.json sit outside the token-guarded /api/ prefix."""
    import podcodex.core.app_paths as app_paths

    monkeypatch.setattr(app_paths, "running_in_bundle", lambda: True)
    client = make_client(tmp_path, monkeypatch)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


def test_feed_urls_are_logged_without_their_secret():
    from podcodex.ingest.rss import loggable_url as _loggable_url

    assert (
        _loggable_url("https://user:pw@feeds.example.com:8443/p/show.xml?token=s3cret")
        == "https://feeds.example.com:8443/p/show.xml"
    )


def test_transcribe_start_accepts_a_null_batch_size():
    from podcodex.api.routes.transcribe import TranscribeRequest

    assert TranscribeRequest(audio_path="/x.mp3", batch_size=None).batch_size is None
    with pytest.raises(ValueError):
        TranscribeRequest(audio_path="/x.mp3", batch_size=0)
