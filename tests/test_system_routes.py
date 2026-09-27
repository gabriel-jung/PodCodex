"""System routes: device override and extras install/remove."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.fixtures.api_client import make_client
from tests.fixtures.tasks import active_task, wait_task


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


def test_install_and_remove_never_run_an_exact_sync(client, monkeypatch):
    """An exact sync would drop the torch variant, the dev group and any
    extra a capability probe missed; remove uninstalls the extra's own
    packages."""
    from podcodex.api.routes import health

    ran: list[list[str]] = []
    monkeypatch.setattr(health, "_run_uv", lambda cmd, *_a: ran.append(cmd) or {})
    monkeypatch.setattr(health, "_removal_plan", lambda _e: (["yt-dlp"], []))
    for route in ("install-extra", "remove-extra"):
        r = client.post(f"/api/system/{route}", json={"extra": "youtube"})
        assert r.status_code == 200, r.text
        wait_task(r.json()["task_id"])
    install, remove = ran
    assert "sync" in install and "--inexact" in install
    assert "pip" in remove and remove[-1] == "yt-dlp" and "sync" not in remove


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
    client = make_client(tmp_path, monkeypatch, fresh=True)
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


def test_cancelling_an_extras_install_stops_uv(monkeypatch):
    """Cancelling stops the uv run in progress instead of letting it finish."""
    import sys
    import threading
    import time

    import podcodex.api.routes.health as health
    from podcodex.api.routes._helpers import TaskCancelled

    cancel = threading.Event()

    def progress_cb(_frac, msg):
        if msg == "working":
            cancel.set()  # cancelled while uv is mid-run

    progress_cb.cancel_event = cancel
    slow = [
        sys.executable,
        "-c",
        "import time; print('working', flush=True); time.sleep(30)",
    ]
    refreshed: list[bool] = []
    monkeypatch.setattr(
        health, "_invalidate_capabilities", lambda: refreshed.append(True)
    )

    start = time.monotonic()
    with pytest.raises(TaskCancelled):
        health._run_uv(slow, progress_cb, "Installing")

    assert time.monotonic() - start < 10
    # uv may have changed packages before it stopped.
    assert refreshed


def test_pipeline_defaults_roundtrip_and_isolation(client):
    """PUT /config/pipeline-defaults persists without touching other config."""
    # Fresh install: the sentinel is None (frontend migrates localStorage on it).
    assert client.get("/api/config").json()["pipeline_defaults"] is None

    r = client.put(
        "/api/config/pipeline-defaults",
        json={"target_lang": "German", "transcribe": {"model_size": "medium"}},
    )
    assert r.status_code == 200
    assert r.json()["target_lang"] == "German"

    cfg = client.get("/api/config").json()
    assert cfg["pipeline_defaults"]["transcribe"]["model_size"] == "medium"
    # Unsent fields land on the model's built-in defaults.
    assert cfg["pipeline_defaults"]["llm"]["batch_minutes"] == 15


def test_health_reports_backend_version(client):
    """Version rides on /health so the sidebar and boot splash can show it
    without a second request; the frontend compares it against the Tauri
    shell version to catch a half-applied installer run."""
    from podcodex import __version__

    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert isinstance(body["capabilities"], dict)


def test_about_reports_environment(client):
    r = client.get("/api/system/about")
    assert r.status_code == 200
    body = r.json()
    from podcodex import __version__

    assert body["version"] == __version__
    assert body["mode"] in {"bundle", "dev"}
    for key in (
        "python_version",
        "platform",
        "machine",
        "data_dir",
        "config_dir",
        "log_path",
    ):
        assert body[key], f"{key} should not be empty"


def test_extras_lists_known_extras(client):
    r = client.get("/api/system/extras")
    assert r.status_code == 200
    body = r.json()
    assert "extras" in body
    # At minimum, these four should always be listed
    assert set(body["extras"].keys()) >= {"pipeline", "rag", "bot", "youtube"}
    for ext in body["extras"].values():
        assert "description" in ext
        assert "installed" in ext


def test_drives_includes_resolved_home(client):
    r = client.get("/api/fs/drives")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["drives"], list)
    assert body["home"] == str(Path.home())


def test_gpu_download_ignores_a_caller_manifest_url(client, monkeypatch):
    """The downloaded archive becomes the executed sidecar, so a manifest URL
    in the body must never reach the installer: only the built-in one does."""
    from podcodex.api import gpu_backend

    used: list[str] = []
    monkeypatch.setattr(gpu_backend, "running_in_bundle", lambda: True)
    monkeypatch.setattr(
        gpu_backend, "default_manifest_url", lambda: "https://builtin.example/m.json"
    )
    monkeypatch.setattr(
        gpu_backend,
        "download_and_install",
        lambda _cb, manifest_url: used.append(manifest_url) or {},
    )
    r = client.post(
        "/api/gpu/download", json={"manifest_url": "http://attacker.example/m.json"}
    )
    assert r.status_code == 200, r.text
    wait_task(r.json()["task_id"])
    assert used == ["https://builtin.example/m.json"]
