"""Moving or deleting a show must wait for every task that writes into it.

``task_manager`` locks are plain strings and each submit path mints its own
shape (bare folder, ``batch:``, ``download:``, per-episode audio path), so
the show-level guard has to answer for all of them.
"""

from __future__ import annotations


import pytest

from podcodex.core.app_config import AppConfig
from tests.fixtures.api_client import make_client
from tests.fixtures.tasks import active_task


@pytest.fixture
def client(tmp_path, monkeypatch):
    return make_client(
        tmp_path,
        monkeypatch,
        config=AppConfig(default_save_path=str(tmp_path / "library")),
    )


@pytest.fixture
def show(client, tmp_path):
    path = tmp_path / "MyShow"
    path.mkdir()
    r = client.post("/api/shows/register", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return path


def _keys(show):
    return [
        str(show),
        f"batch:{show}",
        f"download:{show}",
        str(show / "ep1.mp3"),
        str(show / "ep1.virtual"),
    ]


@pytest.mark.parametrize("key_index", range(5))
def test_delete_blocks_on_every_lock_shape(client, show, key_index):
    key = _keys(show)[key_index]
    with active_task(key):
        r = client.post(f"/api/shows/{show}/delete", json={"delete_files": True})
    assert r.status_code == 409, (key, r.text)
    assert show.exists()


@pytest.mark.parametrize("key_index", range(5))
def test_move_blocks_on_every_lock_shape(client, show, tmp_path, key_index):
    key = _keys(show)[key_index]
    dest = tmp_path / "Elsewhere"
    with active_task(key):
        r = client.post(f"/api/shows/{show}/move", json={"new_path": str(dest)})
    assert r.status_code == 409, (key, r.text)
    assert show.exists()
    assert not dest.exists()


def test_sibling_show_lock_does_not_block(client, show, tmp_path):
    """A prefix match on the string is not enough: ``MyShow2`` is not ``MyShow``."""
    sibling = tmp_path / "MyShow2"
    with active_task(f"batch:{sibling}"), active_task(str(sibling / "x.mp3"), "t2"):
        r = client.post(f"/api/shows/{show}/delete", json={"delete_files": False})
    assert r.status_code == 200, r.text


def test_finished_task_does_not_block(client, show):
    from podcodex.api.tasks import TaskInfo, task_manager

    import time

    info = TaskInfo(task_id="done", audio_path=f"batch:{show}")
    info.status = "completed"
    # run()'s finally stamps this; it is the gate, not the status.
    info.finished_at = time.monotonic()
    task_manager._tasks["done"] = info
    task_manager.lock(f"batch:{show}", "done")
    try:
        r = client.post(f"/api/shows/{show}/delete", json={"delete_files": False})
    finally:
        task_manager.unlock(f"batch:{show}")
        task_manager._tasks.pop("done", None)
    assert r.status_code == 200, r.text
