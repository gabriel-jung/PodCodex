"""Moving or deleting a show must wait for every task that writes into it.

``task_manager`` locks are plain strings and each submit path mints its own
shape (bare folder, ``batch:``, ``download:``, per-episode audio path), so
the show-level guard has to answer for all of them.
"""

from __future__ import annotations

import pytest

from tests.fixtures.api_client import library_client, registered_show
from tests.fixtures.tasks import active_task


@pytest.fixture
def client(tmp_path, monkeypatch):
    return library_client(tmp_path, monkeypatch)


@pytest.fixture
def show(client, tmp_path):
    return registered_show(client, tmp_path / "MyShow")


# Every lock shape is matched by task_manager.get_active_in_show, pinned in
# test_task_manager.py; one shape per route proves each route asks it.


def test_delete_blocks_on_a_task_inside_the_show(client, show):
    with active_task(f"batch:{show}"):
        r = client.post(f"/api/shows/{show}/delete", json={"delete_files": True})
    assert r.status_code == 409, r.text
    assert show.exists()


def test_move_blocks_on_a_task_inside_the_show(client, show, tmp_path):
    dest = tmp_path / "Elsewhere"
    with active_task(str(show / "ep1.mp3")):
        r = client.post(f"/api/shows/{show}/move", json={"new_path": str(dest)})
    assert r.status_code == 409, r.text
    assert show.exists()
    assert not dest.exists()


def test_sibling_show_lock_does_not_block(client, show, tmp_path):
    """A prefix match on the string is not enough: ``MyShow2`` is not ``MyShow``."""
    sibling = tmp_path / "MyShow2"
    with active_task(f"batch:{sibling}"), active_task(str(sibling / "x.mp3"), "t2"):
        r = client.post(f"/api/shows/{show}/delete", json={"delete_files": False})
    assert r.status_code == 200, r.text


def test_finished_task_does_not_block(client, show):
    """``finished_at`` is the gate, not the status: a task winding down still
    holds its lock but no longer blocks."""
    with active_task(f"batch:{show}", "done", finished=True):
        r = client.post(f"/api/shows/{show}/delete", json={"delete_files": False})
    assert r.status_code == 200, r.text


def _seed(show):
    (show / "show.toml").write_text('name = "MyShow"\n')
    (show / "ep1").mkdir()
    (show / "ep1" / "ep1.transcript.json").write_text("[]")


def _registered(client):
    return client.get("/api/config").json()["show_folders"]


def test_move_into_own_subfolder_is_refused(client, show):
    _seed(show)
    r = client.post(f"/api/shows/{show}/move", json={"new_path": str(show / "sub")})
    assert r.status_code == 400, r.text
    assert (show / "ep1" / "ep1.transcript.json").exists()


def test_move_into_own_parent_is_refused(client, show, tmp_path):
    _seed(show)
    r = client.post(f"/api/shows/{show}/move", json={"new_path": str(tmp_path)})
    assert r.status_code in (400, 409), r.text
    assert (show / "ep1" / "ep1.transcript.json").exists()


def test_move_onto_empty_folder_does_not_nest(client, show, tmp_path):
    _seed(show)
    dest = tmp_path / "Empty"
    dest.mkdir()
    r = client.post(f"/api/shows/{show}/move", json={"new_path": str(dest)})
    assert r.status_code == 200, r.text
    assert (dest / "show.toml").exists()
    assert (dest / "ep1" / "ep1.transcript.json").exists()
    assert not (dest / show.name).exists()
    assert not show.exists()
    assert str(dest.resolve()) in _registered(client)


def test_move_onto_non_empty_folder_is_refused(client, show, tmp_path):
    _seed(show)
    dest = tmp_path / "Busy"
    dest.mkdir()
    (dest / "keep.txt").write_text("x")
    r = client.post(f"/api/shows/{show}/move", json={"new_path": str(dest)})
    assert r.status_code == 409, r.text
    assert show.exists()


def test_delete_surfaces_a_failed_index_purge(client, show, monkeypatch):
    from podcodex.api.routes import shows as shows_route

    monkeypatch.setattr(
        shows_route,
        "_purge_show_from_index",
        lambda *_a, **_k: (0, False, "index is locked"),
    )
    r = client.post(f"/api/shows/{show}/delete", json={"delete_files": True})
    assert r.status_code == 200, r.text
    assert "index is locked" in r.json()["warning"]
    assert not show.exists()
