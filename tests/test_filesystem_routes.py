"""File browser routes: the delete gate and the listing filters."""

from __future__ import annotations

import os

import pytest

from tests.fixtures.api_client import make_client


@pytest.fixture
def client(tmp_path, monkeypatch):
    return make_client(tmp_path, monkeypatch)


@pytest.fixture
def show(client, tmp_path):
    path = tmp_path / "Show"
    (path / "ep").mkdir(parents=True)
    r = client.post("/api/shows/register", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return path


def _delete(client, path):
    return client.delete("/api/fs/file", params={"path": str(path)})


def test_delete_an_allowed_file_inside_a_show(client, show):
    f = show / "ep" / "ep.srt"
    f.write_text("x")
    assert _delete(client, f).status_code == 200
    assert not f.exists()


def test_delete_refuses_a_disallowed_suffix(client, show):
    f = show / "ep" / "ep.mp3"
    f.write_bytes(b"x")
    assert _delete(client, f).status_code == 400
    assert f.exists()


def test_delete_refuses_a_file_outside_registered_shows(client, tmp_path):
    """A show.toml alone (unregistered, or planted) is not a show."""
    other = tmp_path / "Other"
    other.mkdir()
    (other / "show.toml").write_text('name = "x"\n')
    f = other / "notes.txt"
    f.write_text("keep")
    assert _delete(client, f).status_code == 403
    assert f.exists()


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_delete_refuses_a_symlink_out_of_the_show(client, show, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("keep")
    link = show / "ep" / "link.txt"
    link.symlink_to(outside)
    assert _delete(client, link).status_code == 403
    assert outside.exists()


def test_list_hides_dotfiles_and_filters_by_extension(client, tmp_path):
    root = tmp_path / "browse"
    root.mkdir()
    (root / ".hidden").mkdir()
    (root / "visible").mkdir()
    (root / "a.mp3").write_bytes(b"")
    (root / "b.txt").write_text("")
    (root / ".c.mp3").write_bytes(b"")

    r = client.get("/api/fs/list", params={"path": str(root), "show_files": True})
    body = r.json()
    assert [d["name"] for d in body["dirs"]] == ["visible"]
    assert [f["name"] for f in body["files"]] == ["a.mp3"]

    r = client.get(
        "/api/fs/list",
        params={"path": str(root), "show_files": True, "extensions": "txt"},
    )
    assert [f["name"] for f in r.json()["files"]] == ["b.txt"]
