"""Tests for the /api/bundle export routes."""

import pytest

from tests.fixtures.api_client import library_client, registered_show


@pytest.fixture
def client(tmp_path, monkeypatch):
    return library_client(tmp_path, monkeypatch)


def test_export_show_refuses_an_unregistered_folder(client, tmp_path):
    """Exporting mints a show id into the folder: an arbitrary directory
    must be refused before anything is written there."""
    stray = tmp_path / "not a show"
    stray.mkdir()
    r = client.post(
        "/api/bundle/export-show",
        json={"show_folder": str(stray), "output_path": str(tmp_path / "x.podcodex")},
    )
    assert r.status_code == 403
    assert not (stray / "show.toml").exists()
    assert not (tmp_path / "x.podcodex").exists()


def test_export_index_refuses_an_unregistered_folder(client, tmp_path):
    show = registered_show(client, tmp_path / "Real Show", name="Real Show")
    stray = tmp_path / "stray"
    stray.mkdir()
    r = client.post(
        "/api/bundle/export-index",
        json={
            "show_folders": [str(show), str(stray)],
            "output_path": str(tmp_path / "x.podcodex"),
        },
    )
    assert r.status_code == 403
    assert not (stray / "show.toml").exists()


def test_export_show_accepts_a_registered_show(client, tmp_path):
    show = registered_show(client, tmp_path / "Real Show", name="Real Show")
    out = tmp_path / "real.podcodex"
    r = client.post(
        "/api/bundle/export-show",
        json={"show_folder": str(show), "output_path": str(out)},
    )
    assert r.status_code == 200, r.text
    assert out.is_file()
