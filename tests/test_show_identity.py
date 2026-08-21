"""Tests for stable show ids in show.toml (``ensure_show_id``, ``show_id``)."""

from __future__ import annotations

import re

from podcodex.ingest.show import (
    ShowMeta,
    ensure_show_id,
    load_show_meta,
    save_show_meta,
    show_display,
    show_id,
)

ID_RE = re.compile(r"^[a-z0-9_]+_[0-9a-f]{8}$")


def test_mint_creates_show_toml_when_missing(tmp_path):
    folder = tmp_path / "My Show"
    folder.mkdir()
    minted = ensure_show_id(folder)
    assert ID_RE.match(minted), minted
    assert minted.startswith("my_show_")
    assert (folder / "show.toml").exists()


def test_mint_is_idempotent(tmp_path):
    folder = tmp_path / "Show"
    folder.mkdir()
    first = ensure_show_id(folder)
    assert ensure_show_id(folder) == first


def test_id_survives_a_rename(tmp_path):
    """The whole point: the label changes, identity does not."""
    folder = tmp_path / "Old Name"
    folder.mkdir()
    minted = ensure_show_id(folder)

    meta = load_show_meta(folder)
    save_show_meta(folder, ShowMeta(id=meta.id, name="Completely New Name"))

    assert show_id(folder) == minted
    assert show_display(folder) == "Completely New Name"
    # The frozen slug still reads like the original name, deliberately.
    assert minted.startswith("old_name_")


def test_two_shows_with_the_same_name_get_different_ids(tmp_path):
    a = tmp_path / "a" / "News"
    b = tmp_path / "b" / "News"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    assert ensure_show_id(a) != ensure_show_id(b)


def test_unnamed_folder_falls_back_to_a_usable_slug(tmp_path):
    folder = tmp_path / "!!!"
    folder.mkdir()
    minted = ensure_show_id(folder)
    assert ID_RE.match(minted), minted
    assert minted.startswith("show_")


def test_slug_is_truncated(tmp_path):
    folder = tmp_path / ("x" * 200)
    folder.mkdir()
    minted = ensure_show_id(folder)
    assert len(minted) <= 64
    assert ID_RE.match(minted), minted


def test_saving_a_show_always_gives_it_an_id(tmp_path):
    """Identity is established when the show first exists on disk, not by
    whichever reader happens to touch it first."""
    folder = tmp_path / "Show"
    folder.mkdir()
    save_show_meta(folder, ShowMeta(name="Show"))
    assert ID_RE.match(show_id(folder))


def test_show_id_empty_without_show_toml(tmp_path):
    folder = tmp_path / "Show"
    folder.mkdir()
    assert show_id(folder) == ""


def test_id_round_trips_through_toml(tmp_path):
    folder = tmp_path / "Show"
    folder.mkdir()
    save_show_meta(folder, ShowMeta(id="show_deadbeef", name="Show", language="en"))
    meta = load_show_meta(folder)
    assert meta.id == "show_deadbeef"
    assert meta.language == "en"


def test_mint_preserves_existing_metadata(tmp_path):
    folder = tmp_path / "Show"
    folder.mkdir()
    save_show_meta(
        folder,
        ShowMeta(name="Show", rss_url="https://example.com/f.xml", speakers=["A", "B"]),
    )
    ensure_show_id(folder)
    meta = load_show_meta(folder)
    assert meta.rss_url == "https://example.com/f.xml"
    assert meta.speakers == ["A", "B"]
    assert meta.id
