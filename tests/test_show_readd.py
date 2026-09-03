"""Re-adding a feed over a kept folder must restore the show, not re-mint it.

Unregistering keeps the folder; its id in ``show.toml`` is what the
collections and bot password are keyed on, so the create routes have to
carry it (and the user's speakers / pipeline defaults) forward.
"""

from __future__ import annotations

import pytest

from podcodex.ingest.rss import RSSEpisode
from podcodex.ingest.show import (
    PipelineDefaults,
    ShowMeta,
    load_show_meta,
    save_show_meta,
)
from tests.fixtures.api_client import make_client

FEED = "https://example.com/feed.xml"
OTHER_FEED = "https://example.com/other.xml"
YT = "https://www.youtube.com/@someone"

_EPISODES = [
    RSSEpisode(guid="a", title="One", pub_date="2024-01-01"),
    RSSEpisode(guid="b", title="Two", pub_date="2024-01-02"),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    import podcodex.api.routes.shows as shows_mod
    import podcodex.ingest.youtube as youtube_mod

    monkeypatch.setattr(
        shows_mod,
        "fetch_feed_with_artwork",
        lambda url: (_EPISODES, "https://art/x.jpg"),
    )
    monkeypatch.setattr(
        youtube_mod,
        "fetch_youtube",
        lambda url: (_EPISODES, {"name": "Chan", "artwork_url": "https://art/y.jpg"}),
    )
    return make_client(tmp_path, monkeypatch)


def _seed(tmp_path, **fields) -> tuple:
    folder = tmp_path / "shows" / "kept"
    folder.mkdir(parents=True)
    meta = ShowMeta(
        name="Kept Show",
        speakers=["Ann", "Bob"],
        broadcast_number_pattern=r"#(\d+)",
        pipeline=PipelineDefaults(model_size="large-v3", target_lang="fr"),
        **fields,
    )
    save_show_meta(folder, meta)
    return folder, load_show_meta(folder).id


def _post_rss(client, tmp_path, url=FEED, **extra):
    return client.post(
        "/api/shows/from-rss",
        json={
            "rss_url": url,
            "save_path": str(tmp_path / "shows"),
            "folder_name": "kept",
            **extra,
        },
    )


def test_rss_readd_keeps_identity_and_user_metadata(client, tmp_path):
    folder, old_id = _seed(tmp_path, rss_url=FEED, artwork_url="https://art/old.jpg")

    r = _post_rss(client, tmp_path, name="New Label", artwork_url="https://art/new.jpg")
    assert r.status_code == 200, r.text

    meta = load_show_meta(folder)
    assert meta.id == old_id
    assert meta.speakers == ["Ann", "Bob"]
    assert meta.broadcast_number_pattern == r"#(\d+)"
    assert meta.pipeline.model_size == "large-v3"
    assert meta.pipeline.target_lang == "fr"
    # Request-supplied fields do overlay.
    assert meta.name == "New Label"
    assert meta.artwork_url == "https://art/new.jpg"
    assert meta.rss_url == FEED


def test_rss_add_over_a_different_feed_is_refused(client, tmp_path):
    folder, _ = _seed(tmp_path, rss_url=OTHER_FEED)
    before = (folder / "show.toml").read_bytes()

    r = _post_rss(client, tmp_path)
    assert r.status_code == 409, r.text
    assert (folder / "show.toml").read_bytes() == before
    assert not (folder / ".feed_cache.json").exists()


def test_feed_over_a_local_show_adopts_its_id(client, tmp_path):
    folder, old_id = _seed(tmp_path)

    r = _post_rss(client, tmp_path)
    assert r.status_code == 200, r.text
    meta = load_show_meta(folder)
    assert meta.id == old_id
    assert meta.rss_url == FEED


def test_readd_keeps_the_other_feed_url(client, tmp_path):
    """A show can carry both URLs; re-adding one must not clear the other."""
    folder, old_id = _seed(tmp_path, rss_url=FEED, youtube_url=YT)

    r = _post_rss(client, tmp_path)
    assert r.status_code == 200, r.text
    meta = load_show_meta(folder)
    assert meta.id == old_id
    assert meta.rss_url == FEED
    assert meta.youtube_url == YT


def test_readd_of_the_second_feed_kind_is_not_refused(client, tmp_path):
    """The request's kind decides which existing URL it must match."""
    folder, old_id = _seed(tmp_path, rss_url=FEED, youtube_url=YT)

    r = client.post(
        "/api/shows/from-youtube",
        json={
            "youtube_url": YT,
            "save_path": str(tmp_path / "shows"),
            "folder_name": "kept",
        },
    )
    assert r.status_code == 200, r.text
    meta = load_show_meta(folder)
    assert meta.id == old_id
    assert meta.rss_url == FEED
    assert meta.youtube_url == YT


def test_youtube_readd_keeps_identity(client, tmp_path):
    folder, old_id = _seed(tmp_path, youtube_url=YT)

    r = client.post(
        "/api/shows/from-youtube",
        json={
            "youtube_url": YT,
            "save_path": str(tmp_path / "shows"),
            "folder_name": "kept",
        },
    )
    assert r.status_code == 200, r.text
    meta = load_show_meta(folder)
    assert meta.id == old_id
    assert meta.speakers == ["Ann", "Bob"]
    assert meta.youtube_url == YT


def test_fresh_folder_still_mints(client, tmp_path):
    (tmp_path / "shows").mkdir()
    r = _post_rss(client, tmp_path, folder_name="brand_new")
    assert r.status_code == 200, r.text
    meta = load_show_meta(tmp_path / "shows" / "brand_new")
    assert meta.id
    assert meta.rss_url == FEED
