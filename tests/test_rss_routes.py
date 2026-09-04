"""RSS fetch route: the cached-feed fallback, and the per-episode stem scan.

Two things the route promises and used not to keep. A network failure is
supposed to serve the cache rather than block the show page, but only
``ValueError`` was caught, so a DNS or CDN failure escaped as a 500 with the
cache sitting right there. And the ``downloaded`` flag is supposed to cost
one directory listing per request, not one per episode.
"""

from __future__ import annotations

import httpx
import pytest

from podcodex.core.app_config import AppConfig
from podcodex.ingest.rss import RSSEpisode, save_feed_cache
from podcodex.ingest.show import ShowMeta, save_show_meta
from tests.fixtures.api_client import make_client

FEED = "https://example.com/feed.xml"

_EPISODES = [
    RSSEpisode(guid="a", title="One", pub_date="2024-01-01", audio_url="https://a/1"),
    RSSEpisode(guid="b", title="Two", pub_date="2024-01-02", audio_url="https://a/2"),
]


@pytest.fixture
def show(tmp_path):
    folder = tmp_path / "shows" / "kept"
    folder.mkdir(parents=True)
    save_show_meta(folder, ShowMeta(name="Kept Show", rss_url=FEED))
    return folder


@pytest.fixture
def client(tmp_path, monkeypatch, show):
    return make_client(
        tmp_path, monkeypatch, config=AppConfig(show_folders=[str(show)])
    )


def _fetch_feed(client, show):
    return client.post(f"/api/shows/{show}/rss/fetch")


def test_a_network_failure_serves_the_cached_feed(client, monkeypatch, show):
    """httpx errors are not ValueError; they used to escape as a 500."""
    import podcodex.api.routes.rss as rss_mod

    save_feed_cache(show, _EPISODES)

    def _boom(_url):
        raise httpx.ConnectError("nodename nor servname provided")

    monkeypatch.setattr(rss_mod, "fetch_feed_with_artwork", _boom)

    r = _fetch_feed(client, show)

    assert r.status_code == 200, r.text
    assert [e["guid"] for e in r.json()] == ["a", "b"]


def test_a_network_failure_with_no_cache_still_reports_an_error(
    client, monkeypatch, show
):
    import podcodex.api.routes.rss as rss_mod

    monkeypatch.setattr(
        rss_mod,
        "fetch_feed_with_artwork",
        lambda _u: (_ for _ in ()).throw(httpx.ReadTimeout("t")),
    )

    r = _fetch_feed(client, show)

    assert r.status_code == 502


def test_a_bad_url_is_still_a_400(client, monkeypatch, show):
    """The scheme rejection is the caller's fault; no cache stands in."""
    import podcodex.api.routes.rss as rss_mod

    save_feed_cache(show, _EPISODES)
    monkeypatch.setattr(
        rss_mod,
        "fetch_feed_with_artwork",
        lambda _u: (_ for _ in ()).throw(ValueError("Only http(s) URLs are allowed")),
    )

    r = _fetch_feed(client, show)

    assert r.status_code == 400


def test_downloaded_costs_one_listing_for_the_whole_feed(client, monkeypatch, show):
    """`is_downloaded` relists and stats the show folder on every call."""
    import podcodex.api.routes._helpers as helpers_mod
    import podcodex.api.routes.rss as rss_mod

    from podcodex.ingest.rss import episode_stem

    downloaded_stem = episode_stem(_EPISODES[0], show)
    (show / f"{downloaded_stem}.mp3").write_bytes(b"x")
    # One call now returns both; the artwork upgrade reuses it rather than
    # re-downloading the feed.
    monkeypatch.setattr(
        rss_mod, "fetch_feed_with_artwork", lambda _u: (list(_EPISODES), "")
    )

    calls: list[str] = []
    real = helpers_mod.is_downloaded
    monkeypatch.setattr(
        helpers_mod,
        "is_downloaded",
        lambda folder, stem: calls.append(stem) or real(folder, stem),
    )

    r = _fetch_feed(client, show)

    assert r.status_code == 200, r.text
    assert calls == []
    flags = {e["guid"]: e["downloaded"] for e in r.json()}
    assert flags == {"a": True, "b": False}


def test_the_feed_is_downloaded_once_even_when_artwork_is_upgraded(
    client, monkeypatch, show
):
    """The artwork upgrade used to re-download and re-parse the same XML."""
    import podcodex.api.routes.rss as rss_mod
    from podcodex.ingest.show import ShowMeta, load_show_meta, save_show_meta

    # A low-res cover is the condition that triggers the upgrade.
    save_show_meta(
        show,
        ShowMeta(name="Kept Show", rss_url=FEED, artwork_url="https://a/60x60.jpg"),
    )

    calls: list[str] = []

    def _fetch(url):
        calls.append(url)
        return list(_EPISODES), "https://a/3000x3000.jpg"

    monkeypatch.setattr(rss_mod, "fetch_feed_with_artwork", _fetch)

    r = _fetch_feed(client, show)

    assert r.status_code == 200, r.text
    assert len(calls) == 1
    assert load_show_meta(show).artwork_url == "https://a/3000x3000.jpg"
