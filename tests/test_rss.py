"""Tests for podcodex.ingest.rss — RSS parsing, caching, slug generation."""

import pytest

from podcodex.ingest.rss import (
    RSSEpisode,
    _BACKFILL_FIELDS,
    _parse_duration,
    fill_empty_fields,
    load_episode_meta,
    load_feed_cache,
    parse_feed_content,
    save_episode_meta,
    save_feed_cache,
    slug_from_title,
)


# ──────────────────────────────────────────────
# slug_from_title
# ──────────────────────────────────────────────


def test_slug_basic():
    assert slug_from_title("Episode 1: Hello World!") == "episode_1_hello_world"


def test_slug_unicode():
    assert slug_from_title("Émission #42 — été") == "émission_42_été"


def test_slug_empty():
    assert slug_from_title("") == "untitled"
    assert slug_from_title("!!!") == "untitled"


def test_slug_truncated():
    long_title = "a" * 200
    assert len(slug_from_title(long_title)) == 120


# ──────────────────────────────────────────────
# _parse_duration
# ──────────────────────────────────────────────


def test_parse_duration_hhmmss():
    assert _parse_duration("01:23:45") == 5025.0


def test_parse_duration_mmss():
    assert _parse_duration("23:45") == 1425.0


def test_parse_duration_seconds():
    assert _parse_duration("3600") == 3600.0


def test_parse_duration_empty():
    assert _parse_duration("") == 0.0


def test_parse_duration_invalid():
    assert _parse_duration("not-a-time") == 0.0


# ──────────────────────────────────────────────
# Feed cache roundtrip
# ──────────────────────────────────────────────


def test_feed_cache_roundtrip(tmp_path):
    episodes = [
        RSSEpisode(
            guid="123",
            title="Test Episode",
            pub_date="2026-01-01",
            description="A test.",
            audio_url="https://example.com/ep.mp3",
            duration=1800.0,
        ),
        RSSEpisode(guid="456", title="Another", pub_date="2026-01-02"),
    ]
    save_feed_cache(tmp_path, episodes)
    loaded = load_feed_cache(tmp_path)

    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].guid == "123"
    assert loaded[0].title == "Test Episode"
    assert loaded[0].duration == 1800.0
    assert loaded[1].description == ""


def test_feed_cache_missing(tmp_path):
    assert load_feed_cache(tmp_path) is None


# ──────────────────────────────────────────────
# Episode meta roundtrip
# ──────────────────────────────────────────────


def test_episode_meta_roundtrip(tmp_path):
    ep = RSSEpisode(
        guid="abc",
        title="Épisode 116 - Solène et la dot au Moyen Âge",
        pub_date="2026-03-15",
        description="A great episode.",
        episode_number=116,
        season_number=2,
    )
    ep_dir = tmp_path / "116_épisode_116"
    save_episode_meta(ep_dir, ep)
    loaded = load_episode_meta(ep_dir)

    assert loaded is not None
    assert loaded.title == "Épisode 116 - Solène et la dot au Moyen Âge"
    assert loaded.episode_number == 116
    assert loaded.season_number == 2
    assert loaded.pub_date == "2026-03-15"


def test_episode_meta_missing(tmp_path):
    assert load_episode_meta(tmp_path) is None


# ──────────────────────────────────────────────
# fetch_feed with a local XML file
# ──────────────────────────────────────────────


_SAMPLE_RSS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Podcast</title>
    <item>
      <guid>ep-001</guid>
      <title>First Episode</title>
      <pubDate>Mon, 01 Jan 2026 00:00:00 GMT</pubDate>
      <description>The very first episode.</description>
      <enclosure url="https://example.com/ep001.mp3" type="audio/mpeg" length="12345"/>
      <itunes:duration>01:30:00</itunes:duration>
    </item>
    <item>
      <guid>ep-002</guid>
      <title>Second Episode</title>
      <pubDate>Tue, 02 Jan 2026 00:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""


def test_fetch_feed_from_local_xml(tmp_path):
    episodes = parse_feed_content(_SAMPLE_RSS)

    assert len(episodes) == 2
    ep1 = episodes[0]
    assert ep1.guid == "ep-001"
    assert ep1.title == "First Episode"
    assert ep1.audio_url == "https://example.com/ep001.mp3"
    assert ep1.duration == 5400.0  # 1:30:00
    assert "first episode" in ep1.description.lower()

    ep2 = episodes[1]
    assert ep2.guid == "ep-002"
    assert ep2.audio_url == ""
    assert ep2.duration == 0.0


# ──────────────────────────────────────────────
# build_episode_context
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# youtube_id — explicit provenance + legacy bridge
# ──────────────────────────────────────────────


def test_episode_meta_roundtrips_youtube_id(tmp_path):
    ep = RSSEpisode(
        guid="pqIcoskUuWs",
        title="CNN lecture",
        pub_date="2026-01-01",
        youtube_id="pqIcoskUuWs",
    )
    save_episode_meta(tmp_path, ep)
    loaded = load_episode_meta(tmp_path)
    assert loaded.youtube_id == "pqIcoskUuWs"


def test_legacy_meta_without_field_bridges_youtube_guid(tmp_path):
    import json

    # A pre-field .episode_meta.json: video-id guid, no enclosure, no youtube_id key.
    (tmp_path / ".episode_meta.json").write_text(
        json.dumps({"guid": "pqIcoskUuWs", "title": "Old YT ep", "pub_date": ""}),
        encoding="utf-8",
    )
    loaded = load_episode_meta(tmp_path)
    assert loaded.youtube_id == "pqIcoskUuWs"


def test_legacy_meta_rss_guid_not_bridged(tmp_path):
    import json

    # Pre-field RSS meta: URL guid + enclosure → must NOT invent a youtube_id.
    (tmp_path / ".episode_meta.json").write_text(
        json.dumps(
            {
                "guid": "http://example.com/ep.mp3",
                "title": "RSS ep",
                "pub_date": "",
                "audio_url": "https://cdn.example/ep.mp3",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_episode_meta(tmp_path)
    assert loaded.youtube_id == ""


def test_new_meta_with_explicit_empty_field_not_bridged(tmp_path):
    # A post-field RSS episode whose guid happens to look like a video id:
    # the explicit (empty) youtube_id must win over any guid-shape inference.
    ep = RSSEpisode(guid="abcdefghij_", title="Odd guid", pub_date="")
    save_episode_meta(tmp_path, ep)
    loaded = load_episode_meta(tmp_path)
    assert loaded.youtube_id == ""


# ──────────────────────────────────────────────
# download_audio: partial writes
# ──────────────────────────────────────────────


def test_download_audio_never_leaves_a_truncated_file(tmp_path, monkeypatch):
    """A crash mid-stream must not leave a {stem}.mp3 that is_downloaded then
    reports as 'exists'. The bytes land in a .part sibling until the rename."""
    import httpx

    from podcodex.ingest import rss as rss_mod

    class _Resp:
        def raise_for_status(self):
            pass

        def iter_bytes(self, chunk_size=0):
            yield b"partial"
            raise httpx.ReadTimeout("connection dropped")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(httpx, "stream", lambda *a, **k: _Resp())
    monkeypatch.setattr("time.sleep", lambda _s: None)

    ep = RSSEpisode(
        guid="abc",
        title="Ep One",
        pub_date="2026-01-01",
        audio_url="https://example.com/ep.mp3",
    )
    path, error = rss_mod.download_audio(ep, tmp_path)

    assert path is None
    assert error == "timeout"
    assert not list(tmp_path.glob("*.mp3"))
    assert not list(tmp_path.glob("*.part"))


# ──────────────────────────────────────────────
# fill_empty_fields
# ──────────────────────────────────────────────
#
# The single facility for episode-meta merges. Three sites used to roll their
# own and drift on which keys count; a sparse .episode_meta.json silently
# breaks the RAG date filters, so _BACKFILL_FIELDS membership is pinned here.


def _episode(**overrides) -> RSSEpisode:
    base = dict(guid="g", title="Ep One", pub_date="")
    base.update(overrides)
    return RSSEpisode(**base)


RICH_VALUES = {
    "pub_date": "2026-01-01T00:00:00Z",
    "duration": 1800.0,
    "artwork_url": "https://example.com/art.jpg",
    "episode_number": 12,
    "season_number": 3,
    "youtube_id": "abc123",
}


def test_backfill_field_list_is_the_documented_one():
    """Adding a field here means every merge site starts copying it."""
    assert set(_BACKFILL_FIELDS) == set(RICH_VALUES)
    assert "pub_date" in _BACKFILL_FIELDS  # the RAG date filters ride on it
    assert "description" not in _BACKFILL_FIELDS  # its rule is caller-specific


@pytest.mark.parametrize("field", sorted(RICH_VALUES))
def test_empty_target_field_is_filled_from_the_source(field):
    target = _episode()
    source = _episode(**RICH_VALUES)
    changed = fill_empty_fields(target, source)
    assert field in changed
    assert getattr(target, field) == RICH_VALUES[field]


@pytest.mark.parametrize("field", sorted(RICH_VALUES))
def test_non_empty_target_field_is_kept(field):
    kept = {
        "pub_date": "1999-12-31T00:00:00Z",
        "duration": 60.0,
        "artwork_url": "https://example.com/mine.jpg",
        "episode_number": 1,
        "season_number": 1,
        "youtube_id": "mine",
    }[field]
    target = _episode(**{field: kept})
    source = _episode(**RICH_VALUES)
    changed = fill_empty_fields(target, source)
    assert field not in changed
    assert getattr(target, field) == kept


def test_zero_and_blank_count_as_empty():
    """0 duration / 0 episode number / whitespace-only strings are 'missing'."""
    target = _episode(duration=0.0, episode_number=0, artwork_url="   ")
    source = _episode(duration=42.0, episode_number=7, artwork_url="https://a/b.jpg")
    changed = fill_empty_fields(target, source)
    assert target.duration == 42.0
    assert target.episode_number == 7
    assert target.artwork_url == "https://a/b.jpg"
    assert set(changed) >= {"duration", "episode_number", "artwork_url"}


def test_empty_source_leaves_the_target_alone():
    target = _episode(**RICH_VALUES)
    changed = fill_empty_fields(target, _episode())
    assert changed == []
    for field, value in RICH_VALUES.items():
        assert getattr(target, field) == value


def test_description_fills_only_when_empty_by_default():
    """RSS refetch: a hand-enriched description must survive the merge."""
    target = _episode(description="mine")
    changed = fill_empty_fields(target, _episode(description="a much longer one"))
    assert changed == []
    assert target.description == "mine"

    blank = _episode(description="  ")
    changed = fill_empty_fields(blank, _episode(description="from the feed"))
    assert changed == ["description"]
    assert blank.description == "from the feed"


def test_prefer_longer_description_adopts_the_richer_text():
    """YouTube subtitle enrichment: the full extract beats the flat one."""
    target = _episode(description="short")
    changed = fill_empty_fields(
        target,
        _episode(description="a considerably longer description"),
        prefer_longer_description=True,
    )
    assert changed == ["description"]
    assert target.description == "a considerably longer description"


def test_prefer_longer_description_keeps_the_longer_target():
    target = _episode(description="a considerably longer description")
    changed = fill_empty_fields(
        target, _episode(description="short"), prefer_longer_description=True
    )
    assert changed == []
    assert target.description == "a considerably longer description"


def test_prefer_longer_description_ignores_a_blank_source():
    target = _episode(description="")
    changed = fill_empty_fields(
        target, _episode(description="   "), prefer_longer_description=True
    )
    assert changed == []
    assert target.description == ""


def test_removed_and_feed_order_are_never_backfilled():
    """``removed`` is live-feed state and ``feed_order`` is per-fetch."""
    target = _episode(removed=True, feed_order=None)
    fill_empty_fields(target, _episode(removed=False, feed_order=5))
    assert target.removed is True
    assert target.feed_order is None
