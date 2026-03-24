"""Tests for podcodex.ingest.rss — RSS parsing, caching, slug generation."""

from podcodex.ingest.rss import (
    RSSEpisode,
    _parse_duration,
    build_episode_context,
    fetch_feed,
    load_episode_meta,
    load_feed_cache,
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
    feed_file = tmp_path / "feed.xml"
    feed_file.write_text(_SAMPLE_RSS, encoding="utf-8")

    episodes = fetch_feed(str(feed_file))

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


def test_build_episode_context_full(tmp_path):
    ep = RSSEpisode(
        guid="abc",
        title="Solène et la dot au Moyen Âge",
        pub_date="2026-03-15",
        description="In this episode we discuss medieval dowries.",
        episode_number=116,
        season_number=2,
    )
    ep_dir = tmp_path / "ep116"
    save_episode_meta(ep_dir, ep)

    ctx = build_episode_context("Mon Podcast", ep_dir)

    assert "Podcast: Mon Podcast" in ctx
    assert 'Episode: "Solène et la dot au Moyen Âge" (S2E116)' in ctx
    assert "Description: In this episode we discuss medieval dowries." in ctx


def test_build_episode_context_episode_number_only(tmp_path):
    ep = RSSEpisode(guid="abc", title="Test", pub_date="2026-01-01", episode_number=5)
    save_episode_meta(tmp_path, ep)

    ctx = build_episode_context("Show", tmp_path)

    assert "(Episode 5)" in ctx
    assert "(S" not in ctx  # no season prefix


def test_build_episode_context_no_numbers(tmp_path):
    ep = RSSEpisode(guid="abc", title="Bonus: Q&A", pub_date="2026-01-01")
    save_episode_meta(tmp_path, ep)

    ctx = build_episode_context("Show", tmp_path)

    assert 'Episode: "Bonus: Q&A"' in ctx
    assert "(" not in ctx.split("Episode:")[-1]  # no parenthetical


def test_build_episode_context_html_stripped(tmp_path):
    ep = RSSEpisode(
        guid="abc",
        title="Test",
        pub_date="2026-01-01",
        description="<p>Hello <b>world</b>!</p><br/>More text.",
    )
    save_episode_meta(tmp_path, ep)

    ctx = build_episode_context("Show", tmp_path)

    assert "<p>" not in ctx
    assert "<b>" not in ctx
    assert "<br/>" not in ctx
    assert "Hello world!" in ctx
    assert "More text." in ctx


def test_build_episode_context_description_truncated(tmp_path):
    ep = RSSEpisode(
        guid="abc",
        title="Test",
        pub_date="2026-01-01",
        description="word " * 200,  # 1000 chars
    )
    save_episode_meta(tmp_path, ep)

    ctx = build_episode_context("Show", tmp_path, max_desc=50)

    desc_line = [line for line in ctx.splitlines() if line.startswith("Description:")][
        0
    ]
    assert desc_line.endswith("…")
    assert len(desc_line) <= len("Description: ") + 55  # some slack for word boundary


def test_build_episode_context_show_name_only():
    ctx = build_episode_context("My Podcast")

    assert ctx == "Podcast: My Podcast"


def test_build_episode_context_no_metadata(tmp_path):
    ctx = build_episode_context("Show", tmp_path)

    assert ctx == "Podcast: Show"


def test_build_episode_context_empty():
    ctx = build_episode_context()

    assert ctx == ""
