"""Tests for podcodex.bot — pure functions only (no Discord, no Qdrant)."""

import pytest

pytest.importorskip("discord")

from podcodex.bot.config import BotConfig
from podcodex.bot.bot import (
    ServerSettings,
    _fmt_time,
    _speaker,
    _result_embed,
    _score_bar,
    _format_context,
)


# ──────────────────────────────────────────────
# BotConfig
# ──────────────────────────────────────────────


def test_botconfig_defaults():
    cfg = BotConfig(token="tok")
    assert cfg.top_k == 5
    assert cfg.qdrant_url is None


def test_botconfig_custom():
    cfg = BotConfig(token="tok", top_k=3, qdrant_url="http://qdrant:6333")
    assert cfg.top_k == 3
    assert cfg.qdrant_url == "http://qdrant:6333"


# ──────────────────────────────────────────────
# ServerSettings
# ──────────────────────────────────────────────


def test_guild_settings_defaults():
    g = ServerSettings()
    assert g.top_k == 5


def test_guild_settings_custom():
    g = ServerSettings(top_k=3)
    assert g.top_k == 3


# ──────────────────────────────────────────────
# _fmt_time
# ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00"),
        (59, "00:59"),
        (60, "01:00"),
        (90, "01:30"),
        (3599, "59:59"),
        (3600, "01:00:00"),
        (3661, "01:01:01"),
        (7322, "02:02:02"),
    ],
)
def test_fmt_time(seconds, expected):
    assert _fmt_time(seconds) == expected


# ──────────────────────────────────────────────
# _speaker
# ──────────────────────────────────────────────


def test_speaker_uses_speaker_field():
    assert _speaker({"speaker": "Alice"}) == "Alice"


def test_speaker_falls_back_to_dominant():
    assert _speaker({"dominant_speaker": "Bob"}) == "Bob"


def test_speaker_prefers_speaker_over_dominant():
    assert _speaker({"speaker": "Alice", "dominant_speaker": "Bob"}) == "Alice"


def test_speaker_unknown_when_missing():
    assert _speaker({}) == "Unknown"


def test_speaker_unknown_when_none():
    assert _speaker({"speaker": None, "dominant_speaker": None}) == "Unknown"


# ──────────────────────────────────────────────
# _score_bar
# ──────────────────────────────────────────────


def test_score_bar_full():
    assert _score_bar(1.0) == "████████"


def test_score_bar_empty():
    assert _score_bar(0.0) == "░░░░░░░░"


def test_score_bar_half():
    bar = _score_bar(0.5)
    assert len(bar) == 8
    assert "█" in bar and "░" in bar


# ──────────────────────────────────────────────
# _result_embed
# ──────────────────────────────────────────────

_CHUNK = {
    "show": "My Podcast",
    "episode": "ep01",
    "speaker": "Alice",
    "start": 83.0,
    "end": 102.5,
    "text": "The composer came in on day one.",
    "score": 0.87,
}


def test_result_embed_show_as_author_episode_as_title():
    embed, _ = _result_embed(_CHUNK, rank=1, collection="my_podcast", label="α=0.50")
    assert embed.author.name == "🎙 My Podcast"
    assert embed.title == "ep01"


def test_result_embed_footer_has_rank_and_label():
    embed, _ = _result_embed(_CHUNK, rank=2, collection="col", label="exact / BM25")
    assert "#2" in embed.footer.text
    assert "exact / BM25" in embed.footer.text


def test_result_embed_description_has_text():
    embed, _ = _result_embed(_CHUNK, rank=1, collection="col", label="α=0.50")
    assert "The composer came in on day one." in embed.description


def test_result_embed_fields():
    embed, _ = _result_embed(_CHUNK, rank=1, collection="col", label="α=0.50")
    fields = {f.name: f.value for f in embed.fields}
    assert fields["Speaker"] == "Alice"
    assert "01:23" in fields["Timestamp"]
    assert "01:42" in fields["Timestamp"]
    assert "87%" in fields["Relevance"]


def test_result_embed_no_show_has_no_author():
    chunk = {**_CHUNK, "show": ""}
    embed, _ = _result_embed(chunk, rank=1, collection="col", label="α=0.50")
    assert embed.author.name is None


# ──────────────────────────────────────────────
# _format_context
# ──────────────────────────────────────────────

_NEIGHBORS = [
    {"speaker": "Alice", "start": 0.0, "end": 5.0, "text": "First turn"},
    {"speaker": "Bob", "start": 5.0, "end": 10.0, "text": "Second turn"},
    {"speaker": "Alice", "start": 10.0, "end": 15.0, "text": "Third turn — matched"},
    {"speaker": "Bob", "start": 15.0, "end": 20.0, "text": "Fourth turn"},
    {"speaker": "Alice", "start": 20.0, "end": 25.0, "text": "Fifth turn"},
]


def test_format_context_highlights_matched_chunk():
    content, _ = _format_context(_NEIGHBORS, start=10.0, n=2, show="S", episode="E")
    assert "▶ Alice" in content
    assert "**Third turn — matched**" in content


def test_format_context_includes_surrounding_turns():
    content, _ = _format_context(_NEIGHBORS, start=10.0, n=2, show="S", episode="E")
    assert "First turn" in content
    assert "Second turn" in content
    assert "Fourth turn" in content
    assert "Fifth turn" in content


def test_format_context_has_more_when_window_smaller_than_episode():
    _, has_more = _format_context(_NEIGHBORS, start=10.0, n=1, show="S", episode="E")
    assert has_more is True


def test_format_context_no_more_at_episode_boundary():
    # n=2 covers all 5 chunks (2 before + matched + 2 after), nothing beyond
    _, has_more = _format_context(_NEIGHBORS, start=10.0, n=2, show="S", episode="E")
    assert has_more is False


def test_format_context_chunk_not_found():
    content, has_more = _format_context(
        _NEIGHBORS, start=99.0, n=2, show="S", episode="E"
    )
    assert "Could not locate" in content
    assert has_more is False


def test_format_context_header_shows_show_and_episode():
    content, _ = _format_context(
        _NEIGHBORS, start=10.0, n=2, show="My Podcast", episode="ep01"
    )
    assert "My Podcast" in content
    assert "ep01" in content


def test_result_embed_returns_expand_view():
    import discord

    _, view = _result_embed(_CHUNK, rank=1, collection="col", label="α=0.50")
    assert isinstance(view, discord.ui.View)
    buttons = [c for c in view.children if isinstance(c, discord.ui.Button)]
    assert len(buttons) == 1
    assert "context" in buttons[0].label.lower()
