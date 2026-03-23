"""Tests for podcodex.bot — pure functions only (no Discord, no Qdrant)."""

import pytest

pytest.importorskip("discord")

from podcodex.bot.bot import BotConfig, ServerSettings, _result_embed
from podcodex.bot.formatting import (
    CooldownManager,
    build_compact_embed,
    fmt_time as _fmt_time,
    merge_results,
    safe_truncate,
    speaker as _speaker,
    score_bar as _score_bar,
    format_context as _format_context,
)


# ──────────────────────────────────────────────
# BotConfig
# ──────────────────────────────────────────────


def test_botconfig_defaults():
    cfg = BotConfig()
    assert cfg.top_k == 5
    assert cfg.qdrant_url is None
    assert cfg.chunker == "semantic"


def test_botconfig_custom():
    cfg = BotConfig(top_k=3, qdrant_url="http://qdrant:6333", chunker="speaker")
    assert cfg.top_k == 3
    assert cfg.qdrant_url == "http://qdrant:6333"
    assert cfg.chunker == "speaker"


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


def test_result_embed_show_in_title_query_as_author():
    embed, _ = _result_embed(
        _CHUNK,
        rank=1,
        total=5,
        collection="my_podcast",
        label="α=0.50",
        question="film music",
    )
    assert embed.author.name == '🔎 "film music"'
    assert "ep01" in embed.title
    assert "My Podcast" in embed.title


def test_result_embed_footer_has_rank_and_label():
    embed, _ = _result_embed(
        _CHUNK, rank=2, total=5, collection="col", label="exact / BM25"
    )
    assert "#2" in embed.footer.text
    assert "exact / BM25" in embed.footer.text


def test_result_embed_description_has_text():
    embed, _ = _result_embed(_CHUNK, rank=1, total=5, collection="col", label="α=0.50")
    assert "The composer came in on day one." in embed.description


def test_result_embed_fields():
    embed, _ = _result_embed(_CHUNK, rank=1, total=5, collection="col", label="α=0.50")
    fields = {f.name: f.value for f in embed.fields}
    assert "01:23" in fields["Timestamp"]
    assert "01:42" in fields["Timestamp"]
    assert "87%" in fields["Relevance"]


def test_result_embed_no_show_has_no_author():
    chunk = {**_CHUNK, "show": ""}
    embed, _ = _result_embed(chunk, rank=1, total=5, collection="col", label="α=0.50")
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

    _, view = _result_embed(_CHUNK, rank=1, total=5, collection="col", label="α=0.50")
    assert isinstance(view, discord.ui.View)
    buttons = [c for c in view.children if isinstance(c, discord.ui.Button)]
    assert len(buttons) == 1
    assert "context" in buttons[0].label.lower()


# ──────────────────────────────────────────────
# safe_truncate
# ──────────────────────────────────────────────


def test_safe_truncate_short_text_unchanged():
    text, truncated = safe_truncate("hello world", max_chars=100)
    assert text == "hello world"
    assert truncated is False


def test_safe_truncate_exact_limit_unchanged():
    text = "a" * 50
    result, truncated = safe_truncate(text, max_chars=50)
    assert result == text
    assert truncated is False


def test_safe_truncate_cuts_at_word_boundary():
    text = "hello world foo bar"
    result, truncated = safe_truncate(text, max_chars=12)
    assert truncated is True
    assert "hello world" in result
    assert "foo" not in result


def test_safe_truncate_adds_truncation_marker():
    text = "word " * 100
    result, truncated = safe_truncate(text, max_chars=20)
    assert truncated is True
    assert "…(truncated)" in result


def test_safe_truncate_no_spaces_cuts_at_max():
    text = "a" * 100
    result, truncated = safe_truncate(text, max_chars=50)
    assert truncated is True
    assert len(result.split("\n")[0]) == 50


# ──────────────────────────────────────────────
# merge_results
# ──────────────────────────────────────────────


def _hits(scores: list[float]) -> list[dict]:
    return [{"text": f"t{i}", "score": s} for i, s in enumerate(scores)]


def test_merge_score_strategy_sorts_globally():
    hits_by_col = {
        "a": _hits([0.9, 0.5]),
        "b": _hits([0.8, 0.6]),
    }
    merged = merge_results(hits_by_col, top_k=4, strategy="score")
    scores = [c.get("score") for c, _ in merged]
    assert scores == [0.9, 0.8, 0.6, 0.5]


def test_merge_score_strategy_respects_top_k():
    hits_by_col = {"a": _hits([0.9, 0.8, 0.7])}
    merged = merge_results(hits_by_col, top_k=2, strategy="score")
    assert len(merged) == 2


def test_merge_roundrobin_interleaves():
    hits_by_col = {
        "a": _hits([0.9, 0.7]),
        "b": _hits([0.8, 0.6]),
    }
    merged = merge_results(hits_by_col, top_k=4, strategy="roundrobin")
    collections = [col for _, col in merged]
    # Round-robin alternates between collections
    assert collections[0] != collections[1]


def test_merge_roundrobin_respects_top_k():
    hits_by_col = {
        "a": _hits([0.9, 0.7, 0.5]),
        "b": _hits([0.8, 0.6, 0.4]),
    }
    merged = merge_results(hits_by_col, top_k=3, strategy="roundrobin")
    assert len(merged) == 3


def test_merge_roundrobin_uneven_collections():
    hits_by_col = {
        "a": _hits([0.9]),
        "b": _hits([0.8, 0.6, 0.4]),
    }
    merged = merge_results(hits_by_col, top_k=4, strategy="roundrobin")
    assert len(merged) == 4
    # "a" exhausted after 1, remaining come from "b"
    assert sum(1 for _, col in merged if col == "a") == 1
    assert sum(1 for _, col in merged if col == "b") == 3


def test_merge_empty_input():
    assert merge_results({}, top_k=5, strategy="score") == []
    assert merge_results({}, top_k=5, strategy="roundrobin") == []


def test_merge_single_collection():
    hits_by_col = {"a": _hits([0.9, 0.5])}
    merged = merge_results(hits_by_col, top_k=5, strategy="roundrobin")
    assert len(merged) == 2
    assert all(col == "a" for _, col in merged)


# ──────────────────────────────────────────────
# CooldownManager
# ──────────────────────────────────────────────


def test_cooldown_allows_first_request():
    cm = CooldownManager(seconds=5.0)
    assert cm.check(123) == 0.0


def test_cooldown_blocks_after_consume():
    cm = CooldownManager(seconds=5.0)
    cm.consume(123)
    remaining = cm.check(123)
    assert remaining > 0.0


def test_cooldown_independent_per_user():
    cm = CooldownManager(seconds=5.0)
    cm.consume(111)
    assert cm.check(222) == 0.0


def test_cooldown_zero_seconds_never_blocks():
    cm = CooldownManager(seconds=0.0)
    cm.consume(123)
    assert cm.check(123) == 0.0


# ──────────────────────────────────────────────
# ServerSettings — new fields + backwards compat
# ──────────────────────────────────────────────


def test_server_settings_new_fields_default():
    s = ServerSettings()
    assert s.allowed_shows == []
    assert s.default_source == ""
    assert s.compact is False


def test_server_settings_with_new_fields():
    s = ServerSettings(
        allowed_shows=["Show A"], default_source="polished", compact=True
    )
    assert s.allowed_shows == ["Show A"]
    assert s.default_source == "polished"
    assert s.compact is True


def test_server_settings_backwards_compat_ignores_unknown_keys():
    """Old config files may have extra keys; construction should not crash."""
    raw = {"model": "bge-m3", "chunker": "semantic", "top_k": 5, "unknown_field": 42}
    import dataclasses

    valid_keys = {f.name for f in dataclasses.fields(ServerSettings)}
    s = ServerSettings(**{k: v for k, v in raw.items() if k in valid_keys})
    assert s.model == "bge-m3"


def test_server_settings_backwards_compat_missing_new_keys():
    """Old config files won't have new fields; defaults should fill in."""
    raw = {"model": "bge-m3", "chunker": "semantic", "top_k": 5}
    s = ServerSettings(**raw)
    assert s.allowed_shows == []
    assert s.default_source == ""
    assert s.compact is False


# ──────────────────────────────────────────────
# build_compact_embed
# ──────────────────────────────────────────────

_COMPACT_RESULTS = [
    (
        {
            "show": "Podcast A",
            "episode": "ep01",
            "speaker": "Alice",
            "start": 60.0,
            "end": 90.0,
            "text": "This is a test result.",
            "score": 0.85,
        },
        "podcast_a__bge-m3__semantic",
    ),
    (
        {
            "show": "Podcast B",
            "episode": "ep02",
            "speaker": "Bob",
            "start": 120.0,
            "end": 150.0,
            "text": "Another test result here.",
            "score": 0.72,
        },
        "podcast_b__bge-m3__semantic",
    ),
]


def test_compact_embed_returns_single_embed():
    embed = build_compact_embed(_COMPACT_RESULTS, "α=0.50 • BGE-M3")
    assert embed.title == "🔎 α=0.50 • BGE-M3"
    assert len(embed.fields) == 2


def test_compact_embed_field_names_have_rank_and_episode():
    embed = build_compact_embed(_COMPACT_RESULTS, "test")
    assert "#1" in embed.fields[0].name
    assert "ep01" in embed.fields[0].name
    assert "Podcast A" in embed.fields[0].name


def test_compact_embed_field_values_have_speaker_and_score():
    embed = build_compact_embed(_COMPACT_RESULTS, "test")
    assert "Alice" in embed.fields[0].value
    assert "85%" in embed.fields[0].value


def test_compact_embed_max_25_fields():
    many = [(_COMPACT_RESULTS[0][0], "col")] * 30
    embed = build_compact_embed(many, "test")
    assert len(embed.fields) == 25


def test_compact_embed_footer_shows_count():
    embed = build_compact_embed(_COMPACT_RESULTS, "test")
    assert "2 results" in embed.footer.text


def test_compact_embed_truncates_long_text():
    chunk = {**_COMPACT_RESULTS[0][0], "text": "word " * 100}
    embed = build_compact_embed([(chunk, "col")], "test")
    assert "…" in embed.fields[0].value


# ──────────────────────────────────────────────
# _effective_settings preserves new fields
# ──────────────────────────────────────────────


def test_effective_settings_carries_new_fields(tmp_path):
    """_effective_settings must propagate allowed_shows/source/compact from server config."""
    cfg_path = tmp_path / "server_config.json"
    import json

    cfg_path.write_text(
        json.dumps(
            {
                "1": {
                    "model": "bge-m3",
                    "chunker": "semantic",
                    "top_k": 5,
                    "allowed_shows": ["ShowA", "ShowB"],
                    "default_source": "polished",
                    "compact": True,
                }
            }
        )
    )
    from unittest.mock import patch

    with patch("podcodex.bot.bot.QdrantStore"), patch("podcodex.bot.bot.Retriever"):
        from podcodex.bot.bot import BotConfig, PodCodexBot

        bot = PodCodexBot(BotConfig(), server_config_path=cfg_path)
    eff = bot._effective_settings(guild_id=1, model="", top_k=0)
    assert eff.allowed_shows == ["ShowA", "ShowB"]
    assert eff.default_source == "polished"
    assert eff.compact is True
    # Per-query model override should still work
    eff2 = bot._effective_settings(guild_id=1, model="e5-small", top_k=10)
    assert eff2.model == "e5-small"
    assert eff2.top_k == 10
    assert eff2.allowed_shows == ["ShowA", "ShowB"]


# ──────────────────────────────────────────────
# _AutocompleteCache
# ──────────────────────────────────────────────


def test_autocomplete_cache_starts_stale():
    from podcodex.bot.bot import _AutocompleteCache

    cache = _AutocompleteCache(episodes={}, sources={}, speakers={})
    assert cache.is_stale() is True


def test_autocomplete_cache_fresh_after_timestamp_set():
    import time
    from podcodex.bot.bot import _AutocompleteCache

    cache = _AutocompleteCache(
        episodes={}, sources={}, speakers={}, timestamp=time.monotonic()
    )
    assert cache.is_stale() is False


def test_autocomplete_cache_stale_after_ttl():
    import time
    from podcodex.bot.bot import _AutocompleteCache

    cache = _AutocompleteCache(
        episodes={},
        sources={},
        speakers={},
        timestamp=time.monotonic() - 301,
        ttl=300.0,
    )
    assert cache.is_stale() is True


# ──────────────────────────────────────────────
# /setup show list mutations
# ──────────────────────────────────────────────


def test_setup_show_add_appends():
    """show_add should append to existing allowed_shows."""
    current = ServerSettings(allowed_shows=["ShowA"])
    new_shows = list(current.allowed_shows)
    show_add = "ShowB"
    if show_add not in new_shows:
        new_shows.append(show_add)
    assert new_shows == ["ShowA", "ShowB"]


def test_setup_show_add_no_duplicate():
    current = ServerSettings(allowed_shows=["ShowA"])
    new_shows = list(current.allowed_shows)
    show_add = "ShowA"
    if show_add not in new_shows:
        new_shows.append(show_add)
    assert new_shows == ["ShowA"]


def test_setup_show_remove():
    current = ServerSettings(allowed_shows=["ShowA", "ShowB"])
    new_shows = list(current.allowed_shows)
    show_remove = "ShowA"
    if show_remove in new_shows:
        new_shows.remove(show_remove)
    assert new_shows == ["ShowB"]


def test_setup_show_clear_then_add():
    """show_clear runs first, then show_add — allows clear-and-add in one call."""
    current = ServerSettings(allowed_shows=["Old1", "Old2"])
    new_shows = list(current.allowed_shows)
    show_clear = True
    show_add = "NewShow"

    if show_clear:
        new_shows = []
    if show_add and show_add not in new_shows:
        new_shows.append(show_add)

    assert new_shows == ["NewShow"]
