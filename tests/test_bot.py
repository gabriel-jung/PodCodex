"""Tests for podcodex.bot — pure functions only (no Discord)."""

import pytest

pytest.importorskip("discord")

from podcodex.bot.bot import BotConfig, ResolvedShows, ServerSettings, ShowAccess
from podcodex.bot.ui import (
    build_details_embed,
    build_episodes_embeds,
    build_listen_button,
    build_result_embed,
    build_stats_embed,
)
from podcodex.bot.formatting import (
    CooldownManager,
    build_compact_embed,
    display_speaker,
    fmt_duration as _fmt_duration,
    fmt_time as _fmt_time,
    safe_truncate,
    speaker as _speaker,
    score_bar as _score_bar,
)


# ──────────────────────────────────────────────
# BotConfig
# ──────────────────────────────────────────────


def test_botconfig_defaults():
    cfg = BotConfig()
    assert cfg.top_k == 5
    assert cfg.index_path is None
    assert cfg.chunker == "semantic"


def test_botconfig_custom():
    cfg = BotConfig(top_k=3, index_path="/tmp/lance-index", chunker="speaker")
    assert cfg.top_k == 3
    assert cfg.index_path == "/tmp/lance-index"
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


def test_speaker_generic_when_missing():
    assert _speaker({}) == "Speaker"


def test_speaker_generic_when_none():
    assert _speaker({"speaker": None, "dominant_speaker": None}) == "Speaker"


def test_resolve_show_collections_precedence(monkeypatch):
    from podcodex.bot.bot import PodCodexBot, ServerSettings

    col_info = {
        "s__aardvark__semantic": {
            "show": "S",
            "model": "aardvark",
            "chunker": "semantic",
        },
        "s__bge-m3__semantic": {"show": "S", "model": "bge-m3", "chunker": "semantic"},
        "s__e5-small__speaker": {
            "show": "S",
            "model": "e5-small",
            "chunker": "speaker",
        },
    }
    bot = PodCodexBot.__new__(PodCodexBot)  # no Discord login needed
    bot._shows = {}  # nothing protected: _show_allowed always True

    def resolve(settings, prefs, explicit=None, cols=col_info):
        monkeypatch.setattr("podcodex.bot.bot.load_show_rag_prefs", lambda: prefs)
        shows = ResolvedShows(ShowAccess.ALL)
        return bot._resolve_show_collections(shows, settings, cols, explicit=explicit)

    # guild default wins over global default
    got = resolve(ServerSettings(model="e5-small", chunker="speaker"), {})
    assert [(c.name, c.model) for c in got] == [("s__e5-small__speaker", "e5-small")]
    # show.toml pref wins over guild default
    got = resolve(
        ServerSettings(model="bge-m3", chunker="semantic"),
        {"s": ("e5-small", "speaker")},
    )
    assert [(c.name, c.model) for c in got] == [("s__e5-small__speaker", "e5-small")]
    # explicit user pick wins over show pref
    got = resolve(
        ServerSettings(),
        {"s": ("e5-small", "speaker")},
        explicit=("bge-m3", "semantic"),
    )
    assert [(c.name, c.model) for c in got] == [("s__bge-m3__semantic", "bge-m3")]
    # guild default mismatch falls to the global default rung,
    # not to alphabetical-first (aardvark)
    got = resolve(ServerSettings(model="nope", chunker="nope"), {})
    assert [(c.name, c.model) for c in got] == [("s__bge-m3__semantic", "bge-m3")]
    # no tier matches at all (no global-default collection either):
    # first by name keeps the show reachable
    no_default_cols = {
        "s__aardvark__semantic": {
            "show": "S",
            "model": "aardvark",
            "chunker": "semantic",
        },
        "s__e5-small__speaker": {
            "show": "S",
            "model": "e5-small",
            "chunker": "speaker",
        },
    }
    got = resolve(
        ServerSettings(model="nope", chunker="nope"), {}, cols=no_default_cols
    )
    assert [(c.name, c.model) for c in got] == [("s__aardvark__semantic", "aardvark")]


def test_display_speaker_maps_raw_diarization_labels():
    assert display_speaker("SPEAKER_01") == "Speaker 1"
    assert display_speaker("speaker_12") == "Speaker 12"
    assert display_speaker("") == "Speaker"
    assert display_speaker(None) == "Speaker"
    assert display_speaker("Patrick Beja") == "Patrick Beja"


# ──────────────────────────────────────────────
# _score_bar
# ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "score,expected",
    [
        (1.0, "████████"),
        (0.0, "░░░░░░░░"),
    ],
)
def test_score_bar_extremes(score, expected):
    assert _score_bar(score) == expected


def test_score_bar_half_is_mixed():
    bar = _score_bar(0.5)
    assert len(bar) == 8
    assert "█" in bar and "░" in bar


# ──────────────────────────────────────────────
# build_result_embed
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
    embed = build_result_embed(
        _CHUNK,
        rank=1,
        total=5,
        label="α=0.50",
        text="film music",
    )
    # Show moved to the author line; the episode title no longer repeats it.
    assert embed.author.name == "My Podcast"
    assert "Ep01" in embed.title
    assert "My Podcast" not in embed.title


def test_result_embed_footer_is_rank_only():
    embed = build_result_embed(_CHUNK, rank=2, total=5, label="exact / BM25")
    assert embed.footer.text == "2 of 5"
    # The search label is telemetry — it belongs in Details, not the card.
    assert "exact / BM25" not in embed.footer.text


def test_result_embed_description_has_text():
    embed = build_result_embed(_CHUNK, rank=1, total=5, label="α=0.50")
    assert "The composer came in on day one." in embed.description


def test_result_embed_meta_line_in_description_no_engine_fields():
    embed = build_result_embed(_CHUNK, rank=1, total=5, label="α=0.50")
    # Start time rides a quiet meta line in the body, not a field.
    assert "01:23" in embed.description
    names = {f.name for f in embed.fields}
    assert "Relevance" not in names
    assert "Timestamp" not in names


def test_details_embed_carries_the_engine_numbers():
    embed = build_details_embed(_CHUNK, label="α=0.50")
    fields = {f.name: f.value for f in embed.fields}
    assert "87%" in fields["Relevance"]
    assert "01:23" in fields["Full range"]
    assert "01:42" in fields["Full range"]
    assert fields["Search"] == "α=0.50"


def _ep_stats(n: int) -> list[dict]:
    return [
        {
            "episode": f"ep{i}",
            "episode_title": f"Episode {i}",
            "pub_date": "2026-01-08",
            "duration": 60.0,
            "speakers": [],
            "description": "",
        }
        for i in range(n)
    ]


def test_episodes_embeds_sorted_by_number_desc_over_pub_date():
    eps = _ep_stats(3)
    # pub_date order contradicts the numbering; the number must win.
    eps[0].update(broadcast_number=247, episode_title="Ep 247", pub_date="2026-03-03")
    eps[1].update(broadcast_number=249, episode_title="Ep 249", pub_date="2026-01-01")
    eps[2].update(broadcast_number=248, episode_title="Ep 248", pub_date="2026-02-02")
    embeds = build_episodes_embeds("My Show", eps, footer="3 episodes")
    assert [f.name for f in embeds[0].fields] == ["Ep 249", "Ep 248", "Ep 247"]


def test_episodes_embeds_unnumbered_fall_back_to_pub_date_desc():
    eps = _ep_stats(2)
    eps[0].update(episode_title="Older", pub_date="2026-01-01")
    eps[1].update(episode_title="Newer", pub_date="2026-06-01")
    embeds = build_episodes_embeds("My Show", eps, footer="2 episodes")
    assert [f.name for f in embeds[0].fields] == ["Newer", "Older"]


def test_episodes_embeds_use_plain_title_without_number_prefix():
    eps = _ep_stats(1)
    eps[0]["broadcast_number"] = 249
    eps[0]["episode_title"] = "(249) Isabelle Durin : un Violon au Cinéma"
    embeds = build_episodes_embeds("My Show", eps, footer="1 episodes")
    assert embeds[0].fields[0].name == "(249) Isabelle Durin : un Violon au Cinéma"


def test_episodes_embeds_show_full_date_with_day():
    embeds = build_episodes_embeds("My Show", _ep_stats(1), footer="1 episodes")
    assert "8 Jan 2026" in embeds[0].fields[0].value


def test_episodes_embeds_omit_description():
    eps = _ep_stats(1)
    eps[0]["description"] = "A long summary that used to weigh the list down."
    embeds = build_episodes_embeds("My Show", eps, footer="1 episodes")
    assert "summary" not in embeds[0].fields[0].value


def test_episodes_embeds_use_show_artwork():
    embeds = build_episodes_embeds(
        "My Show",
        _ep_stats(1),
        footer="1 episodes",
        artwork_url="https://cdn.example/show.jpg",
    )
    assert len(embeds) == 1
    assert embeds[0].thumbnail.url == "https://cdn.example/show.jpg"


def test_episodes_embeds_without_artwork_have_no_thumbnail():
    embeds = build_episodes_embeds("My Show", _ep_stats(1), footer="1 episodes")
    assert embeds[0].thumbnail.url is None


def test_episodes_embeds_paginate_ten_per_page():
    embeds = build_episodes_embeds(
        "My Show",
        _ep_stats(11),
        footer="11 episodes",
        artwork_url="https://cdn.example/show.jpg",
    )
    assert len(embeds) == 2
    assert len(embeds[0].fields) == 10
    assert len(embeds[1].fields) == 1
    # Every page carries the show artwork and footer.
    assert all(e.thumbnail.url == "https://cdn.example/show.jpg" for e in embeds)
    assert all(e.footer.text == "11 episodes" for e in embeds)


def test_result_embed_footer_carries_match_total_when_given():
    embed = build_result_embed(
        _CHUNK, rank=3, total=1473, label="", footer_extra="2444 matches"
    )
    assert embed.footer.text == "3 of 1473 excerpts · 2444 matches"


def test_result_embed_footer_plain_without_extra():
    embed = build_result_embed(_CHUNK, rank=1, total=5, label="")
    assert embed.footer.text == "1 of 5"


def test_result_embed_sets_thumbnail_from_artwork():
    chunk = {**_CHUNK, "artwork_url": "https://cdn.example/art.jpg"}
    embed = build_result_embed(chunk, rank=1, total=5, label="")
    assert embed.thumbnail.url == "https://cdn.example/art.jpg"
    # No artwork → no thumbnail.
    assert build_result_embed(_CHUNK, rank=1, total=5, label="").thumbnail.url is None


def test_listen_button_youtube_jumps_to_timestamp():
    chunk = {**_CHUNK, "youtube_id": "pqIcoskUuWs", "start": 614.0}
    btn = build_listen_button(chunk)
    assert btn is not None
    assert btn.url == "https://www.youtube.com/watch?v=pqIcoskUuWs&t=614s"
    assert "YouTube" in btn.label


def test_listen_button_rss_links_episode_no_seek():
    chunk = {**_CHUNK, "audio_url": "https://cdn.example/ep.mp3", "start": 614.0}
    btn = build_listen_button(chunk)
    assert btn is not None
    assert btn.url == "https://cdn.example/ep.mp3"  # no timestamp appended
    assert "Listen" in btn.label


def test_listen_button_none_for_local_import():
    assert build_listen_button(_CHUNK) is None


def test_result_embed_no_show_has_no_author():
    chunk = {**_CHUNK, "show": ""}
    embed = build_result_embed(chunk, rank=1, total=5, label="α=0.50")
    assert embed.author.name is None


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
        allowed_shows=["Show A"], default_source="corrected", compact=True
    )
    assert s.allowed_shows == ["Show A"]
    assert s.default_source == "corrected"
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
    assert "Ep01" in embed.fields[0].name
    assert "Podcast A" in embed.fields[0].name


def test_compact_embed_field_values_have_speaker_and_score():
    embed = build_compact_embed(_COMPACT_RESULTS, "test")
    assert "Alice" in embed.fields[0].value
    assert "85%" in embed.fields[0].value


def test_compact_embed_max_25_fields():
    many = [(_COMPACT_RESULTS[0][0], "col")] * 30
    embed = build_compact_embed(many, "test")
    assert len(embed.fields) == 25


def test_compact_embed_stays_under_discord_total_limit():
    # 25 realistic long-text results burst Discord's 6000-char total embed
    # cap (observed: 7637 chars for a 1473-hit /exact); Discord then rejects
    # the message with HTTP 400 and the interaction hangs on "thinking".
    long_chunk = {
        "show": "Total Trax",
        "episode": "21_249_isabelle_durin_un_violon_au_cinema",
        "episode_title": "Isabelle Durin, un violon au cinéma (excerpt)",
        "speaker": "Isabelle",
        "start": 1234.0,
        "end": 1290.0,
        "text": (
            "La musique de film est un art à part entière et la musique "
            "accompagne chaque scène avec une intensité remarquable, la "
            "musique soulignant l'émotion du récit à chaque instant du film "
            "pour le spectateur attentif."
        ),
        "score": 1.0,
    }
    many = [(dict(long_chunk), "col")] * 30
    embed = build_compact_embed(many, "1473 matches", query="musique")
    assert len(embed) <= 6000
    # Trimmed, not emptied: still shows a useful number of rows.
    assert len(embed.fields) >= 5


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
                    "default_source": "corrected",
                    "compact": True,
                }
            }
        )
    )
    from unittest.mock import patch

    with patch("podcodex.bot.bot.IndexStore"), patch("podcodex.bot.bot.Retriever"):
        from podcodex.bot.bot import BotConfig, PodCodexBot

        bot = PodCodexBot(BotConfig(), server_config_path=cfg_path)
    eff = bot._effective_settings(guild_id=1, model="", top_k=0)
    assert eff.allowed_shows == ["ShowA", "ShowB"]
    assert eff.default_source == "corrected"
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

    cache = _AutocompleteCache(episodes={}, episode_titles={}, sources={}, speakers={})
    assert cache.is_stale() is True


def test_autocomplete_cache_fresh_after_timestamp_set():
    import time
    from podcodex.bot.bot import _AutocompleteCache

    cache = _AutocompleteCache(
        episodes={},
        episode_titles={},
        sources={},
        speakers={},
        timestamp=time.monotonic(),
    )
    assert cache.is_stale() is False


def test_autocomplete_cache_stale_after_ttl():
    import time
    from podcodex.bot.bot import _AutocompleteCache

    cache = _AutocompleteCache(
        episodes={},
        episode_titles={},
        sources={},
        speakers={},
        timestamp=time.monotonic() - 301,
        ttl=300.0,
    )
    assert cache.is_stale() is True


# ──────────────────────────────────────────────
# _resolve_show_collections — one collection per show, model-agnostic
# ──────────────────────────────────────────────


def _seed_multimodel_index(tmp_path):
    """Two shows: Alpha under the default model, Beta only under e5-small."""
    import numpy as np

    from podcodex.rag.index_store import IndexStore

    dim = 8
    store = IndexStore(tmp_path / "index")
    for show, model in (("Alpha", "bge-m3"), ("Beta", "e5-small")):
        col = f"{show.lower()}__{model}__semantic"
        store.ensure_collection(
            col, show=show, model=model, chunker="semantic", dim=dim
        )
        chunks = [
            {
                "text": "x",
                "episode": "ep1",
                "show": show,
                "source": "transcript",
                "dominant_speaker": "sp",
                "start": 0.0,
                "end": 1.0,
            }
        ]
        store.save_chunks(
            col,
            "ep1",
            chunks,
            np.random.default_rng(0).random((1, dim), dtype=np.float32),
        )
    return store


def _bot_for(tmp_path, store):
    from podcodex.bot.bot import PodCodexBot

    bot = PodCodexBot(
        BotConfig(index_path=str(tmp_path / "index")),
        server_config_path=tmp_path / "server_config.json",
    )
    _ = bot.local
    return bot


def test_resolve_show_collections_reaches_every_show_one_each(tmp_path):
    from podcodex.bot.bot import ResolvedShows, ShowAccess

    store = _seed_multimodel_index(tmp_path)
    bot = _bot_for(tmp_path, store)
    settings = bot._server_settings(None)
    col_info = bot.local.get_all_collection_info()

    cols = bot._resolve_show_collections(
        ResolvedShows(ShowAccess.ALL), settings, col_info
    )
    # One collection per show, and Beta is reachable under e5-small without a model arg.
    assert sorted((c.name, c.model) for c in cols) == [
        ("alpha__bge-m3__semantic", "bge-m3"),
        ("beta__e5-small__semantic", "e5-small"),
    ]


def test_resolve_show_collections_excludes_locked_show(tmp_path):
    from podcodex.bot.bot import ResolvedShows, ShowAccess

    store = _seed_multimodel_index(tmp_path)
    bot = _bot_for(tmp_path, store)
    store.set_show_password("Beta", "sha256:" + "0" * 64)
    bot._reload_shows()
    settings = bot._server_settings(None)
    col_info = bot.local.get_all_collection_info()

    cols = bot._resolve_show_collections(
        ResolvedShows(ShowAccess.ALL), settings, col_info
    )
    assert [(c.name, c.model) for c in cols] == [("alpha__bge-m3__semantic", "bge-m3")]


# ──────────────────────────────────────────────
# fmt_duration
# ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0m"),
        (59, "1m"),
        (60, "1m"),
        (3540, "59m"),
        (3600, "1h"),
        (3660, "1h 1m"),
        (7322, "2h 2m"),
        (770400, "214h"),
    ],
)
def test_fmt_duration(seconds, expected):
    assert _fmt_duration(seconds) == expected


# ──────────────────────────────────────────────
# build_stats_embed
# ──────────────────────────────────────────────


def _stats_rows(pub_date="2026-01-08", n=2, dur=3600.0) -> list[dict]:
    return [
        {"episode": f"ep{i}", "pub_date": pub_date, "duration": dur} for i in range(n)
    ]


def test_stats_embed_multi_show_totals_line():
    per_show = {"A": _stats_rows(n=2), "B": _stats_rows(n=3)}
    embed = build_stats_embed(per_show, [])
    assert embed.title == "📊 PodCodex Index"
    first = embed.description.splitlines()[0]
    assert first == "2 shows · 5 episodes · 5h"


def test_stats_embed_single_show_is_a_show_card():
    per_show = {"My Show": _stats_rows(pub_date="2026-07-08", n=3, dur=7200.0)}
    embed = build_stats_embed(per_show, [])
    # One show: the card belongs to the show, no index header, no
    # duplicated per-show block.
    assert embed.title == "🎙 My Show"
    assert embed.description.splitlines()[0] == "3 episodes · 6h · newest 8 Jul 2026"
    assert "My Show" not in embed.description
    assert "1 show" not in embed.description


def test_stats_embed_single_show_omits_newest_without_pub_date():
    per_show = {"My Show": _stats_rows(pub_date="", n=1)}
    embed = build_stats_embed(per_show, [])
    assert "newest" not in embed.description
    assert embed.description.splitlines()[0] == "1 episode · 1h"


def test_stats_embed_shows_sorted_newest_first_dateless_last():
    per_show = {
        "Old": _stats_rows(pub_date="2024-05-01"),
        "New": _stats_rows(pub_date="2026-07-08"),
        "Zeta": _stats_rows(pub_date=""),
        "Alpha": _stats_rows(pub_date=""),
    }
    embed = build_stats_embed(per_show, [])
    order = [line for line in embed.description.splitlines() if line.startswith("🎙")]
    assert [line.split("**")[1] for line in order] == ["New", "Old", "Alpha", "Zeta"]


def test_stats_embed_multi_show_line_content():
    per_show = {
        "My Show": _stats_rows(pub_date="2026-07-08", n=3, dur=7200.0),
        "Other": _stats_rows(n=1),
    }
    embed = build_stats_embed(per_show, [])
    assert "🎙 **My Show**" in embed.description
    assert "3 episodes · 6h · newest 8 Jul 2026" in embed.description


def test_stats_embed_speaker_line_top5_with_others():
    speakers = [
        {"speaker": "Olivier", "total_duration": 166320.0},
        {"speaker": "David", "total_duration": 111840.0},
        {"speaker": "Eve", "total_duration": 33120.0},
        {"speaker": "Ana", "total_duration": 7200.0},
        {"speaker": "Bob", "total_duration": 3600.0},
        {"speaker": "X1", "total_duration": 10.0},
        {"speaker": "X2", "total_duration": 10.0},
    ]
    embed = build_stats_embed({"A": _stats_rows()}, speakers)
    assert (
        "🎤 Olivier (46h 12m), David (31h 4m), Eve (9h 12m), "
        "Ana (2h), Bob (1h), and 2 others" in embed.description
    )


def test_stats_embed_speaker_line_singular_other():
    speakers = [
        {"speaker": f"S{i}", "total_duration": 3600.0 * (9 - i)} for i in range(6)
    ]
    embed = build_stats_embed({"A": _stats_rows()}, speakers)
    assert "and 1 other" in embed.description
    assert "others" not in embed.description


def test_stats_embed_speaker_line_no_tail_when_five_or_fewer():
    speakers = [{"speaker": "Solo", "total_duration": 3600.0}]
    embed = build_stats_embed({"A": _stats_rows()}, speakers)
    assert "🎤 Solo (1h)" in embed.description
    assert "other" not in embed.description


def test_stats_embed_omits_speaker_line_without_speakers():
    embed = build_stats_embed({"A": _stats_rows()}, [])
    assert "🎤" not in embed.description


def test_stats_embed_maps_raw_diarization_speaker_labels():
    speakers = [{"speaker": "SPEAKER_01", "total_duration": 3600.0}]
    embed = build_stats_embed({"A": _stats_rows()}, speakers)
    assert "Speaker 1 (1h)" in embed.description


def test_stats_embed_artwork_thumbnail():
    embed = build_stats_embed(
        {"A": _stats_rows()}, [], artwork_url="https://img.example/a.jpg"
    )
    assert embed.thumbnail.url == "https://img.example/a.jpg"


def test_stats_embed_no_thumbnail_without_artwork():
    embed = build_stats_embed({"A": _stats_rows()}, [])
    assert embed.thumbnail.url is None
