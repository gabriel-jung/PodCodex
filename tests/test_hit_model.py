"""Tests for podcodex.rag.hit, the typed RAG read boundary.

Covers:
- chunker conformance: every key the chunkers emit is declared on ``Hit``
  (``model_extra`` empty), so write-side schema drift fails here first
- legacy on-disk indexes: unknown meta keys load into ``model_extra``,
  RFC-2822 ``rss_pub_date`` normalizes, turns without offsets validate
- display properties (title fallback, speaker coalesce, pub date)
- the write-shape roundtrip used by the indexer's cache reuse
"""

import json

import pytest

from podcodex.rag.chunker import semantic_chunks, speaker_chunks
from tests.fixtures.chunker_mocks import make_mock_chunk, mock_chonkie
from podcodex.rag.hit import Hit, SpeakerTurn

_FULL_META = {
    "show": "My Show",
    "episode": "0027_episode_1_pilot",
    "source": "rss",
    "rss_title": "The Pilot",
    "rss_pub_date": "Mon, 15 Jan 2024 12:00:00 GMT",
    "episode_number": 1,
    "broadcast_number": 27,
    "rss_description": "<p>First episode.</p>",
    "rss_artwork_url": "https://example.com/art.jpg",
    "rss_audio_url": "https://example.com/ep1.mp3",
    "youtube_id": "abc123",
    "timed": False,
}

_TRANSCRIPT = {
    "meta": _FULL_META,
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "Alice",
            "text": "Hello world, this is a long enough test segment.",
        },
        {
            "start": 5.5,
            "end": 10.0,
            "speaker": "Bob",
            "text": "Hi there, how are you doing on this fine day?",
        },
    ],
}


# ──────────────────────────────────────────────
# Chunker conformance (write-side drift guard)
# ──────────────────────────────────────────────


def test_speaker_chunks_conform_to_hit():
    chunks = speaker_chunks(_TRANSCRIPT, min_chars=5)
    assert chunks
    for chunk in chunks:
        hit = Hit.model_validate(chunk)
        assert not hit.model_extra, (
            f"chunker emitted undeclared keys {sorted(hit.model_extra)}; "
            "declare them on Hit"
        )


def test_semantic_chunks_conform_to_hit():
    from podcodex.rag.chunker import _SEP

    full_text = _SEP.join(s["text"] for s in _TRANSCRIPT["segments"])
    mock_chunk = make_mock_chunk(full_text, 0, len(full_text), token_count=20)

    ctx, mock_mod = mock_chonkie()
    mock_mod.SemanticChunker.return_value.chunk.return_value = [mock_chunk]
    with ctx:
        chunks = semantic_chunks(_TRANSCRIPT, chunk_size=256, min_chars=5)

    assert chunks
    for chunk in chunks:
        hit = Hit.model_validate(chunk)
        assert not hit.model_extra, (
            f"chunker emitted undeclared keys {sorted(hit.model_extra)}; "
            "declare them on Hit"
        )
        assert hit.speakers is not None
        assert all(isinstance(t, SpeakerTurn) for t in hit.speakers)


def test_chunker_extras_land_on_declared_fields():
    chunks = speaker_chunks(_TRANSCRIPT, min_chars=5)
    hit = Hit.model_validate(chunks[0])
    assert hit.episode_title == "The Pilot"
    assert hit.pub_date == "2024-01-15"
    assert hit.episode_number == 1
    assert hit.broadcast_number == 27
    assert hit.description == "First episode."
    assert hit.artwork_url == "https://example.com/art.jpg"
    assert hit.audio_url == "https://example.com/ep1.mp3"
    assert hit.youtube_id == "abc123"
    assert hit.timed is False


# ──────────────────────────────────────────────
# Legacy index tolerance
# ──────────────────────────────────────────────


def test_legacy_row_with_unknown_meta_keys_loads():
    # Shape _row_to_chunk produces from an old index: meta blob keys merged
    # flat with the fixed columns, including keys this version never writes.
    legacy = {
        "chunk_index": 3,
        "show": "My Show",
        "episode": "0001_old_episode",
        "source": "rss",
        "pub_date": "",
        "dominant_speaker": "Alice",
        "start": 12.0,
        "end": 30.0,
        "text": "old text",
        "rss_pub_date": "Mon, 15 Jan 2024 12:00:00 GMT",
        "legacy_flag": "kept",
        "timed": False,
        "speakers": [{"speaker": "Alice", "text": "old text"}],
    }
    hit = Hit.model_validate(legacy)
    assert hit.model_extra == {"legacy_flag": "kept"}
    assert hit.timed is False
    turn = hit.speakers[0]
    assert turn.speaker == "Alice"
    assert turn.start_char is None


def test_effective_pub_date_normalizes_legacy_rss_date():
    hit = Hit(rss_pub_date="Mon, 15 Jan 2024 12:00:00 GMT")
    assert hit.effective_pub_date == "2024-01-15"


def test_effective_pub_date_prefers_normalized_column():
    hit = Hit(pub_date="2024-02-01", rss_pub_date="Mon, 15 Jan 2024 12:00:00 GMT")
    assert hit.effective_pub_date == "2024-02-01"


def test_effective_pub_date_empty_when_unset():
    assert Hit().effective_pub_date == ""


# ──────────────────────────────────────────────
# Display properties
# ──────────────────────────────────────────────


def test_display_title_prefers_rss_title():
    hit = Hit(episode="0027_episode_1_pilot", episode_title="The Pilot")
    assert hit.display_title == "The Pilot"


def test_display_title_falls_back_to_humanized_stem():
    hit = Hit(episode="0027_some_episode_name")
    assert hit.display_title == "Some episode name"


def test_speaker_label_prefers_turn_speaker():
    # Turn-flattened random hit: the flat speaker must win over dominant.
    hit = Hit(dominant_speaker="Alice", speaker="Bob")
    assert hit.speaker_label == "Bob"


def test_speaker_label_falls_back_to_dominant():
    hit = Hit(dominant_speaker="Alice")
    assert hit.speaker_label == "Alice"


# ──────────────────────────────────────────────
# Write-shape roundtrip (indexer cache reuse)
# ──────────────────────────────────────────────


def test_to_chunk_dict_drops_retrieval_fields_keeps_extras():
    hit = Hit.model_validate(
        {
            "text": "x",
            "show": "S",
            "score": 0.9,
            "accent_match": True,
            "legacy_flag": 1,
        }
    )
    d = hit.to_chunk_dict()
    assert d["text"] == "x" and d["show"] == "S"
    assert "score" not in d and "accent_match" not in d
    assert d["legacy_flag"] == 1


# ──────────────────────────────────────────────
# Legacy meta that cannot validate degrades, never raises
# ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "meta",
    [
        {"description": None, "artwork_url": None},  # explicit nulls
        {"episode_number": "S2E4"},  # un-coercible int
        {"speakers": ["Alice"]},  # turns stored as bare strings
    ],
)
def test_row_to_chunk_degrades_on_bad_legacy_meta(meta):
    from podcodex.rag.index_store import _row_to_chunk

    row = {
        "chunk_index": 3,
        "episode": "ep1",
        "show": "S",
        "text": "t",
        "start": 1.0,
        "end": 2.0,
        "meta": json.dumps(meta),
    }
    hit = _row_to_chunk(row)
    # Fixed columns survive; the unparseable meta is dropped rather than
    # raising and 500-ing every read for the show.
    assert hit.text == "t"
    assert hit.chunk_index == 3
    assert hit.episode == "ep1"


def test_row_to_chunk_keeps_meta_when_a_turn_offset_is_null():
    """A null inside one speaker turn must not discard the row's whole meta
    blob (episode_title, youtube_id, the turn list itself)."""
    from podcodex.rag.index_store import _row_to_chunk

    meta = {
        "episode_title": "The Pilot",
        "youtube_id": "abc123",
        "speakers": [
            {"speaker": "Alice", "start": None, "end": 2.0, "text": "hi"},
        ],
    }
    row = {
        "chunk_index": 0,
        "episode": "ep1",
        "show": "S",
        "text": "hi",
        "start": 0.0,
        "end": 2.0,
        "meta": json.dumps(meta),
    }
    hit = _row_to_chunk(row)
    assert hit.episode_title == "The Pilot"
    assert hit.youtube_id == "abc123"
    assert hit.speakers is not None
    assert hit.speakers[0].speaker == "Alice"
    assert hit.speakers[0].start == 0.0  # field default fills the nulled offset


def test_to_chunk_dict_keeps_write_keys_at_default_values():
    # embedder subscripts chunk["text"]; exclude_defaults must not drop it.
    d = Hit(text="", episode="", show="").to_chunk_dict()
    assert d["text"] == "" and d["episode"] == "" and d["show"] == ""
