"""Tests for podcodex.rag.chunker — pure logic only (no model loading)."""

from unittest.mock import MagicMock, patch


from podcodex.rag.chunker import (
    _build_episode_text,
    _map_offsets_to_metadata,
    semantic_chunks,
    speaker_chunks,
)

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

_TRANSCRIPT = {
    "meta": {"show": "My Show", "episode": "Episode 1"},
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "Alice",
            "text": "Hello world, this is a test.",
        },
        {
            "start": 5.5,
            "end": 10.0,
            "speaker": "Bob",
            "text": "Hi there, how are you doing today?",
        },
        {"start": 10.5, "end": 11.0, "speaker": "Alice", "text": "..."},  # noise
        {
            "start": 11.5,
            "end": 15.0,
            "speaker": "Alice",
            "text": "I am doing very well, thank you.",
        },
    ],
}


# ──────────────────────────────────────────────
# _build_episode_text
# ──────────────────────────────────────────────


def test_build_episode_text_concatenates_with_spaces():
    segs = [
        {"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "Hello"},
        {"start": 1.0, "end": 2.0, "speaker": "Bob", "text": "World"},
    ]
    full, offset_map = _build_episode_text(segs)
    assert full == "Hello World"
    assert len(offset_map) == 2


def test_build_episode_text_offsets_are_correct():
    segs = [
        {"start": 0.0, "end": 1.0, "speaker": "Alice", "text": "Hello"},
        {"start": 1.0, "end": 2.0, "speaker": "Bob", "text": "World"},
    ]
    _, offset_map = _build_episode_text(segs)
    assert offset_map[0]["start_char"] == 0
    assert offset_map[0]["end_char"] == 5  # len("Hello")
    assert offset_map[1]["start_char"] == 6  # 5 + 1 space
    assert offset_map[1]["end_char"] == 11


# ──────────────────────────────────────────────
# _map_offsets_to_metadata
# ──────────────────────────────────────────────


def test_map_offsets_to_metadata_single_turn():
    offset_map = [
        {
            "start_char": 0,
            "end_char": 10,
            "speaker": "Alice",
            "start": 0.0,
            "end": 5.0,
            "text": "Hello text",
        },
    ]
    meta = _map_offsets_to_metadata(0, 10, offset_map)
    assert meta is not None
    assert meta["dominant_speaker"] == "Alice"
    assert meta["start"] == 0.0
    assert meta["end"] == 5.0
    assert len(meta["speakers"]) == 1


def test_map_offsets_to_metadata_dominant_speaker():
    # Alice has 8 chars, Bob has 2 chars → Alice dominates
    offset_map = [
        {
            "start_char": 0,
            "end_char": 8,
            "speaker": "Alice",
            "start": 0.0,
            "end": 4.0,
            "text": "AAAAAAAA",
        },
        {
            "start_char": 9,
            "end_char": 11,
            "speaker": "Bob",
            "start": 4.0,
            "end": 5.0,
            "text": "BB",
        },
    ]
    meta = _map_offsets_to_metadata(0, 11, offset_map)
    assert meta["dominant_speaker"] == "Alice"


def test_map_offsets_to_metadata_no_overlap_returns_none():
    offset_map = [
        {
            "start_char": 100,
            "end_char": 200,
            "speaker": "Alice",
            "start": 0.0,
            "end": 5.0,
            "text": "x",
        },
    ]
    assert _map_offsets_to_metadata(0, 50, offset_map) is None


# ──────────────────────────────────────────────
# semantic_chunks — mocked (no model loading)
# ──────────────────────────────────────────────


def _make_mock_chunk(text: str, start: int, end: int, token_count: int = 10):
    c = MagicMock()
    c.text = text
    c.start_index = start
    c.end_index = end
    c.token_count = token_count
    return c


def _mock_chonkie():
    """Return a context manager that mocks the chonkie module in sys.modules."""
    mock_mod = MagicMock()
    return patch.dict("sys.modules", {"chonkie": mock_mod}), mock_mod


def test_semantic_chunks_returns_expected_fields():
    transcript = {
        "meta": {"show": "S", "episode": "E"},
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "Alice",
                "text": "Hello world this is a longer sentence for testing.",
            },
            {
                "start": 5.5,
                "end": 10.0,
                "speaker": "Bob",
                "text": "And here is another sentence with more words.",
            },
        ],
    }
    full_text = "Hello world this is a longer sentence for testing. And here is another sentence with more words."
    mock_chunk = _make_mock_chunk(full_text, 0, len(full_text), token_count=20)

    ctx, mock_mod = _mock_chonkie()
    mock_mod.SemanticChunker.return_value.chunk.return_value = [mock_chunk]

    with ctx:
        result = semantic_chunks(transcript, chunk_size=256)

    assert len(result) == 1
    c = result[0]
    assert c["show"] == "S"
    assert c["episode"] == "E"
    assert c["token_count"] == 20
    assert "dominant_speaker" in c
    assert "speakers" in c
    assert "start" in c and "end" in c


def test_semantic_chunks_filters_noise_before_chunking():
    transcript = {
        "meta": {},
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "A",
                "text": "...",
            },  # noise, filtered
            {
                "start": 1.0,
                "end": 5.0,
                "speaker": "A",
                "text": "This is a valid sentence that is long enough.",
            },
        ],
    }
    ctx, mock_mod = _mock_chonkie()
    instance = mock_mod.SemanticChunker.return_value
    instance.chunk.return_value = []

    with ctx:
        semantic_chunks(transcript, min_chars=30)

    called_text = instance.chunk.call_args[0][0]
    assert "..." not in called_text


# ──────────────────────────────────────────────
# speaker_chunks
# ──────────────────────────────────────────────


def test_speaker_chunks_filters_noise():
    chunks = speaker_chunks(_TRANSCRIPT, min_chars=30)
    texts = [c["text"] for c in chunks]
    assert "..." not in texts


def test_speaker_chunks_attaches_metadata():
    chunks = speaker_chunks(_TRANSCRIPT, min_chars=5)
    for c in chunks:
        assert c["show"] == "My Show"
        assert c["episode"] == "Episode 1"
        assert "start" in c and "end" in c and "speaker" in c and "text" in c


def test_speaker_chunks_empty_transcript():
    assert speaker_chunks({"meta": {}, "segments": []}) == []


def test_speaker_chunks_all_noise_filtered():
    t = {
        "meta": {},
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "A", "text": "..."}],
    }
    assert speaker_chunks(t, min_chars=30) == []


def test_speaker_chunks_missing_meta_defaults_empty_strings():
    t = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "A",
                "text": "Hello world this is fine",
            }
        ]
    }
    chunks = speaker_chunks(t, min_chars=5)
    assert chunks[0]["show"] == ""
    assert chunks[0]["episode"] == ""


# ──────────────────────────────────────────────
# semantic_chunks — mocked (no model loading)
# ──────────────────────────────────────────────


def test_semantic_chunks_empty_after_filter_returns_empty():
    transcript = {
        "meta": {},
        "segments": [{"start": 0.0, "end": 1.0, "speaker": "A", "text": ".."}],
    }
    ctx, _ = _mock_chonkie()
    with ctx:
        result = semantic_chunks(transcript, min_chars=30)
    assert result == []


# ──────────────────────────────────────────────
# Source field propagation
# ──────────────────────────────────────────────


def test_speaker_chunks_include_source_field():
    t = {
        "meta": {"show": "S", "episode": "E", "source": "polished"},
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "A",
                "text": "Hello world this is fine enough.",
            },
        ],
    }
    chunks = speaker_chunks(t, min_chars=5)
    assert len(chunks) == 1
    assert chunks[0]["source"] == "polished"


def test_speaker_chunks_source_defaults_empty():
    t = {
        "meta": {"show": "S", "episode": "E"},
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "A",
                "text": "Hello world this is fine enough.",
            },
        ],
    }
    chunks = speaker_chunks(t, min_chars=5)
    assert chunks[0]["source"] == ""


def test_semantic_chunks_include_source_field():
    transcript = {
        "meta": {"show": "S", "episode": "E", "source": "english"},
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "speaker": "Alice",
                "text": "Hello world this is a longer sentence for testing.",
            },
        ],
    }
    full_text = "Hello world this is a longer sentence for testing."
    mock_chunk = _make_mock_chunk(full_text, 0, len(full_text), token_count=10)

    ctx, mock_mod = _mock_chonkie()
    mock_mod.SemanticChunker.return_value.chunk.return_value = [mock_chunk]

    with ctx:
        result = semantic_chunks(transcript, chunk_size=256)

    assert len(result) == 1
    assert result[0]["source"] == "english"
