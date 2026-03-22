"""Tests for podcodex.ingest.importer — transcript import and position computation."""

import json

from podcodex.ingest.importer import _compute_positions, import_transcript
from podcodex.ingest.rss import RSSEpisode


# ──────────────────────────────────────────────
# _compute_positions
# ──────────────────────────────────────────────


def test_compute_positions_adds_percentages():
    segments = [
        {"speaker": "A", "text": "a" * 25},  # 25 chars = 25%
        {"speaker": "B", "text": "b" * 75},  # 75 chars = 75%
    ]
    _compute_positions(segments)

    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 25.0
    assert segments[1]["start"] == 25.0
    assert segments[1]["end"] == 100.0


def test_compute_positions_skips_when_real_timestamps():
    segments = [
        {"speaker": "A", "text": "hello", "start": 1.0, "end": 5.0},
        {"speaker": "B", "text": "world", "start": 5.0, "end": 10.0},
    ]
    _compute_positions(segments)

    # Should not be modified
    assert segments[0]["start"] == 1.0
    assert segments[0]["end"] == 5.0


def test_compute_positions_empty():
    segments: list[dict] = []
    _compute_positions(segments)  # should not crash


# ──────────────────────────────────────────────
# import_transcript
# ──────────────────────────────────────────────


def test_import_standard_format(tmp_path):
    transcript = {
        "meta": {"speakers": ["Alice", "Bob"]},
        "segments": [
            {"speaker": "Alice", "text": "Hello", "start": 0.0, "end": 3.0},
            {"speaker": "Bob", "text": "Hi there", "start": 3.0, "end": 6.0},
        ],
    }
    src = tmp_path / "source.json"
    src.write_text(json.dumps(transcript))

    show_folder = tmp_path / "myshow"
    dest = import_transcript(src, show_folder, "ep01", "My Show")

    assert dest.exists()
    assert dest.parent.name == "ep01"
    data = json.loads(dest.read_text())
    assert data["meta"]["show"] == "My Show"
    assert data["meta"]["episode"] == "ep01"
    assert data["meta"]["source"] == "imported"
    assert data["meta"]["timed"] is True


def test_import_bare_list_format(tmp_path):
    segments = [
        {"speaker": "A", "text": "Segment one", "start": 0.0, "end": 5.0},
    ]
    src = tmp_path / "bare.json"
    src.write_text(json.dumps(segments))

    show_folder = tmp_path / "show"
    dest = import_transcript(src, show_folder, "ep", "Show")

    data = json.loads(dest.read_text())
    assert "meta" in data
    assert "segments" in data


def test_import_without_timestamps(tmp_path):
    transcript = {
        "meta": {},
        "segments": [
            {"speaker": "A", "text": "a" * 50},
            {"speaker": "B", "text": "b" * 50},
        ],
    }
    src = tmp_path / "notimed.json"
    src.write_text(json.dumps(transcript))

    show_folder = tmp_path / "show"
    dest = import_transcript(src, show_folder, "ep", "Show")

    data = json.loads(dest.read_text())
    assert data["meta"]["timed"] is False
    # Positions should be set
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][0]["end"] == 50.0
    assert data["segments"][1]["start"] == 50.0
    assert data["segments"][1]["end"] == 100.0


def test_import_with_rss_metadata(tmp_path):
    transcript = {
        "meta": {},
        "segments": [{"speaker": "A", "text": "Hello", "start": 0.0, "end": 1.0}],
    }
    src = tmp_path / "ep.json"
    src.write_text(json.dumps(transcript))

    rss = RSSEpisode(
        guid="123",
        title="Great Episode",
        pub_date="2026-01-15",
        description="An amazing episode.",
        duration=3600.0,
    )
    show_folder = tmp_path / "show"
    dest = import_transcript(src, show_folder, "ep", "Show", rss_episode=rss)

    data = json.loads(dest.read_text())
    assert data["meta"]["rss_title"] == "Great Episode"
    assert data["meta"]["rss_pub_date"] == "2026-01-15"
    assert data["meta"]["rss_description"] == "An amazing episode."
    assert data["meta"]["rss_duration"] == 3600.0


def test_import_creates_output_dir(tmp_path):
    transcript = {"meta": {}, "segments": []}
    src = tmp_path / "ep.json"
    src.write_text(json.dumps(transcript))

    show_folder = tmp_path / "new_show"
    dest = import_transcript(src, show_folder, "ep01", "New Show")

    assert (show_folder / "ep01").is_dir()
    assert dest.exists()
