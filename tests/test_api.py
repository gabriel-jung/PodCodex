"""Smoke tests for the FastAPI backend.

Covers the highest-traffic, lowest-dependency routes:
- /api/health, /api/system/extras (no filesystem)
- /api/shows/* CRUD (config + meta round-trip)
- /api/transcribe/segments GET/PUT (versioned editor endpoint)
- /api/export/text,srt,vtt (segment formatters)
- _helpers.read_segments / load_segments_or_404 / is_flagged (pure)

All tests isolate state by redirecting CONFIG_PATH and operating in tmp_path.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcodex.api.app import create_app
from podcodex.core.versions import save_version


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI TestClient with an isolated config file."""
    from podcodex.api.routes import config as config_mod

    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(config_mod, "CONFIG_PATH", cfg_path)

    app = create_app()
    return TestClient(app)


# ──────────────────────────────────────────────
# Health & capabilities
# ──────────────────────────────────────────────


def test_health_returns_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "capabilities" in body
    assert isinstance(body["capabilities"], dict)


def test_extras_lists_known_extras(client):
    r = client.get("/api/system/extras")
    assert r.status_code == 200
    body = r.json()
    assert "extras" in body
    # At minimum, these four should always be listed
    assert set(body["extras"].keys()) >= {"pipeline", "rag", "bot", "youtube"}
    for ext in body["extras"].values():
        assert "description" in ext
        assert "installed" in ext


# ──────────────────────────────────────────────
# Shows CRUD (register → list → meta round-trip)
# ──────────────────────────────────────────────


def test_register_and_list_shows(client, tmp_path):
    show_dir = tmp_path / "myshow"
    show_dir.mkdir()

    r = client.post("/api/shows/register", json={"path": str(show_dir)})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/api/shows/")
    assert r.status_code == 200
    shows = r.json()
    assert len(shows) == 1
    assert shows[0]["path"] == str(show_dir.resolve())
    assert shows[0]["name"] == "myshow"


def test_register_rejects_missing_folder(client, tmp_path):
    missing = tmp_path / "does_not_exist"
    r = client.post("/api/shows/register", json={"path": str(missing)})
    assert r.status_code == 400


def test_show_meta_round_trip(client, tmp_path):
    show_dir = tmp_path / "show"
    show_dir.mkdir()
    client.post("/api/shows/register", json={"path": str(show_dir)})

    # Default meta for a new show: name derived from folder.
    r = client.get(f"/api/shows/{show_dir}/meta")
    assert r.status_code == 200
    default = r.json()
    assert default["name"] == "show"

    # Update and read back.
    updated = {
        "name": "My Podcast",
        "rss_url": "https://example.com/rss",
        "youtube_url": "",
        "language": "English",
        "speakers": [],
        "artwork_url": "https://example.com/art.jpg",
        "pipeline": {
            "model_size": "large-v3",
            "diarize": True,
            "llm_mode": "ollama",
            "llm_provider": "",
            "llm_model": "qwen3:4b",
            "target_lang": "",
        },
    }
    r = client.put(f"/api/shows/{show_dir}/meta", json=updated)
    assert r.status_code == 200

    r = client.get(f"/api/shows/{show_dir}/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "My Podcast"
    assert body["rss_url"] == "https://example.com/rss"
    assert body["pipeline"]["model_size"] == "large-v3"
    assert body["pipeline"]["llm_model"] == "qwen3:4b"


def test_get_meta_missing_show_returns_404(client):
    r = client.get("/api/shows/nonexistent/meta")
    assert r.status_code == 404


# ──────────────────────────────────────────────
# Transcript segments GET/PUT (editor endpoint)
# ──────────────────────────────────────────────


def _make_audio_dir(tmp_path) -> tuple[str, str]:
    """Create a stub audio file + per-episode output dir, return (audio_path, ep_dir)."""
    show = tmp_path / "s"
    show.mkdir()
    audio = show / "ep.mp3"
    audio.touch()
    ep_dir = show / "ep"
    ep_dir.mkdir()
    return str(audio), str(ep_dir)


def test_get_transcript_segments_404_when_missing(client, tmp_path):
    audio, _ = _make_audio_dir(tmp_path)
    r = client.get("/api/transcribe/segments", params={"audio_path": audio})
    assert r.status_code == 404


def test_get_transcript_segments_reads_legacy_file(client, tmp_path):
    """When no version exists, the route falls back to {stem}.transcript.json."""
    audio, ep_dir = _make_audio_dir(tmp_path)
    segs = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "hello world"},
        {"speaker": "Bob", "start": 2.0, "end": 4.0, "text": "hi there"},
    ]
    (tmp_path / "s" / "ep" / "ep.transcript.json").write_text(json.dumps(segs))

    r = client.get("/api/transcribe/segments", params={"audio_path": audio})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["text"] == "hello world"
    # annotate_flags adds a "flagged" key
    assert "flagged" in body[0]


# ──────────────────────────────────────────────
# Export endpoints (pure formatters)
# ──────────────────────────────────────────────


def test_export_text_from_transcript(client, tmp_path):
    audio, ep_dir = _make_audio_dir(tmp_path)
    segs = [
        {"speaker": "Alice", "start": 0.0, "end": 2.0, "text": "hello"},
        {"speaker": "Bob", "start": 2.0, "end": 4.0, "text": "world"},
    ]
    save_version(
        Path(ep_dir) / "ep", "transcript", segs, {"step": "transcript", "type": "raw"}
    )

    r = client.get(
        "/api/export/text",
        params={"audio_path": audio, "source": "transcript"},
    )
    assert r.status_code == 200
    assert "Alice" in r.text
    assert "hello" in r.text
    assert "Bob" in r.text


def test_export_srt_has_timestamps(client, tmp_path):
    audio, ep_dir = _make_audio_dir(tmp_path)
    segs = [{"speaker": "A", "start": 0.0, "end": 1.5, "text": "go"}]
    save_version(
        Path(ep_dir) / "ep", "transcript", segs, {"step": "transcript", "type": "raw"}
    )

    r = client.get(
        "/api/export/srt",
        params={"audio_path": audio, "source": "transcript"},
    )
    assert r.status_code == 200
    assert "-->" in r.text
    assert "00:00:00,000" in r.text


def test_export_vtt_has_header(client, tmp_path):
    audio, ep_dir = _make_audio_dir(tmp_path)
    segs = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hey"}]
    save_version(
        Path(ep_dir) / "ep", "transcript", segs, {"step": "transcript", "type": "raw"}
    )

    r = client.get(
        "/api/export/vtt",
        params={"audio_path": audio, "source": "transcript"},
    )
    assert r.status_code == 200
    assert r.text.startswith("WEBVTT")


def test_export_missing_source_returns_404(client, tmp_path):
    audio, _ = _make_audio_dir(tmp_path)
    r = client.get(
        "/api/export/text",
        params={"audio_path": audio, "source": "transcript"},
    )
    assert r.status_code == 404


# ──────────────────────────────────────────────
# _helpers pure functions
# ──────────────────────────────────────────────


def test_read_segments_plain_array(tmp_path):
    from podcodex.api.routes._helpers import read_segments

    p = tmp_path / "segs.json"
    segs = [{"speaker": "A", "text": "x", "start": 0, "end": 1}]
    p.write_text(json.dumps(segs))
    assert read_segments(p) == segs


def test_read_segments_wrapped_format(tmp_path):
    from podcodex.api.routes._helpers import read_segments

    p = tmp_path / "segs.json"
    wrapped = {"meta": {"show": "x"}, "segments": [{"text": "hi"}]}
    p.write_text(json.dumps(wrapped))
    assert read_segments(p) == [{"text": "hi"}]


def test_read_segments_missing_file(tmp_path):
    from podcodex.api.routes._helpers import read_segments

    assert read_segments(tmp_path / "nope.json") is None


def test_read_segments_corrupt_returns_none(tmp_path):
    from podcodex.api.routes._helpers import read_segments

    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert read_segments(p) is None


def test_is_flagged_break_not_flagged():
    from podcodex.api.routes._helpers import is_flagged

    assert is_flagged({"speaker": "[BREAK]", "text": "", "start": 0, "end": 5}) is False


def test_is_flagged_unknown_speaker():
    from podcodex.api.routes._helpers import is_flagged

    assert (
        is_flagged({"speaker": "UNKNOWN", "text": "hi", "start": 0, "end": 1}) is True
    )


def test_is_flagged_low_density():
    from podcodex.api.routes._helpers import is_flagged

    # 3 chars over 5s = 0.6 chars/s, below threshold of 2
    assert is_flagged({"speaker": "A", "text": "hmm", "start": 0, "end": 5}) is True


def test_is_flagged_normal_segment():
    from podcodex.api.routes._helpers import is_flagged

    assert (
        is_flagged(
            {"speaker": "A", "text": "This is a normal sentence.", "start": 0, "end": 2}
        )
        is False
    )


def test_build_provenance_shape():
    from podcodex.api.routes._helpers import build_provenance

    prov = build_provenance("transcript", ptype="validated", model="large-v3")
    assert prov["step"] == "transcript"
    assert prov["type"] == "validated"
    assert prov["model"] == "large-v3"
    assert prov["params"] == {}
    assert prov["manual_edit"] is False
