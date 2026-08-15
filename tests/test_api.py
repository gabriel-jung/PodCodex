"""Smoke tests for the FastAPI backend.

Covers the highest-traffic, lowest-dependency routes:
- /api/health, /api/system/extras (no filesystem)
- /api/shows/* CRUD (config + meta round-trip)
- /api/transcribe/segments GET/PUT (versioned editor endpoint)
- /api/export/text,srt,vtt (segment formatters)
- _helpers.load_segments_or_404 / is_flagged (pure)

All tests isolate state by redirecting CONFIG_PATH and operating in tmp_path.
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from podcodex.core.versions import save_version


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI TestClient with an isolated config file (see fixtures/api_client)."""
    from tests.fixtures.api_client import make_client

    return make_client(tmp_path, monkeypatch)


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


def test_drives_includes_resolved_home(client):
    r = client.get("/api/fs/drives")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["drives"], list)
    assert body["home"] == str(Path.home())


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
            "llm_models_by_mode": {"ollama": "qwen3:4b"},
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
    assert body["pipeline"]["llm_models_by_mode"] == {"ollama": "qwen3:4b"}


def test_broadcast_pattern_round_trip(client, tmp_path):
    show_dir = tmp_path / "show"
    show_dir.mkdir()
    client.post("/api/shows/register", json={"path": str(show_dir)})

    updated = {
        "name": "Total Trax",
        "broadcast_number_pattern": r"\((\d+)\)",
        "speakers": [],
        "pipeline": {},
    }
    r = client.put(f"/api/shows/{show_dir}/meta", json=updated)
    assert r.status_code == 200

    r = client.get(f"/api/shows/{show_dir}/meta")
    assert r.json()["broadcast_number_pattern"] == r"\((\d+)\)"


def test_broadcast_preview(client, tmp_path):
    from podcodex.ingest.rss import RSSEpisode, save_feed_cache

    show_dir = tmp_path / "show"
    show_dir.mkdir()
    client.post("/api/shows/register", json={"path": str(show_dir)})
    save_feed_cache(
        show_dir,
        [
            RSSEpisode(guid="b", title="(252) John Powell", pub_date="", feed_order=0),
            RSSEpisode(guid="a", title="(251) Older", pub_date="", feed_order=1),
        ],
    )

    # Newest episode title drives the preview; pattern extracts its number.
    r = client.get(
        f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": r"\((\d+)\)"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "(252) John Powell"
    assert body["number"] == 252
    assert body["error"] is None

    # Invalid regex surfaces an error instead of raising.
    r = client.get(f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": "("})
    assert r.status_code == 200
    assert r.json()["error"]
    assert r.json()["number"] is None

    # Empty pattern: title still returned, no number, no error.
    r = client.get(f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": ""})
    assert r.json()["number"] is None
    assert r.json()["error"] is None

    # Valid pattern with no capture group: explicit error, not a false
    # "no match" (the regex matched; it just captures nothing).
    r = client.get(
        f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": r"\d+"}
    )
    assert "capture group" in (r.json()["error"] or "")


def test_broadcast_preview_invalid_pattern_without_title(client, tmp_path):
    """An invalid regex must be surfaced even when the show has no titled
    episode (no feed cache): otherwise it autosaves silently."""
    show_dir = tmp_path / "show"
    show_dir.mkdir()
    client.post("/api/shows/register", json={"path": str(show_dir)})
    r = client.get(f"/api/shows/{show_dir}/broadcast-preview", params={"pattern": "("})
    assert r.status_code == 200
    body = r.json()
    assert body["title"] is None
    assert body["error"]


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


# ──────────────────────────────────────────────
# Episode listing / status poll
# ──────────────────────────────────────────────


def test_status_matches_unified_status_fields(client, tmp_path):
    """The status poll must report exactly what /unified reports.

    They share `_build_status_out`; this pins the contract so a field added to
    one payload can't silently skip the other and leave the polled UI stale.
    """
    from podcodex.api.schemas import EpisodeStatusOut
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    _, ep_dir = _make_audio_dir(tmp_path)
    show_dir = Path(ep_dir).parent
    save_version(
        Path(ep_dir) / "ep",
        "transcript",
        [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi"}],
        {"step": "transcript", "type": "raw", "model": "small"},
    )
    get_pipeline_db(show_dir).mark("ep", transcribed=True)
    client.post("/api/shows/register", json={"path": str(show_dir)})

    unified = client.get(f"/api/shows/{show_dir}/unified")
    assert unified.status_code == 200, unified.text
    status = client.get(f"/api/shows/{show_dir}/status")
    assert status.status_code == 200, status.text

    status_keys = set(EpisodeStatusOut.model_fields)
    by_stem = {row["stem"]: row for row in status.json()}
    assert by_stem, "status poll returned no episodes"
    assert by_stem["ep"]["transcribed"] is True
    assert by_stem["ep"]["downloaded"] is True
    for ep in unified.json():
        if ep["stem"] is None:
            continue
        expected = {k: v for k, v in ep.items() if k in status_keys}
        assert by_stem[ep["stem"]] == expected

    close_pipeline_db(show_dir)


def _age_dirs(root: Path, seconds: float) -> None:
    """Push mtimes of *root* and every directory under it into the past.

    The file scan ignores cached mtimes younger than a couple of seconds
    (coarse-timestamp filesystems can hide a same-tick write), so a test that
    wants the cache-hit path has to age the tree first.
    """
    stamp = time.time() - seconds
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        os.utime(d, (stamp, stamp))


def _make_episode_tree(tmp_path) -> Path:
    show = tmp_path / "show"
    (show / "ep" / "transcript").mkdir(parents=True)
    (show / "ep" / "ep.vtt").touch()
    (show / "ep" / "transcript" / "v1.json").touch()
    return show


def test_status_reconcile_keeps_flags_bootstrapped_from_disk(tmp_path):
    """A DB built from a filesystem scan must survive the reconcile pass.

    `populate_from_scan` derives transcribed/synthesized from the step
    directories and writes no `versions` rows, so reconciling against rows
    alone would undo the bootstrap in the same call, and `POST /resync`
    (which deletes the DB file) would report a whole library as not started.
    """
    from podcodex.api.routes.shows import _load_status_context
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    show = tmp_path / "show"
    (show / "ep1" / "transcript").mkdir(parents=True)
    (show / "ep1" / "transcript" / "20260101T000000000000Z_raw.json").write_text(
        '[{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi"}]'
    )

    ctx = _load_status_context(show, None)
    assert ctx.status_map["ep1"]["transcribed"] is True
    # And it must not have been persisted as False either.
    assert get_pipeline_db(show).get_episode("ep1")["transcribed"] is True
    close_pipeline_db(show)


def test_status_reconcile_demotes_when_nothing_is_left(tmp_path):
    """The demote half still fires when neither a row nor a file remains."""
    from podcodex.api.routes.shows import _load_status_context
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    show = tmp_path / "show"
    (show / "ep1").mkdir(parents=True)
    get_pipeline_db(show).mark("ep1", transcribed=True)

    ctx = _load_status_context(show, None)
    assert ctx.status_map["ep1"]["transcribed"] is False
    close_pipeline_db(show)


def test_episode_file_scan_sees_nested_writes(tmp_path):
    """Caching is keyed on the whole recorded directory tree, not just the top.

    A version landing in `ep/transcript/` leaves `ep/`'s own mtime untouched,
    so validating only the episode directory would serve a stale file list.
    """
    from podcodex.api.routes.shows import _scan_episode_files

    show = _make_episode_tree(tmp_path)
    _age_dirs(show, 60)
    assert _scan_episode_files(show, {})["ep"] == [
        "ep/ep.vtt",
        "ep/transcript/v1.json",
    ]

    versions = show / "ep" / "transcript"
    (versions / "v2.json").touch()
    # Age only the subdirectory: `show/` and `ep/` keep the mtimes just
    # recorded, so nothing but the nested change can trip the cache.
    stamp = time.time() - 30
    os.utime(versions, (stamp, stamp))
    assert _scan_episode_files(show, {})["ep"] == [
        "ep/ep.vtt",
        "ep/transcript/v1.json",
        "ep/transcript/v2.json",
    ]

    os.remove(versions / "v1.json")
    stamp = time.time() - 20
    os.utime(versions, (stamp, stamp))
    assert _scan_episode_files(show, {})["ep"] == ["ep/ep.vtt", "ep/transcript/v2.json"]

    # A removed episode directory drops out of the result and the cache.
    shutil.rmtree(show / "ep")
    assert _scan_episode_files(show, {}) == {}


def test_episode_file_scan_reuses_settled_results(tmp_path, monkeypatch):
    """An untouched, settled tree is served from cache instead of re-walked."""
    from podcodex.api.routes import shows as shows_routes

    show = _make_episode_tree(tmp_path)
    _age_dirs(show, 60)
    shows_routes._scan_episode_files(show, {})  # records settled mtimes

    def _fail(*args):
        raise AssertionError("settled directories should not be re-walked")

    monkeypatch.setattr(shows_routes, "_walk_episode_dir", _fail)
    assert shows_routes._scan_episode_files(show, {})["ep"] == [
        "ep/ep.vtt",
        "ep/transcript/v1.json",
    ]


def test_episode_file_scan_rewalks_recent_changes(tmp_path):
    """A just-written directory is never trusted, whatever its mtime says.

    FAT32 rounds mtimes to 2s, so a write can land in the same tick as the
    scan that recorded the mtime and leave it looking unchanged.
    """
    from podcodex.api.routes.shows import _scan_episode_files

    show = _make_episode_tree(tmp_path)
    _scan_episode_files(show, {})

    versions = show / "ep" / "transcript"
    recorded = os.stat(versions).st_mtime
    (versions / "v2.json").touch()
    os.utime(versions, (recorded, recorded))  # simulate a same-tick write

    assert "ep/transcript/v2.json" in _scan_episode_files(show, {})["ep"]


# ──────────────────────────────────────────────
# Verified pointer endpoint
# ──────────────────────────────────────────────


def test_verified_set_and_clear(client, tmp_path):
    from podcodex.core.pipeline_db import get_pipeline_db, close_pipeline_db

    audio, ep_dir = _make_audio_dir(tmp_path)
    segs = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi"}]
    vid = save_version(
        Path(ep_dir) / "ep",
        "corrected",
        segs,
        {"step": "corrected", "type": "raw", "model": "x"},
    )
    # The pipeline routes always populate an episodes row before save_version
    # via their own provenance writes; the bare save_version in this test
    # leaves the episode unregistered, so we mark it manually to match the
    # production invariant the endpoint relies on.
    show_dir = Path(ep_dir).parent
    get_pipeline_db(show_dir).mark("ep", transcribed=True, corrected=True)

    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": "corrected", "version_id": vid},
    )
    assert r.status_code == 200, r.text
    assert r.json()["verified"] == {"step": "corrected", "version_id": vid}

    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": None, "version_id": None},
    )
    assert r.status_code == 200
    assert r.json()["verified"] is None
    close_pipeline_db(show_dir)


def test_verified_rejects_unregistered_episode(client, tmp_path):
    """Endpoint must refuse to materialize an episode row for an unknown stem."""
    audio, ep_dir = _make_audio_dir(tmp_path)
    segs = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi"}]
    vid = save_version(
        Path(ep_dir) / "ep",
        "corrected",
        segs,
        {"step": "corrected", "type": "raw", "model": "x"},
    )
    # No mark() / populate_from_scan: episode row absent in pipeline_db.
    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": "corrected", "version_id": vid},
    )
    assert r.status_code == 404
    assert "not registered" in r.text


def test_verified_rejects_invalid_step(client, tmp_path):
    audio, _ = _make_audio_dir(tmp_path)
    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": "english", "version_id": "v-1"},
    )
    assert r.status_code == 400
    assert "transcript" in r.text or "corrected" in r.text


def test_episode_speakers_airtime(client, tmp_path):
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    _, ep_dir = _make_audio_dir(tmp_path)
    show_dir = Path(ep_dir).parent
    # Alice 15s, Bob 4s, a [BREAK] gap and an empty-speaker segment (both skipped).
    segs = [
        {"speaker": "Alice", "start": 0.0, "end": 10.0, "text": "a"},
        {"speaker": "Bob", "start": 10.0, "end": 14.0, "text": "b"},
        {"speaker": "[BREAK]", "start": 14.0, "end": 20.0, "text": ""},
        {"speaker": "Alice", "start": 20.0, "end": 25.0, "text": "a"},
        {"speaker": "", "start": 25.0, "end": 26.0, "text": "?"},
    ]
    save_version(
        Path(ep_dir) / "ep",
        "corrected",
        segs,
        {"step": "corrected", "type": "raw", "model": "x"},
    )
    get_pipeline_db(show_dir).mark("ep", transcribed=True, corrected=True)
    client.post("/api/shows/register", json={"path": str(show_dir)})

    r = client.get(f"/api/shows/{show_dir}/episode/ep/speakers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_transcript"] is True
    assert body["episode_seconds"] == 26.0  # last segment end (no audio duration)
    names = [s["name"] for s in body["speakers"]]
    assert names == ["Alice", "Bob"]  # sorted by airtime desc, gaps excluded
    alice = body["speakers"][0]
    assert alice["total_seconds"] == 15.0
    assert round(alice["pct"], 1) == round(15.0 / 26.0 * 100, 1)
    # Music/gap time is unattributed, so shares sum to under 100%.
    assert sum(s["pct"] for s in body["speakers"]) < 100
    close_pipeline_db(show_dir)


def test_episode_speakers_no_transcript(client, tmp_path):
    _, ep_dir = _make_audio_dir(tmp_path)
    show_dir = Path(ep_dir).parent
    client.post("/api/shows/register", json={"path": str(show_dir)})
    r = client.get(f"/api/shows/{show_dir}/episode/ep/speakers")
    assert r.status_code == 200
    body = r.json()
    assert body["has_transcript"] is False
    assert body["speakers"] == []


def test_verified_pointer_wins_for_speakers(client, tmp_path):
    """The verified version, not the latest, feeds both the episode speaker
    endpoint and the show roster, and they agree."""
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    _, ep_dir = _make_audio_dir(tmp_path)
    show_dir = Path(ep_dir).parent
    base = Path(ep_dir) / "ep"
    tv = save_version(
        base,
        "transcript",
        [{"speaker": "OldGuest", "start": 0.0, "end": 10.0, "text": "t"}],
        {"step": "transcript", "type": "raw", "model": "x"},
    )
    save_version(
        base,
        "corrected",
        [{"speaker": "NewHost", "start": 0.0, "end": 10.0, "text": "c"}],
        {"step": "corrected", "type": "raw", "model": "x"},
    )
    db = get_pipeline_db(show_dir)
    db.mark("ep", transcribed=True, corrected=True)
    client.post("/api/shows/register", json={"path": str(show_dir)})

    # No verified pointer: canonical is the corrected version.
    r = client.get(f"/api/shows/{show_dir}/episode/ep/speakers")
    assert [s["name"] for s in r.json()["speakers"]] == ["NewHost"]

    # Verify the older transcript version: it must now win everywhere.
    db.set_verified("ep", "transcript", tv)
    r = client.get(f"/api/shows/{show_dir}/episode/ep/speakers")
    assert [s["name"] for s in r.json()["speakers"]] == ["OldGuest"]

    roster = client.get(f"/api/shows/{show_dir}/speakers/roster").json()
    names = {sp["name"] for sp in roster["speakers"]}
    assert "OldGuest" in names and "NewHost" not in names
    close_pipeline_db(show_dir)


def test_verified_rejects_missing_version(client, tmp_path):
    audio, _ = _make_audio_dir(tmp_path)
    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": "corrected", "version_id": "nonexistent"},
    )
    assert r.status_code == 404


def test_verified_rejects_partial_body(client, tmp_path):
    audio, _ = _make_audio_dir(tmp_path)
    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": "corrected"},
    )
    assert r.status_code == 400


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
# Search routes (query/exact/random via podcodex.rag.search_service)
# ──────────────────────────────────────────────

SEARCH_DIM = 8


@pytest.fixture
def seeded_index(tmp_path, monkeypatch):
    """IndexStore with show "Alpha" indexed only under e5-small/semantic.

    The API's request default is bge-m3/semantic, which this fixture never
    creates for Alpha. Route tests use this gap to confirm the resolver
    chain falls through to Alpha's actual collection instead of querying a
    collection that doesn't exist.
    """
    from podcodex.rag import index_store as rag_index_store
    from podcodex.rag import retriever as rag_retriever
    from podcodex.rag.index_store import IndexStore

    index_path = tmp_path / "search-index"
    store = IndexStore(index_path)
    col = "alpha__e5-small__semantic"
    store.ensure_collection(
        col, show="Alpha", model="e5-small", chunker="semantic", dim=SEARCH_DIM
    )
    chunks = [
        {
            "text": f"hello world chunk {i}",
            "episode": "ep1",
            "show": "Alpha",
            "source": "transcript",
            "dominant_speaker": "Alice",
            "start": float(i),
            "end": float(i + 1),
        }
        for i in range(3)
    ]
    rng = np.random.default_rng(0)
    store.save_chunks(col, "ep1", chunks, rng.random((3, SEARCH_DIM), dtype=np.float32))

    monkeypatch.setenv("PODCODEX_INDEX", str(index_path))
    rag_index_store.get_index_store.cache_clear()
    rag_retriever.get_retriever.cache_clear()
    # Stub the embedder so the fallback resolves against e5-small without
    # pulling live model weights; only encode_query is on the query path.
    retriever = rag_retriever.get_retriever("e5-small")
    monkeypatch.setattr(
        retriever, "encode_query", lambda _q: np.zeros(SEARCH_DIM, dtype=np.float32)
    )
    yield store
    rag_index_store.get_index_store.cache_clear()
    rag_retriever.get_retriever.cache_clear()


def test_search_falls_back_when_requested_combo_missing(client, seeded_index):
    """Requesting a model/chunking combo a show doesn't have must still
    return the show's actual results, not an empty/404-ish response."""
    resp = client.post(
        "/api/search/query",
        json={"query": "hello", "show": "Alpha", "model": "bge-m3"},
        headers={"X-PodCodex": "1"},
    )
    assert resp.status_code == 200
    assert resp.json()  # old code: empty (queried a nonexistent collection)


def test_exact_endpoint_falls_back_when_requested_combo_missing(client, seeded_index):
    """/exact must resolve through the same fallback chain as /query: a
    show indexed only under a non-default model must still be searchable."""
    resp = client.post(
        "/api/search/exact",
        json={"query": "hello world chunk 1", "show": "Alpha", "model": "bge-m3"},
        headers={"X-PodCodex": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body
    assert any("hello world chunk 1" in r["text"] for r in body)
    result = body[0]
    assert "episode" in result
    assert "episode_stem" in result
    assert "score" in result
    assert "match_text" in result


def test_random_endpoint_falls_back_when_requested_combo_missing(client, seeded_index):
    """/random must resolve through the same fallback chain as the other
    search routes instead of silently returning None for an indexed show."""
    resp = client.post(
        "/api/search/random",
        json={"show": "Alpha", "model": "bge-m3"},
        headers={"X-PodCodex": "1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body is not None
    assert body["score"] == 1.0
    assert body["text"]


def test_speakers_falls_back_when_requested_combo_missing(client, seeded_index):
    """/speakers must resolve through the same fallback chain as the other
    search routes: a wrong model param must not silently return []."""
    resp = client.get(
        "/api/search/speakers",
        params={"show": "Alpha", "model": "bge-m3"},
        headers={"X-PodCodex": "1"},
    )
    assert resp.status_code == 200
    assert resp.json() == ["Alice"]


# ──────────────────────────────────────────────
# _helpers pure functions
# ──────────────────────────────────────────────


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
