"""Episode routes: transcript segments, status poll, verified pointer,
subtitle upload and manual-mode apply."""

from pathlib import Path

import pytest

from podcodex.core.versions import save_version
from tests.fixtures.episodes import audio_and_output_dir


@pytest.fixture
def client(tmp_path, monkeypatch):
    from tests.fixtures.api_client import make_client

    return make_client(tmp_path, monkeypatch)


def test_get_transcript_segments_404_when_missing(client, tmp_path):
    audio, _ = audio_and_output_dir(tmp_path)
    r = client.get("/api/transcribe/segments", params={"audio_path": audio})
    assert r.status_code == 404


def test_status_matches_unified_status_fields(client, tmp_path):
    """The status poll must report exactly what /unified reports.

    They share `_build_status_out`; this pins the contract so a field added to
    one payload can't silently skip the other and leave the polled UI stale.
    """
    from podcodex.api.schemas import EpisodeStatusOut
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    _, ep_dir = audio_and_output_dir(tmp_path)
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


def test_verified_set_and_clear(client, tmp_path):
    from podcodex.core.pipeline_db import get_pipeline_db, close_pipeline_db

    audio, ep_dir = audio_and_output_dir(tmp_path)
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
    audio, ep_dir = audio_and_output_dir(tmp_path)
    segs = [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi"}]
    vid = save_version(
        Path(ep_dir) / "ep",
        "corrected",
        segs,
        {"step": "corrected", "type": "raw", "model": "x"},
    )
    # No mark() / _populate_from_scan: episode row absent in pipeline_db.
    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": "corrected", "version_id": vid},
    )
    assert r.status_code == 404
    assert "not registered" in r.text


def test_verified_rejects_invalid_step(client, tmp_path):
    audio, _ = audio_and_output_dir(tmp_path)
    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": "english", "version_id": "v-1"},
    )
    assert r.status_code == 400
    assert "transcript" in r.text or "corrected" in r.text


def test_episode_speakers_airtime(client, tmp_path):
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    _, ep_dir = audio_and_output_dir(tmp_path)
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
    _, ep_dir = audio_and_output_dir(tmp_path)
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

    _, ep_dir = audio_and_output_dir(tmp_path)
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
    audio, _ = audio_and_output_dir(tmp_path)
    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": "corrected", "version_id": "nonexistent"},
    )
    assert r.status_code == 404


def test_verified_rejects_partial_body(client, tmp_path):
    audio, _ = audio_and_output_dir(tmp_path)
    r = client.put(
        "/api/shows/verified",
        params={"audio_path": audio},
        json={"step": "corrected"},
    )
    assert r.status_code == 400


_SRT_BODY = (
    "1\n00:00:00,000 --> 00:00:02,000\nAlice: Hello there\n\n"
    "2\n00:00:02,000 --> 00:00:04,000\nBob: Café au lait\n"
)


def test_upload_srt_transcript(client, tmp_path):
    """An SRT upload keeps its original beside the transcript, inside the
    episode directory (``p.base`` is a path prefix, not a directory)."""
    audio, ep_dir = audio_and_output_dir(tmp_path)

    r = client.post(
        "/api/transcribe/upload",
        params={"audio_path": audio},
        files={"file": ("ep.srt", _SRT_BODY.encode("utf-8"), "text/plain")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2

    # The reference copy lands beside the version root, not inside it.
    assert (Path(ep_dir) / "ep.subtitles.srt").exists()

    segs = client.get("/api/transcribe/segments", params={"audio_path": audio}).json()
    assert [s["text"] for s in segs] == ["Hello there", "Café au lait"]


def test_upload_srt_non_utf8(client, tmp_path):
    """cp1252 is what Windows subtitle tools emit; a strict decode 500s."""
    audio, _ = audio_and_output_dir(tmp_path)

    r = client.post(
        "/api/transcribe/upload",
        params={"audio_path": audio},
        files={"file": ("ep.srt", _SRT_BODY.encode("cp1252"), "text/plain")},
    )
    assert r.status_code == 200, r.text
    segs = client.get("/api/transcribe/segments", params={"audio_path": audio}).json()
    assert segs[1]["text"] == "Café au lait"


def _seed_transcript(ep_dir: str, n: int = 2) -> None:
    save_version(
        Path(ep_dir) / Path(ep_dir).name,
        "transcript",
        [
            {"speaker": "A", "start": float(i), "end": float(i + 1), "text": f"t{i}"}
            for i in range(n)
        ],
        {"step": "transcript", "type": "raw"},
    )


def test_apply_manual_rejects_a_count_mismatch(client, tmp_path):
    """A paste whose segment count does not match is rejected; accepted, it
    would save untouched source text as a finished step."""
    audio, ep_dir = audio_and_output_dir(tmp_path)
    _seed_transcript(ep_dir, n=3)

    r = client.post(
        "/api/correct/apply-manual",
        json={"audio_path": audio, "corrections": [{"text": "only one"}]},
    )
    assert r.status_code == 400
    assert "mismatch" in r.json()["detail"].lower()

    r = client.post(
        "/api/translate/apply-manual",
        json={
            "audio_path": audio,
            "lang": "French",
            "corrections": [{"text": "un seul"}],
        },
    )
    assert r.status_code == 400


def test_translate_apply_manual_is_not_marked_edited(client, tmp_path):
    """manual_edit made an unreviewed paste outrank every later auto run; the
    correct route already keeps it False."""
    from podcodex.core.versions import is_edited, list_versions

    audio, ep_dir = audio_and_output_dir(tmp_path)
    _seed_transcript(ep_dir, n=2)

    r = client.post(
        "/api/translate/apply-manual",
        json={
            "audio_path": audio,
            "lang": "French",
            "corrections": [{"text": "un"}, {"text": "deux"}],
        },
    )
    assert r.status_code == 200, r.text

    versions = list_versions(Path(ep_dir) / "ep", "french")
    assert len(versions) == 1
    assert not is_edited(versions[0])


# ── Paths built from request fields stay inside the episode ─────────────


def _episode(tmp_path):
    """Stub show folder + episode dir, returning (audio_path, ep_dir)."""
    show = tmp_path / "show"
    show.mkdir()
    audio = show / "ep.mp3"
    audio.touch()
    (show / "ep").mkdir()
    return str(audio), str(show / "ep")


def test_translate_version_lang_traversal_rejected(client, tmp_path):
    """`lang` becomes a directory name, so traversal must 400, not read files.

    `normalize_lang` only lowercases and de-spaces, so a lang of
    "../../../../.config/podcodex" would resolve version_path onto any JSON
    file on disk, for the GET to return and the DELETE to unlink.
    """
    audio, _ = _episode(tmp_path)
    secret = tmp_path / "api_keys.json"
    secret.write_text('{"openai": "sk-secret"}', encoding="utf-8")

    params = {"audio_path": audio, "lang": "../../../.."}
    r = client.get("/api/translate/versions/api_keys", params=params)
    assert r.status_code == 400
    r = client.delete("/api/translate/versions/api_keys", params=params)
    assert r.status_code == 400
    assert secret.exists()

    r = client.get("/api/translate/versions", params=params)
    assert r.status_code == 400


def test_upload_sample_rejects_path_speaker(client, tmp_path):
    audio, _ = _episode(tmp_path)
    r = client.post(
        "/api/synthesize/upload-sample",
        data={"audio_path": audio, "speaker": "../../evil"},
        files={"file": ("a.wav", b"RIFF", "audio/wav")},
    )
    assert r.status_code == 400
