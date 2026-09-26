"""YouTube download route: pacing only where a request is made."""

from __future__ import annotations

import time

import pytest

from podcodex.ingest.rss import RSSEpisode, save_feed_cache
from tests.fixtures.api_client import make_client


@pytest.fixture
def client(tmp_path, monkeypatch):
    return make_client(tmp_path, monkeypatch)


@pytest.fixture
def show(client, tmp_path):
    path = tmp_path / "Chan"
    path.mkdir()
    r = client.post("/api/shows/register", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return path


def _wait(task_id, timeout=10.0):
    from podcodex.api.tasks import task_manager

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = task_manager.get(task_id)
        if info is not None and info.finished_at is not None:
            return info
        time.sleep(0.02)
    raise AssertionError("task never finished")


def test_episodes_already_on_disk_are_not_paced(client, show, monkeypatch):
    """A mostly-downloaded show used to sleep the pacer delay per skipped video."""
    import podcodex.api.routes.youtube as yt_route
    import podcodex.ingest.youtube as yt

    eps = [
        RSSEpisode(guid=f"v{i}", title=f"Video {i}", pub_date="2024-01-01")
        for i in range(5)
    ]
    save_feed_cache(show, eps)
    monkeypatch.setattr(yt_route, "is_downloaded", lambda *_a: True)
    waits: list[int] = []
    monkeypatch.setattr(yt.Pacer, "wait", lambda self: waits.append(1))
    monkeypatch.setattr(
        yt, "download_youtube_audio", lambda *_a, **_k: pytest.fail("no download")
    )

    r = client.post(f"/api/shows/{show}/youtube/download", json={})
    assert r.status_code == 200, r.text
    info = _wait(r.json()["task_id"])
    assert info.status == "completed", info.error
    assert waits == []
    assert {e["status"] for e in info.result} == {"exists"}


def _subs_run(client, show, monkeypatch, outcomes):
    import podcodex.ingest.youtube as yt

    eps = [
        RSSEpisode(guid=f"v{i}", title=f"Video {i}", pub_date="2024-01-01")
        for i in range(len(outcomes))
    ]
    save_feed_cache(show, eps)
    monkeypatch.setattr(yt.Pacer, "wait", lambda self: None)
    by_guid = {e.guid: o for e, o in zip(eps, outcomes)}

    def fake_cache(guid, *_a, **_k):
        outcome = by_guid[guid]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(yt, "cache_youtube_subtitles", fake_cache)
    r = client.post(
        f"/api/shows/{show}/youtube/import-subs",
        json={"video_ids": [e.guid for e in eps]},
    )
    assert r.status_code == 200, r.text
    info = _wait(r.json()["task_id"])
    assert info.status == "completed", info.error
    return info


def test_subtitle_import_reports_each_outcome(client, show, monkeypatch):
    info = _subs_run(
        client, show, monkeypatch, [True, False, RuntimeError("HTTP Error 403")]
    )
    statuses = [r["status"] for r in info.result["results"]]
    assert statuses == ["cached", "no_subtitles", "failed"]
    assert info.result["imported"] == 1
    assert info.result["failed"] == 1
    assert info.result["throttled"] is False


def test_subtitle_import_stops_after_consecutive_failures(client, show, monkeypatch):
    from podcodex.ingest.youtube import _CONSECUTIVE_FAIL_LIMIT

    n = _CONSECUTIVE_FAIL_LIMIT + 2
    info = _subs_run(client, show, monkeypatch, [RuntimeError("HTTP Error 429")] * n)
    assert len(info.result["results"]) == _CONSECUTIVE_FAIL_LIMIT
    assert info.result["throttled"] is True
    assert any("Stopped" in step for step in info.steps)


def test_keyed_lock_is_shared_across_spellings_of_one_folder(tmp_path):
    from podcodex.api.routes._helpers import keyed_lock

    (tmp_path / "Show").mkdir()
    a = keyed_lock("artwork", tmp_path / "Show")
    b = keyed_lock("artwork", f"{tmp_path}/Show/")
    c = keyed_lock("feed-cache", tmp_path / "Show")
    assert a is b and a is not c


def test_one_pacer_tick_per_episode_with_subtitles(client, show, monkeypatch):
    import podcodex.api.routes.youtube as yt_route
    import podcodex.ingest.youtube as yt

    eps = [
        RSSEpisode(guid=f"v{i}", title=f"V {i}", pub_date="2024-01-01")
        for i in range(3)
    ]
    save_feed_cache(show, eps)
    waits: list[int] = []
    monkeypatch.setattr(yt.Pacer, "wait", lambda self: waits.append(1))
    monkeypatch.setattr(yt_route, "is_downloaded", lambda *_a: False)
    monkeypatch.setattr(yt, "download_youtube_audio", lambda *_a, **_k: show / "x.mp3")
    monkeypatch.setattr(yt, "cache_youtube_subtitles", lambda *_a, **_k: True)
    r = client.post(f"/api/shows/{show}/youtube/download", json={"import_subs": True})
    assert _wait(r.json()["task_id"]).status == "completed"
    assert len(waits) == 3
