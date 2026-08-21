"""Tests for whole-episode deletion and the audio-delete ghost cleanup.

Covers ``POST /api/shows/{folder}/episodes/delete``,
``DELETE /api/audio/file``'s orphan-row cleanup, and
``DELETE /api/shows/artwork``.
"""

from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import pytest

from tests.fixtures.api_client import make_client


class FakeIndexStore:
    """Minimal ``IndexStore`` stand-in: chunk bookkeeping only.

    Real LanceDB is not needed to pin the delete's contract: what matters is
    that every collection of the show is visited, not one named collection,
    and that a failure propagates rather than being swallowed per collection.
    """

    def __init__(self, chunks: dict[str, dict[str, int]] | None = None):
        # {collection: {episode: chunk_count}}
        self.chunks = chunks or {}
        self.raise_on_delete = False

    def get_all_collection_info(self) -> dict[str, dict]:
        return {col: {"show": "MyShow", "show_id": ""} for col in self.chunks}

    def collections_for_show(self, show_id: str, show_label: str = "") -> list[str]:
        return sorted(self.chunks)

    def list_episodes(self, collection: str) -> list[str]:
        return sorted(self.chunks.get(collection, {}))

    def delete_episode_everywhere(
        self, show: str, episode: str, show_id: str = ""
    ) -> list[str]:
        touched = []
        for col in sorted(self.chunks):
            if episode not in self.chunks[col]:
                continue
            if self.raise_on_delete:
                raise RuntimeError("lance is busy")
            del self.chunks[col][episode]
            touched.append(col)
        return touched


@pytest.fixture(autouse=True)
def store(monkeypatch):
    """Patch the index store the delete service resolves at call time.

    Autouse so no test in this module can reach real LanceDB, including the
    ones that expect to be refused long before the index is consulted.
    """
    from podcodex.rag import index_store as index_store_mod

    fake = FakeIndexStore()
    monkeypatch.setattr(index_store_mod, "get_index_store", lambda *a, **k: fake)
    return fake


@contextmanager
def active_task(key: str, task_id: str = "t1"):
    """Register a running task holding ``key``, and always release it.

    Pokes ``task_manager`` internals because there is no public way to fake a
    running task; the teardown matters, since a leaked lock would make every
    later delete in the session 409.
    """
    from podcodex.api.tasks import TaskInfo, task_manager

    info = TaskInfo(task_id=task_id, audio_path=key)
    info.status = "running"
    task_manager._tasks[task_id] = info
    task_manager.lock(key, task_id)
    try:
        yield
    finally:
        task_manager.unlock(key)
        task_manager._tasks.pop(task_id, None)


@pytest.fixture
def client(tmp_path, monkeypatch):
    from podcodex.core.app_config import AppConfig

    return make_client(
        tmp_path,
        monkeypatch,
        config=AppConfig(default_save_path=str(tmp_path / "library")),
    )


@pytest.fixture
def show(client, tmp_path):
    """A registered local show folder."""
    path = tmp_path / "MyShow"
    path.mkdir()
    r = client.post("/api/shows/register", json={"path": str(path)})
    assert r.status_code == 200, r.text
    return path


def _add_episode(show: Path, stem: str, *, audio: bool = True) -> Path:
    """Create an episode on disk with one transcript version and a DB row."""
    from podcodex.core.versions import save_version

    if audio:
        (show / f"{stem}.mp3").write_bytes(b"fake audio")
    base = show / stem / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    save_version(
        base,
        "transcript",
        [{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "Narrator"}],
        {"step": "transcript", "type": "asr", "model": "tiny", "params": {}},
    )
    return show / stem


def _delete(client, show: Path, stem: str):
    return client.post(
        f"/api/shows/{quote(str(show), safe='')}/episodes/delete", json={"stem": stem}
    )


def _stems(client, show: Path) -> set[str]:
    r = client.get(f"/api/shows/{quote(str(show), safe='')}/unified")
    assert r.status_code == 200, r.text
    return {e["stem"] for e in r.json()}


# ── Happy path ────────────────────────────────────────────


def test_deletes_every_store(client, show, store):
    _add_episode(show, "ep1")
    store.chunks = {"col-a": {"ep1": 3}, "col-b": {"ep1": 2, "other": 9}}
    assert "ep1" in _stems(client, show)

    r = _delete(client, show, "ep1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "deleted"
    assert body["warnings"] == []
    assert body["collections"] == 2
    assert body["db_row_removed"] is True

    assert not (show / "ep1.mp3").exists()
    assert not (show / "ep1").exists()
    assert store.chunks == {"col-a": {}, "col-b": {"other": 9}}
    assert "ep1" not in _stems(client, show)


def test_leaves_other_episodes_alone(client, show):
    _add_episode(show, "ep1")
    _add_episode(show, "ep2")

    assert _delete(client, show, "ep1").status_code == 200

    assert (show / "ep2.mp3").exists()
    assert (show / "ep2" / "transcript").is_dir()
    assert _stems(client, show) == {"ep2"}


def test_subtitle_only_episode_deletes(client, show, store):
    """The case with no delete action at all today: no audio, output dir only."""
    _add_episode(show, "ep1", audio=False)
    store.chunks = {"col-a": {"ep1": 4}}
    assert "ep1" in _stems(client, show)

    r = _delete(client, show, "ep1")
    assert r.status_code == 200, r.text
    assert r.json()["audio_removed"] is True
    assert not (show / "ep1").exists()
    assert "ep1" not in _stems(client, show)


def test_second_delete_is_a_no_op(client, show):
    _add_episode(show, "ep1")
    assert _delete(client, show, "ep1").status_code == 200

    r = _delete(client, show, "ep1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "deleted"
    assert body["collections"] == 0
    assert body["db_row_removed"] is False


def test_output_dir_is_not_recreated(client, show):
    """Regression pin for ``AudioPaths.from_audio``'s mkdir side effect.

    Resolving episode paths through it (``core/_utils.py``) would recreate
    ``{show}/{stem}/`` while deleting it, leaving an empty directory that the
    scanner then keeps healing back into a row.
    """
    _add_episode(show, "ep1")
    assert _delete(client, show, "ep1").status_code == 200

    assert not (show / "ep1").exists()
    # And a status read afterwards must not resurrect it either.
    _stems(client, show)
    assert not (show / "ep1").exists()


# ── Partial failure ───────────────────────────────────────


def test_chunk_failure_deletes_nothing(client, show, store):
    """Chunks that survive must never outlive the episode's visibility.

    Listing is derived from disk, so removing the files after a failed chunk
    delete would leave chunks answering searches for an episode that can no
    longer be seen or retried. The delete must abort with everything intact.
    """
    _add_episode(show, "ep1")
    store.chunks = {"col-a": {"ep1": 3}}
    store.raise_on_delete = True

    r = _delete(client, show, "ep1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "partial"
    assert body["warnings"]
    assert body["db_row_removed"] is False
    assert body["output_dir_removed"] is False
    assert body["audio_removed"] is False

    assert store.chunks == {"col-a": {"ep1": 3}}
    assert (show / "ep1.mp3").exists()
    assert (show / "ep1" / "transcript").is_dir()
    assert "ep1" in _stems(client, show)


def test_undeletable_audio_keeps_the_row(client, show, monkeypatch):
    """Files left behind must keep the episode listed so it can be retried."""
    _add_episode(show, "ep1")

    real_unlink = Path.unlink

    def refuse(self, *a, **k):
        if self.suffix == ".mp3":
            raise PermissionError("file is open")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", refuse)
    body = _delete(client, show, "ep1").json()
    monkeypatch.undo()

    assert body["status"] == "partial"
    assert body["audio_removed"] is False
    assert body["db_row_removed"] is False
    assert (show / "ep1.mp3").exists()
    # The heal pass rebuilds a row from the surviving audio, so the episode
    # stays on screen even though its output dir is gone.
    assert "ep1" in _stems(client, show)


def test_retry_after_a_chunk_failure_finishes(client, show, store):
    _add_episode(show, "ep1")
    store.chunks = {"col-a": {"ep1": 3}}
    store.raise_on_delete = True
    assert _delete(client, show, "ep1").json()["status"] == "partial"

    store.raise_on_delete = False
    r = _delete(client, show, "ep1")
    assert r.json()["status"] == "deleted"
    assert "ep1" not in _stems(client, show)


def test_every_audio_file_for_the_stem_goes(client, show):
    """Nothing enforces one audio file per stem, and one left behind resurrects."""
    _add_episode(show, "ep1")
    (show / "ep1.wav").write_bytes(b"second encoding")

    body = _delete(client, show, "ep1").json()

    assert body["status"] == "deleted"
    assert not (show / "ep1.mp3").exists()
    assert not (show / "ep1.wav").exists()
    assert "ep1" not in _stems(client, show)


def test_uppercase_extension_is_found(client, show):
    """The scanner matches on a lowercased suffix, so the delete must too.

    Probing ``{stem}.mp3`` would miss ``ep1.MP3`` on a case-sensitive
    filesystem: the episode would list as downloaded and the delete would
    report the audio gone while the file stayed.
    """
    (show / "ep1.MP3").write_bytes(b"fake audio")
    assert "ep1" in _stems(client, show)

    body = _delete(client, show, "ep1").json()

    assert body["audio_removed"] is True
    assert not (show / "ep1.MP3").exists()
    assert "ep1" not in _stems(client, show)


def test_db_failure_is_reported_not_swallowed(client, show, monkeypatch):
    """A failed row delete must not read as a clean delete."""
    _add_episode(show, "ep1")

    from podcodex.core import delete_episode as svc

    def boom(*_a, **_k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(svc, "drop_db_row", boom)
    body = _delete(client, show, "ep1").json()

    assert body["status"] == "partial"
    assert body["db_row_removed"] is False
    assert any("database" in w for w in body["warnings"])


# ── Feed cache ────────────────────────────────────────────


def _feed_show(client, tmp_path, *, removed: bool):
    """A feed-backed show whose cache holds one entry for episode 'ep1'."""
    from podcodex.ingest.rss import RSSEpisode, episode_stem, save_feed_cache
    from podcodex.ingest.show import ShowMeta, save_show_meta

    show = tmp_path / "FeedShow"
    show.mkdir()
    assert (
        client.post("/api/shows/register", json={"path": str(show)}).status_code == 200
    )
    save_show_meta(show, ShowMeta(name="FeedShow", rss_url="https://example.com/f.xml"))

    ep = RSSEpisode(
        guid="guid-1",
        title="Episode One",
        pub_date="2026-01-01",
        audio_url="https://example.com/1.mp3",
        removed=removed,
    )
    stem = episode_stem(ep, show)
    save_feed_cache(show, [ep])
    (show / f"{stem}.mp3").write_bytes(b"fake audio")
    return show, stem


def test_delete_prunes_an_episode_the_feed_dropped(client, tmp_path):
    """A removed=True entry must not outlive its own deletion.

    merge_with_cache re-emits every cached guid forever, so without the prune
    the episode stays listed with all four stores empty and no way to clear it.
    """
    show, stem = _feed_show(client, tmp_path, removed=True)
    assert stem in _stems(client, show)

    assert _delete(client, show, stem).json()["status"] == "deleted"

    from podcodex.ingest.rss import load_feed_cache

    assert load_feed_cache(show) == []
    assert stem not in _stems(client, show)


def test_delete_keeps_an_episode_the_feed_still_lists(client, tmp_path):
    """The documented behavior, and what the confirm dialog promises.

    Pruning a live entry would only make it blink out until the next refresh
    put it straight back.
    """
    show, stem = _feed_show(client, tmp_path, removed=False)

    assert _delete(client, show, stem).json()["status"] == "deleted"

    from podcodex.ingest.rss import load_feed_cache

    assert [e.guid for e in load_feed_cache(show)] == ["guid-1"]
    assert stem in _stems(client, show)


def test_failed_delete_leaves_the_feed_cache_alone(client, tmp_path, store):
    """The prune is a reward for a clean delete, not a side effect of trying."""
    show, stem = _feed_show(client, tmp_path, removed=True)
    store.chunks = {"col-a": {stem: 2}}
    store.raise_on_delete = True

    assert _delete(client, show, stem).json()["status"] == "partial"

    from podcodex.ingest.rss import load_feed_cache

    assert [e.guid for e in load_feed_cache(show)] == ["guid-1"]


# ── Guards ────────────────────────────────────────────────


def test_unregistered_show_is_refused(client, tmp_path):
    stray = tmp_path / "not-a-show"
    stray.mkdir()
    (stray / "ep1.mp3").write_bytes(b"x")

    r = _delete(client, stray, "ep1")
    assert r.status_code == 403
    assert (stray / "ep1.mp3").exists()


@pytest.mark.parametrize("stem", ["", ".", "..", "../evil", "sub/ep"])
def test_bad_stems_are_refused(client, show, stem):
    r = _delete(client, show, stem)
    assert r.status_code == 400
    assert show.is_dir()


def test_active_task_blocks_the_delete(client, show):
    """Both the show-level keys and the episode's own ref must be checked."""
    _add_episode(show, "ep1")
    with active_task(str(show / "ep1.mp3")):
        assert _delete(client, show, "ep1").status_code == 409
        assert (show / "ep1.mp3").exists()


def test_virtual_lock_blocks_a_subtitle_only_delete(client, show):
    """The lock key for an audio-less episode is ``{output_dir}.virtual``.

    That string is minted by ``lib/episodeRef.ts:getEpisodeBatchPath`` and
    locked verbatim by the batch runner, so a delete that only looks at
    ``audio_path`` would run straight through an active batch.
    """
    from podcodex.core._utils import virtual_audio_path

    _add_episode(show, "ep1", audio=False)
    with active_task(virtual_audio_path(show / "ep1")):
        assert _delete(client, show, "ep1").status_code == 409
        assert (show / "ep1").is_dir()


def test_service_refuses_a_stem_that_escapes_the_show(tmp_path):
    """Direct-caller guard: the route is not the only entry point.

    ``bad_path_component`` misses Windows drive-relative names, which pathlib
    joins by replacing the base. Unreachable through the route (which also
    runs ``resolve_inside_show_root``) and inert on POSIX, so this asserts the
    service's own check rather than a platform-specific string.
    """
    from podcodex.core.delete_episode import delete_episode

    show = tmp_path / "MyShow"
    (show / "sub").mkdir(parents=True)
    with pytest.raises(ValueError):
        delete_episode(show, "..")
    assert (show / "sub").is_dir()


def test_single_episode_lock_blocks_a_subtitle_only_delete(client, show):
    """Single-episode runs use a different key than the batch runner.

    ``stores/episodeStore.ts:useAudioPath`` falls back to ``{folder}/{stem}.mp3``
    for an episode with no audio, so a Correct or Translate run on a
    subtitle-only episode holds that key, not the ``.virtual`` one. Checking
    only ``.virtual`` would rmtree the output dir under a live job.
    """
    _add_episode(show, "ep1", audio=False)
    with active_task(f"{show / 'ep1'}.mp3"):
        assert _delete(client, show, "ep1").status_code == 409
        assert (show / "ep1").is_dir()


def test_download_lock_blocks_the_delete(client, show):
    """A bulk RSS download writes the audio back after it finishes."""
    _add_episode(show, "ep1")
    with active_task(f"download:{show}"):
        assert _delete(client, show, "ep1").status_code == 409


def test_batch_lock_blocks_the_delete(client, show):
    _add_episode(show, "ep1")
    with active_task(f"batch:{show}"):
        assert _delete(client, show, "ep1").status_code == 409


# ── Audio-delete ghost cleanup ────────────────────────────


def test_delete_audio_drops_an_orphan_row(client, show):
    """An imported file with no transcripts leaves no ghost behind."""
    (show / "solo.mp3").write_bytes(b"fake audio")
    assert "solo" in _stems(client, show)

    r = client.delete(f"/api/audio/file?path={quote(str(show / 'solo.mp3'), safe='')}")
    assert r.status_code == 200, r.text
    assert r.json()["episode_removed"] is True
    assert "solo" not in _stems(client, show)


def test_delete_audio_keeps_a_row_with_transcripts(client, show):
    _add_episode(show, "ep1")

    r = client.delete(f"/api/audio/file?path={quote(str(show / 'ep1.mp3'), safe='')}")
    assert r.status_code == 200, r.text
    assert r.json()["episode_removed"] is False
    assert "ep1" in _stems(client, show)


def test_deleting_a_nested_audio_file_makes_no_stray_db(client, show):
    """`.wav` is an audio extension, so synthesized output reaches this route.

    Its parent is the synthesize dir, not the show root, and opening a pipeline
    DB there would create a stray ``pipeline.db`` inside the episode folder.
    """
    nested = show / "ep1" / "synthesize"
    nested.mkdir(parents=True)
    wav = nested / "v1.wav"
    wav.write_bytes(b"fake synth")

    r = client.delete(f"/api/audio/file?path={quote(str(wav), safe='')}")

    assert r.status_code == 200, r.text
    assert r.json()["episode_removed"] is False
    assert not wav.exists()
    assert not (nested / "pipeline.db").exists()


# ── Cover removal ─────────────────────────────────────────


def test_remove_cover_clears_file_and_marker(client, show):
    from podcodex.ingest.show import load_show_meta

    q = f"show_folder={quote(str(show), safe='')}"
    r = client.post(
        f"/api/shows/artwork?{q}",
        files={"file": ("cover.png", b"\x89PNG fake", "image/png")},
    )
    assert r.status_code == 200, r.text
    assert (show / "artwork.png").exists()
    assert load_show_meta(show).artwork_url == "local"

    r = client.delete(f"/api/shows/artwork?{q}")
    assert r.status_code == 200, r.text

    assert not (show / "artwork.png").exists()
    assert not (show / ".artwork_url_hash").exists()
    # Empty, not a sentinel: this is what lets a feed refresh restore the
    # feed's own artwork on an RSS / YouTube show.
    assert load_show_meta(show).artwork_url == ""
    assert client.get(f"/api/shows/artwork?{q}").status_code == 404


def test_remove_cover_is_idempotent(client, show):
    q = f"show_folder={quote(str(show), safe='')}"
    assert client.delete(f"/api/shows/artwork?{q}").status_code == 200
    assert client.delete(f"/api/shows/artwork?{q}").status_code == 200


def test_remove_cover_needs_a_registered_show(client, tmp_path):
    stray = tmp_path / "stray"
    stray.mkdir()
    r = client.delete(f"/api/shows/artwork?show_folder={quote(str(stray), safe='')}")
    assert r.status_code == 403
