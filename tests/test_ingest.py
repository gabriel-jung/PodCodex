"""Tests for podcodex.ingest.folder — filesystem mocked with tmp_path."""


# ──────────────────────────────────────────────
# Audio file detection
# ──────────────────────────────────────────────


def test_scan_folder_ignores_non_audio(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    (tmp_path / "ep01.transcript.json").touch()
    (tmp_path / "ep01.segments.parquet").touch()
    (tmp_path / "image.png").touch()
    (tmp_path / "README.md").touch()

    result = scan_folder(tmp_path)

    assert len(result) == 1
    assert result[0].stem == "ep01"


# ──────────────────────────────────────────────
# output_dir derivation
# ──────────────────────────────────────────────


def test_scan_folder_output_dir_derivation(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    result = scan_folder(tmp_path)

    assert result[0].output_dir == tmp_path / "ep01"


# ──────────────────────────────────────────────
# transcribed flag
# ──────────────────────────────────────────────


def test_scan_folder_transcribed_true(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "transcript").mkdir()
    (ep_dir / "transcript" / "v1.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].transcribed is True


# ──────────────────────────────────────────────
# indexed flag
# ──────────────────────────────────────────────


def test_scan_folder_indexed_true(tmp_path, monkeypatch):
    from podcodex.ingest import folder as folder_mod

    (tmp_path / "ep01.mp3").touch()
    monkeypatch.setattr(folder_mod, "lance_indexed_stems", lambda _: {"ep01"})
    folder_mod.invalidate_scan_cache()

    result = folder_mod.scan_folder(tmp_path)

    assert result[0].indexed is True


def test_note_episode_indexed_updates_cache(tmp_path, monkeypatch):
    from podcodex.ingest import folder as folder_mod

    version = {"show_bge": 7}

    class FakeStore:
        def collection_version(self, col):
            return version[col]

    monkeypatch.setattr("podcodex.rag.index_store.get_index_store", lambda: FakeStore())
    monkeypatch.setattr(folder_mod, "_INDEXED_STEMS_CACHE", {})

    # Cold cache: nothing to keep warm, next request scans anyway.
    folder_mod.note_episode_indexed(tmp_path, "ep01")
    assert folder_mod._INDEXED_STEMS_CACHE == {}

    key = folder_mod._stems_cache_key(tmp_path)
    folder_mod._INDEXED_STEMS_CACHE[key] = ((("show_bge", "t1", 3),), {"ep00"}, None)
    folder_mod.note_episode_indexed(tmp_path, "ep01")
    versions, stems, records = folder_mod._INDEXED_STEMS_CACHE[key]
    assert stems == {"ep00", "ep01"}
    assert records is None
    assert versions == (("show_bge", "t1", 7),)

    # The collection was dropped and recreated (its version starts over): the
    # cached set holds the old table's episodes, so it is dropped rather than
    # added to.
    version["show_bge"] = 2
    folder_mod.note_episode_indexed(tmp_path, "ep02")
    assert key not in folder_mod._INDEXED_STEMS_CACHE


def test_note_episode_indexed_refreshes_that_episodes_source_record(
    tmp_path, monkeypatch
):
    """An index write bumps every version: without the refresh each status
    poll of an index batch rescanned every collection's first chunks."""
    from podcodex.ingest import folder as folder_mod

    fresh = {"hash": "sha256:new", "ref": "auto", "step": "corrected"}

    class FakeStore:
        def collection_version(self, col):
            return 8

        def episode_source_record(self, col, stem):
            return fresh if stem == "ep01" else None

    monkeypatch.setattr("podcodex.rag.index_store.get_index_store", lambda: FakeStore())
    key = folder_mod._stems_cache_key(tmp_path)
    before = (("show_bge", "t1", 7),)
    old = {"hash": "sha256:old", "ref": "auto", "step": "corrected"}
    monkeypatch.setattr(
        folder_mod,
        "_INDEXED_STEMS_CACHE",
        {key: (before, {"ep01", "ep02"}, {"ep01": [old], "ep02": [old]})},
    )

    folder_mod.note_episode_indexed(tmp_path, "ep01")
    versions, _stems, records = folder_mod._INDEXED_STEMS_CACHE[key]
    assert versions == (("show_bge", "t1", 8),)
    assert records == {"ep01": [fresh], "ep02": [old]}


def test_stems_and_source_records_share_one_scan(tmp_path, monkeypatch):
    """The records ride on the stems' cache entry: asking for both reads the
    collection list once and the records only when first asked for."""
    from podcodex.ingest import folder as folder_mod
    from podcodex.ingest.show import ShowMeta, save_show_meta

    save_show_meta(tmp_path, ShowMeta(name="MyShow"))
    record = {"hash": "sha256:a", "ref": "auto", "step": "corrected"}
    calls = {"episodes": 0, "records": 0}

    class FakeStore:
        def collections_for_show(self, *_a, **_kw):
            return ["show_bge", "show_old"]

        def get_all_collection_info(self):
            return {
                "show_bge": {
                    "created_at": "t1",
                    "model": "bge-m3",
                    "chunker": "semantic",
                },
                "show_old": {
                    "created_at": "t1",
                    "model": "retired",
                    "chunker": "semantic",
                },
            }

        def collection_version(self, col):
            return 3

        def list_episodes(self, col):
            calls["episodes"] += 1
            return ["ep01"]

        def episode_source_records(self, col):
            # A model this install does not know: its stale chunks could
            # never be refreshed, so they must not flag the episode.
            assert col == "show_bge"
            calls["records"] += 1
            return {"ep01": record}

    monkeypatch.setattr("podcodex.rag.index_store.get_index_store", lambda: FakeStore())
    monkeypatch.setattr(folder_mod, "_INDEXED_STEMS_CACHE", {})
    assert folder_mod.lance_indexed_stems(tmp_path) == {"ep01"}
    assert calls == {"episodes": 2, "records": 0}
    assert folder_mod.lance_source_records(tmp_path) == {"ep01": [record]}
    assert folder_mod.lance_source_records(tmp_path) == {"ep01": [record]}
    assert folder_mod.lance_indexed_stems(tmp_path) == {"ep01"}
    assert calls == {"episodes": 2, "records": 1}


def test_a_recreated_collection_rescans_at_the_same_version(tmp_path, monkeypatch):
    """A recreated table counts up from 1 again: at the old version number
    the key must still differ, or the dropped table's episodes are served."""
    from podcodex.ingest import folder as folder_mod
    from podcodex.ingest.show import ShowMeta, save_show_meta

    save_show_meta(tmp_path, ShowMeta(name="MyShow"))
    state = {"created": "t1", "episodes": ["old"]}

    class FakeStore:
        def collections_for_show(self, *_a, **_kw):
            return ["show_bge"]

        def get_all_collection_info(self):
            return {"show_bge": {"created_at": state["created"]}}

        def collection_version(self, col):
            return 3

        def list_episodes(self, col):
            return list(state["episodes"])

    monkeypatch.setattr("podcodex.rag.index_store.get_index_store", lambda: FakeStore())
    monkeypatch.setattr(folder_mod, "_INDEXED_STEMS_CACHE", {})
    assert folder_mod.lance_indexed_stems(tmp_path) == {"old"}
    state.update(created="t2", episodes=["new"])
    assert folder_mod.lance_indexed_stems(tmp_path) == {"new"}


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


def test_scan_folder_empty(tmp_path):
    from podcodex.ingest.folder import scan_folder

    result = scan_folder(tmp_path)

    assert result == []


def test_scan_folder_sorted(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "zzz.mp3").touch()
    (tmp_path / "aaa.wav").touch()
    (tmp_path / "mmm.flac").touch()

    result = scan_folder(tmp_path)

    assert [ep.stem for ep in result] == ["aaa", "mmm", "zzz"]


def test_scan_folder_no_output_dir_yet(tmp_path):
    """Episodes with no output_dir yet should have all flags False."""
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    result = scan_folder(tmp_path)
    ep = result[0]

    assert ep.transcribed is False
    assert ep.corrected is False
    assert ep.indexed is False
    assert ep.translations == []


# ──────────────────────────────────────────────
# Transcript-only episodes (no audio)
# ──────────────────────────────────────────────


def test_scan_folder_transcript_only_episode(tmp_path):
    """A subdir with a transcript version but no audio file is discovered."""
    from podcodex.ingest.folder import scan_folder

    ep_dir = tmp_path / "ep_rss"
    ep_dir.mkdir()
    (ep_dir / "transcript").mkdir()
    (ep_dir / "transcript" / "v1.json").touch()

    result = scan_folder(tmp_path)

    assert len(result) == 1
    ep = result[0]
    assert ep.stem == "ep_rss"
    assert ep.audio_path is None
    assert ep.transcribed is True


def test_scan_folder_mixed_audio_and_transcript_only(tmp_path):
    """Both audio-based and transcript-only episodes in one folder."""
    from podcodex.ingest.folder import scan_folder

    # Audio episode
    (tmp_path / "ep01.mp3").touch()
    ep1 = tmp_path / "ep01"
    ep1.mkdir()
    (ep1 / "transcript").mkdir()
    (ep1 / "transcript" / "v1.json").touch()

    # Transcript-only episode
    ep2 = tmp_path / "ep02"
    ep2.mkdir()
    (ep2 / "transcript").mkdir()
    (ep2 / "transcript" / "v1.json").touch()

    result = scan_folder(tmp_path)

    assert len(result) == 2
    assert result[0].stem == "ep01"
    assert result[0].audio_path is not None
    assert result[1].stem == "ep02"
    assert result[1].audio_path is None


def test_scan_folder_subdir_without_transcript_ignored(tmp_path):
    """A subdir with no transcript files is not treated as an episode."""
    from podcodex.ingest.folder import scan_folder

    ep_dir = tmp_path / "random_dir"
    ep_dir.mkdir()
    (ep_dir / "some_file.txt").touch()

    result = scan_folder(tmp_path)

    assert result == []


def test_scan_folder_loads_episode_title(tmp_path):
    """scan_folder populates title from .episode_meta.json when present."""
    import json
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    meta = {"guid": "g1", "title": "My Great Episode", "pub_date": "2026-01-01"}
    (ep_dir / ".episode_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    result = scan_folder(tmp_path)
    assert result[0].title == "My Great Episode"


def test_scan_folder_metadata_only_episode(tmp_path):
    """Subdirs with only .episode_meta.json are discovered as episodes."""
    import json
    from podcodex.ingest.folder import scan_folder

    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    meta = {"guid": "g1", "title": "My Episode", "pub_date": "2026-01-01"}
    (ep_dir / ".episode_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    result = scan_folder(tmp_path)
    assert len(result) == 1
    assert result[0].stem == "ep01"
    assert result[0].title == "My Episode"
    assert result[0].audio_path is None
    assert not result[0].transcribed


# ──────────────────────────────────────────────
# download_youtube_audio: force
# ──────────────────────────────────────────────


def test_download_youtube_audio_force_replaces_an_existing_file(tmp_path, monkeypatch):
    """force re-downloads even when output_path already exists, as the
    route documents; an exists() short-circuit must not swallow the flag."""
    from podcodex.ingest import youtube as yt

    calls: list[str] = []

    class _FakeYDL:
        def __init__(self, opts):
            self._opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            calls.append(urls[0])
            (tmp_path / "ep.mp3").write_bytes(b"fresh")
            return 0

    monkeypatch.setattr(
        yt, "_require_yt_dlp", lambda: type("M", (), {"YoutubeDL": _FakeYDL})
    )
    (tmp_path / "ep.mp3").write_bytes(b"stale")

    out = yt.download_youtube_audio("vid", tmp_path, "ep")
    assert calls == []
    assert out.read_bytes() == b"stale"

    out = yt.download_youtube_audio("vid", tmp_path, "ep", force=True)
    assert len(calls) == 1
    assert out.read_bytes() == b"fresh"


def test_a_records_read_does_not_overwrite_a_newer_entry(tmp_path, monkeypatch):
    """An index write may store a newer entry while a poll scans records:
    putting the older one back made the next poll rescan every chunk."""
    from podcodex.ingest import folder as folder_mod
    from podcodex.ingest.show import ShowMeta, save_show_meta

    save_show_meta(tmp_path, ShowMeta(name="MyShow"))
    key = folder_mod._stems_cache_key(tmp_path)
    newer = ((("show_bge", "t1", 4),), {"ep01", "ep02"}, None)

    class FakeStore:
        def collections_for_show(self, *_a, **_kw):
            return ["show_bge"]

        def get_all_collection_info(self):
            return {
                "show_bge": {
                    "created_at": "t1",
                    "model": "bge-m3",
                    "chunker": "semantic",
                }
            }

        def collection_version(self, col):
            return 3

        def list_episodes(self, col):
            return ["ep01"]

        def episode_source_records(self, col):
            folder_mod._INDEXED_STEMS_CACHE[key] = newer  # the write lands
            return {}

    monkeypatch.setattr("podcodex.rag.index_store.get_index_store", lambda: FakeStore())
    monkeypatch.setattr(folder_mod, "_INDEXED_STEMS_CACHE", {})
    assert folder_mod.lance_indexed_stems(tmp_path) == {"ep01"}
    assert folder_mod.lance_source_records(tmp_path) == {}
    assert folder_mod._INDEXED_STEMS_CACHE[key] is newer
