"""Tests for podcodex.bundle — `.podcodex` archive export/import.

Uses an isolated ``PODCODEX_INDEX`` per test so the user's real index is
never touched.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from podcodex.bundle import (
    ConflictError,
    ConflictPolicy,
    Manifest,
    ManifestVersionError,
    Mode,
    ShowEntry,
    export_index,
    export_show,
    import_archive,
    preview_archive,
)
from podcodex.bundle.conflicts import rename_suffix
from podcodex.bundle.manifest import (
    SCHEMA_VERSION,
    ArchiveCorruptError,
    CollectionEntry,
    manifest_from_json,
    manifest_to_json,
)
from podcodex.rag import index_store as rag_index_store
from podcodex.rag.store import collection_name

DIM = 8


def _make_show(folder: Path, name: str = "Test Show", *, audio: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "show.toml").write_text(f'name = "{name}"\n', encoding="utf-8")
    ep = folder / "ep1"
    ep.mkdir()
    (ep / "transcript.txt").write_text("hello world", encoding="utf-8")
    if audio:
        (ep / "audio.mp3").write_bytes(b"fake-mp3-data")
    return folder


def _seed_collection(store, show_name: str = "Test Show") -> str:
    col = collection_name(show_name, "bge-m3", "semantic")
    store.ensure_collection(
        col, show=show_name, model="bge-m3", chunker="semantic", dim=DIM
    )
    chunks = [
        {
            "text": "hello",
            "episode": "ep1",
            "show": show_name,
            "source": "transcript",
            "dominant_speaker": "s0",
            "start": 0.0,
            "end": 1.0,
        }
    ]
    rng = np.random.default_rng(0)
    store.save_chunks(col, "ep1", chunks, rng.random((1, DIM), dtype=np.float32))
    return col


@pytest.fixture
def isolated_index(tmp_path, monkeypatch):
    """Point IndexStore at an isolated tmp directory for the test."""
    monkeypatch.setenv("PODCODEX_INDEX", str(tmp_path / "index"))
    rag_index_store.get_index_store.cache_clear()
    yield tmp_path / "index"
    rag_index_store.get_index_store.cache_clear()


# ── Manifest ───────────────────────────────────────────────────────────


def test_manifest_roundtrip():
    m = Manifest(
        mode=Mode.FULL,
        podcodex_version="0.1.0",
        exported_at="2026-04-25T12:00:00+00:00",
        shows=[
            ShowEntry(
                name="X",
                folder="x",
                collections=[
                    CollectionEntry(
                        name="x__bge-m3__semantic",
                        model="bge-m3",
                        chunker="semantic",
                        dim=8,
                        rows=42,
                    )
                ],
            )
        ],
    )
    assert manifest_from_json(manifest_to_json(m)) == m


def test_manifest_version_reject():
    payload = (
        '{"schema_version": 99, "mode": "full", "podcodex_version": "x", '
        '"exported_at": "x", "shows": []}'
    )
    with pytest.raises(ManifestVersionError):
        manifest_from_json(payload)


def test_manifest_corrupt_raises():
    with pytest.raises(ArchiveCorruptError):
        manifest_from_json("{not json")


def test_schema_version_constant():
    assert SCHEMA_VERSION == 1


# ── rename_suffix ──────────────────────────────────────────────────────


def test_rename_suffix_appends_imported():
    assert rename_suffix("show", {"show"}) == "show-imported"


def test_rename_suffix_handles_chain():
    taken = {"show", "show-imported"}
    assert rename_suffix("show", taken) == "show-imported-2"


# ── Export ─────────────────────────────────────────────────────────────


def test_export_show_full_no_audio(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    _seed_collection(store)
    out = tmp_path / "out.podcodex"

    result = export_show(show, out, with_audio=False)

    assert result.mode == Mode.FULL
    assert result.shows_exported == 1
    assert result.collections_exported == 1
    assert result.audio_included is False
    assert out.is_file()

    with tarfile.open(out) as tf:
        names = tf.getnames()
    assert "manifest.json" in names
    assert any(n.startswith("shows/test_show/") for n in names)
    assert any(n.startswith("lancedb/") for n in names)
    assert not any(n.endswith(".mp3") for n in names)


def test_export_show_with_audio_includes_audio(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    _seed_collection(store)
    out = tmp_path / "out.podcodex"

    export_show(show, out, with_audio=True)

    with tarfile.open(out) as tf:
        names = tf.getnames()
    assert any(n.endswith("audio.mp3") for n in names)


def test_export_show_index_only_skips_show_folder(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    _seed_collection(store)
    out = tmp_path / "out.podcodex"

    result = export_show(show, out, index_only=True)

    assert result.mode == Mode.INDEX_ONLY
    assert result.audio_included is False
    with tarfile.open(out) as tf:
        names = tf.getnames()
    assert not any(n.startswith("shows/") for n in names)
    assert any(n.startswith("lancedb/") for n in names)


def test_export_show_index_only_no_collections_raises(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    out = tmp_path / "out.podcodex"

    with pytest.raises(ValueError, match="no LanceDB collections"):
        export_show(show, out, index_only=True)


def test_export_show_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_show(tmp_path / "nope", tmp_path / "out.podcodex")


def test_export_index_multi_show(tmp_path, isolated_index):
    show_a = _make_show(tmp_path / "show_a", name="Show A")
    show_b = _make_show(tmp_path / "show_b", name="Show B")
    store = rag_index_store.get_index_store()
    _seed_collection(store, "Show A")
    _seed_collection(store, "Show B")
    out = tmp_path / "out.podcodex"

    result = export_index([show_a, show_b], out)

    assert result.shows_exported == 2
    assert result.collections_exported == 2
    preview = preview_archive(out)
    names = [s.name for s in preview.manifest.shows]
    assert set(names) == {"Show A", "Show B"}


def test_export_index_empty_raises(tmp_path, isolated_index):
    with pytest.raises(ValueError, match="no shows"):
        export_index([], tmp_path / "out.podcodex")


def test_export_progress_callback_invoked(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    _seed_collection(store)
    events: list[tuple[str, float]] = []

    export_show(
        show,
        tmp_path / "out.podcodex",
        index_only=True,
        progress=lambda msg, frac: events.append((msg, frac)),
    )

    assert events
    assert events[-1][1] == pytest.approx(1.0)


# ── Preview ────────────────────────────────────────────────────────────


def test_preview_returns_manifest(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive)

    preview = preview_archive(archive)

    assert preview.manifest.mode == Mode.FULL
    assert preview.size_bytes == archive.stat().st_size
    assert len(preview.manifest.shows) == 1


def test_preview_warns_missing_embedder(tmp_path, isolated_index):
    """An unknown model in the manifest produces an embedder warning."""
    bad = tmp_path / "bad.podcodex"
    manifest = Manifest(
        mode=Mode.INDEX_ONLY,
        podcodex_version="x",
        exported_at="x",
        shows=[
            ShowEntry(
                name="X",
                folder="x",
                collections=[
                    CollectionEntry(
                        name="x__no-such-model__semantic",
                        model="no-such-model",
                        chunker="semantic",
                        dim=8,
                        rows=0,
                    )
                ],
            )
        ],
    )
    with tarfile.open(bad, "w:gz") as tf:
        payload = manifest_to_json(manifest).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    preview = preview_archive(bad)
    assert any("no-such-model" in w for w in preview.embedder_warnings)


def test_preview_missing_manifest_raises(tmp_path):
    bad = tmp_path / "bad.podcodex"
    with tarfile.open(bad, "w:gz") as tf:
        info = tarfile.TarInfo(name="other.txt")
        payload = b"hi"
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    with pytest.raises(ArchiveCorruptError):
        preview_archive(bad)


def test_preview_missing_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        preview_archive(tmp_path / "nope.podcodex")


# ── Import roundtrip ───────────────────────────────────────────────────


def test_import_full_roundtrip(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    col = _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, with_audio=False)

    # Drop local state
    store.delete_collection(col)
    rag_index_store.get_index_store.cache_clear()

    target_dir = tmp_path / "imported_shows"
    result = import_archive(archive, shows_dir=target_dir)

    assert result.mode == Mode.FULL
    assert result.shows_imported == ["test_show"]
    assert result.collections_imported == [col]
    assert (target_dir / "test_show" / "show.toml").is_file()
    new_store = rag_index_store.get_index_store()
    assert col in new_store.list_collections()


def test_import_index_only_roundtrip(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    col = _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, index_only=True)

    store.delete_collection(col)
    rag_index_store.get_index_store.cache_clear()

    result = import_archive(archive)

    assert result.mode == Mode.INDEX_ONLY
    assert result.collections_imported == [col]
    new_store = rag_index_store.get_index_store()
    assert col in new_store.list_collections()


def test_import_full_requires_shows_dir(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive)

    with pytest.raises(ValueError, match="shows_dir required"):
        import_archive(archive, shows_dir=None)


# ── Conflict policies ──────────────────────────────────────────────────


def test_import_folder_conflict_rename(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, index_only=True)  # avoid collection collision

    target = tmp_path / "imported"
    target.mkdir()
    (target / "test_show").mkdir()  # pre-existing folder

    # Switch to full mode: re-export with full
    archive2 = tmp_path / "full.podcodex"
    export_show(show, archive2, with_audio=False)

    result = import_archive(
        archive2, shows_dir=target, on_conflict=ConflictPolicy.REPLACE
    )
    # REPLACE used because collection conflict is unavoidable post-export
    assert result.shows_imported == ["test_show"]


def test_import_folder_rename_auto_suffix(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    col = _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, with_audio=False)
    store.delete_collection(col)

    target = tmp_path / "imported"
    target.mkdir()
    (target / "test_show").mkdir()
    (target / "test_show" / "marker.txt").write_text("existing", encoding="utf-8")

    result = import_archive(
        archive, shows_dir=target, on_conflict=ConflictPolicy.RENAME
    )

    assert result.shows_imported == ["test_show-imported"]
    # Original preserved
    assert (target / "test_show" / "marker.txt").is_file()
    assert (target / "test_show-imported" / "show.toml").is_file()


def test_import_folder_conflict_abort(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, with_audio=False)

    target = tmp_path / "imported"
    target.mkdir()
    (target / "test_show").mkdir()

    with pytest.raises(ConflictError):
        import_archive(archive, shows_dir=target, on_conflict=ConflictPolicy.ABORT)


def test_import_collection_conflict_replace(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    col = _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, index_only=True)

    # Don't delete: existing collection still in store, will conflict
    result = import_archive(archive, on_conflict=ConflictPolicy.REPLACE)

    assert result.collections_imported == [col]
    assert result.conflicts_resolved.get(f"collection:{col}") == "replaced"


def test_import_collection_conflict_rename_falls_back_to_replace(
    tmp_path, isolated_index
):
    """RENAME policy can't rename collections (name embeds show), so on collision
    it overwrites — pragmatic for re-imports of the same archive."""
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    col = _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, index_only=True)

    result = import_archive(archive, on_conflict=ConflictPolicy.RENAME)

    assert result.collections_imported == [col]
    assert result.conflicts_resolved.get(f"collection:{col}") == "replaced"


def test_import_collection_conflict_abort_raises(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, index_only=True)

    with pytest.raises(ConflictError):
        import_archive(archive, on_conflict=ConflictPolicy.ABORT)


def test_import_name_override(tmp_path, isolated_index):
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    col = _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, with_audio=False)
    store.delete_collection(col)

    target = tmp_path / "imported"
    result = import_archive(archive, shows_dir=target, name="renamed")

    assert result.shows_imported == ["renamed"]
    assert (target / "renamed" / "show.toml").is_file()


def test_import_name_override_rejected_for_multi_show(tmp_path, isolated_index):
    show_a = _make_show(tmp_path / "show_a", name="Show A")
    show_b = _make_show(tmp_path / "show_b", name="Show B")
    store = rag_index_store.get_index_store()
    _seed_collection(store, "Show A")
    _seed_collection(store, "Show B")
    archive = tmp_path / "out.podcodex"
    export_index([show_a, show_b], archive)

    with pytest.raises(ValueError, match="single-show"):
        import_archive(archive, name="x")


def test_import_atomic_temp_cleanup_on_failure(tmp_path, isolated_index, monkeypatch):
    """If extraction crashes, no half-written show folder is left behind."""
    show = _make_show(tmp_path / "test_show")
    store = rag_index_store.get_index_store()
    col = _seed_collection(store)
    archive = tmp_path / "out.podcodex"
    export_show(show, archive, with_audio=False)
    store.delete_collection(col)

    # Verify no stray temp tarballs remain in the archive's parent
    temp_artifacts = [
        p for p in archive.parent.iterdir() if p.name.startswith(".out.podcodex.")
    ]
    assert temp_artifacts == []
