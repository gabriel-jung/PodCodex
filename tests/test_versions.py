"""Tests for podcodex.core.versions — generation versioning."""

import json

import pytest

from podcodex.core.versions import (
    VersionMeta,
    archive_version,
    compute_hash,
    list_versions,
    load_version,
    prune_versions,
    version_count,
)


@pytest.fixture
def episode_dir(tmp_path):
    """Create a minimal episode directory and return the 'base' path."""
    ep = tmp_path / "my_episode"
    ep.mkdir()
    return ep / "my_episode"  # base = dir / stem


SAMPLE_SEGMENTS = [
    {"speaker": "Alice", "text": "Hello", "start": 0.0, "end": 1.0},
    {"speaker": "Bob", "text": "Hi there", "start": 1.0, "end": 2.5},
]


class TestComputeHash:
    def test_deterministic(self):
        h1 = compute_hash(SAMPLE_SEGMENTS)
        h2 = compute_hash(SAMPLE_SEGMENTS)
        assert h1 == h2

    def test_starts_with_sha256(self):
        h = compute_hash(SAMPLE_SEGMENTS)
        assert h.startswith("sha256:")

    def test_different_content_different_hash(self):
        other = [{"speaker": "Alice", "text": "Goodbye", "start": 0.0, "end": 1.0}]
        assert compute_hash(SAMPLE_SEGMENTS) != compute_hash(other)

    def test_key_order_irrelevant(self):
        seg1 = [{"a": 1, "b": 2}]
        seg2 = [{"b": 2, "a": 1}]
        assert compute_hash(seg1) == compute_hash(seg2)


class TestArchiveVersion:
    def test_creates_archive(self, episode_dir):
        meta = VersionMeta(step="polished", type="raw", model="gpt-4o")
        vid = archive_version(episode_dir, "polished", SAMPLE_SEGMENTS, meta)

        assert vid.endswith("_raw")
        assert meta.id == vid
        assert meta.segment_count == 2
        assert meta.content_hash.startswith("sha256:")

        # Manifest exists
        manifest_path = episode_dir.parent / ".versions" / "polished.json"
        assert manifest_path.exists()
        entries = json.loads(manifest_path.read_text())
        assert len(entries) == 1
        assert entries[0]["id"] == vid
        assert entries[0]["model"] == "gpt-4o"

        # Archived segments exist
        seg_path = episode_dir.parent / ".versions" / "polished" / f"{vid}.json"
        assert seg_path.exists()
        segments = json.loads(seg_path.read_text())
        assert len(segments) == 2

    def test_multiple_archives(self, episode_dir):
        m1 = VersionMeta(step="polished", type="raw", model="v1")
        archive_version(episode_dir, "polished", SAMPLE_SEGMENTS, m1)

        m2 = VersionMeta(
            step="polished", type="validated", model=None, manual_edit=True
        )
        archive_version(episode_dir, "polished", SAMPLE_SEGMENTS, m2)

        entries = list_versions(episode_dir, "polished")
        assert len(entries) == 2
        # Newest first
        assert entries[0]["type"] == "validated"
        assert entries[1]["type"] == "raw"

    def test_params_stored(self, episode_dir):
        meta = VersionMeta(
            step="transcript",
            type="raw",
            model="large-v3-turbo",
            params={"language": "fr", "batch_size": 16},
        )
        archive_version(episode_dir, "transcript", SAMPLE_SEGMENTS, meta)
        entries = list_versions(episode_dir, "transcript")
        assert entries[0]["params"]["language"] == "fr"


class TestListVersions:
    def test_empty_when_no_versions(self, episode_dir):
        assert list_versions(episode_dir, "polished") == []

    def test_returns_entries(self, episode_dir):
        meta = VersionMeta(step="polished", type="raw")
        archive_version(episode_dir, "polished", SAMPLE_SEGMENTS, meta)
        entries = list_versions(episode_dir, "polished")
        assert len(entries) == 1


class TestLoadVersion:
    def test_load_existing(self, episode_dir):
        meta = VersionMeta(step="polished", type="raw")
        vid = archive_version(episode_dir, "polished", SAMPLE_SEGMENTS, meta)
        segments = load_version(episode_dir, "polished", vid)
        assert len(segments) == 2
        assert segments[0]["text"] == "Hello"

    def test_load_missing_raises(self, episode_dir):
        with pytest.raises(FileNotFoundError):
            load_version(episode_dir, "polished", "nonexistent")


class TestVersionCount:
    def test_zero_when_empty(self, episode_dir):
        assert version_count(episode_dir, "polished") == 0

    def test_counts_correctly(self, episode_dir):
        for i in range(3):
            meta = VersionMeta(step="polished", type="raw", model=f"m{i}")
            archive_version(episode_dir, "polished", SAMPLE_SEGMENTS, meta)
        assert version_count(episode_dir, "polished") == 3


class TestPruneVersions:
    def test_prune_keeps_newest(self, episode_dir):
        ids = []
        for i in range(5):
            meta = VersionMeta(step="polished", type="raw", model=f"m{i}")
            ids.append(archive_version(episode_dir, "polished", SAMPLE_SEGMENTS, meta))

        removed = prune_versions(episode_dir, "polished", keep=2)
        assert removed == 3

        remaining = list_versions(episode_dir, "polished")
        assert len(remaining) == 2
        # Newest are kept (first two in manifest since newest-first)
        assert remaining[0]["id"] == ids[-1]
        assert remaining[1]["id"] == ids[-2]

        # Old files deleted
        sdir = episode_dir.parent / ".versions" / "polished"
        for old_id in ids[:3]:
            assert not (sdir / f"{old_id}.json").exists()

    def test_prune_noop_when_under_limit(self, episode_dir):
        meta = VersionMeta(step="polished", type="raw")
        archive_version(episode_dir, "polished", SAMPLE_SEGMENTS, meta)
        removed = prune_versions(episode_dir, "polished", keep=5)
        assert removed == 0
        assert version_count(episode_dir, "polished") == 1


class TestDifferentSteps:
    def test_steps_isolated(self, episode_dir):
        m1 = VersionMeta(step="polished", type="raw")
        archive_version(episode_dir, "polished", SAMPLE_SEGMENTS, m1)

        m2 = VersionMeta(step="english", type="raw")
        archive_version(episode_dir, "english", SAMPLE_SEGMENTS, m2)

        assert version_count(episode_dir, "polished") == 1
        assert version_count(episode_dir, "english") == 1


class TestSaveSegmentsJsonIntegration:
    """Test that save_segments_json archives when provenance is provided."""

    def test_with_provenance(self, episode_dir):
        from podcodex.core._utils import save_segments_json

        path = episode_dir.with_suffix(".polished.raw.json")
        provenance = {
            "step": "polished",
            "type": "raw",
            "model": "test-model",
            "params": {"mode": "ollama"},
            "manual_edit": False,
            "base": str(episode_dir),
        }
        save_segments_json(path, SAMPLE_SEGMENTS, "Test", provenance=provenance)

        # File was written
        assert path.exists()
        # Version was archived
        assert version_count(episode_dir, "polished") == 1
        entries = list_versions(episode_dir, "polished")
        assert entries[0]["model"] == "test-model"

    def test_without_provenance(self, episode_dir):
        from podcodex.core._utils import save_segments_json

        path = episode_dir.with_suffix(".polished.raw.json")
        save_segments_json(path, SAMPLE_SEGMENTS, "Test")

        # File was written
        assert path.exists()
        # No version archived
        assert version_count(episode_dir, "polished") == 0
