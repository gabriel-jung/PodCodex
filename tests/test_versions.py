"""Tests for podcodex.core.versions — generation versioning."""

import json
import pytest

from podcodex.core.versions import (
    compute_hash,
    has_matching_version,
    has_version,
    list_versions,
    load_latest,
    load_version,
    prune_versions,
    save_version,
    version_count,
)


@pytest.fixture
def episode_dir(tmp_path):
    """Create a show/episode structure and return the 'base' path.

    Layout: tmp_path/show/episode/episode  (base = episode dir / stem)
    The show dir is base.parent.parent, which is where pipeline.db lives.
    """
    show = tmp_path / "show"
    show.mkdir()
    ep = show / "my_episode"
    ep.mkdir()
    return ep / "my_episode"  # base = dir / stem


SAMPLE_SEGMENTS = [
    {"speaker": "Alice", "text": "Hello", "start": 0.0, "end": 1.0},
    {"speaker": "Bob", "text": "Hi there", "start": 1.0, "end": 2.5},
]

SAMPLE_PROVENANCE = {
    "step": "polished",
    "type": "raw",
    "model": "gpt-4o",
    "params": {"mode": "api"},
    "manual_edit": False,
}


def _prov(step="polished", type_="raw", model=None, params=None, manual_edit=False):
    """Build a provenance dict for tests."""
    return {
        "step": step,
        "type": type_,
        "model": model,
        "params": params or {},
        "manual_edit": manual_edit,
    }


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


class TestSaveVersion:
    def test_creates_version(self, episode_dir):
        vid = save_version(
            episode_dir,
            "polished",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o"),
        )

        assert vid.endswith("_raw")

        # Segment file exists
        seg_path = episode_dir.parent / ".versions" / "polished" / f"{vid}.json"
        assert seg_path.exists()
        segments = json.loads(seg_path.read_text())
        assert len(segments) == 2

    def test_multiple_versions(self, episode_dir):
        save_version(episode_dir, "polished", SAMPLE_SEGMENTS, _prov(model="v1"))
        save_version(
            episode_dir,
            "polished",
            SAMPLE_SEGMENTS,
            _prov(type_="validated", manual_edit=True),
        )

        entries = list_versions(episode_dir, "polished")
        assert len(entries) == 2
        # Newest first
        assert entries[0]["type"] == "validated"
        assert entries[1]["type"] == "raw"

    def test_params_stored(self, episode_dir):
        save_version(
            episode_dir,
            "transcript",
            SAMPLE_SEGMENTS,
            _prov(step="transcript", model="large-v3", params={"language": "fr"}),
        )
        entries = list_versions(episode_dir, "transcript")
        assert entries[0]["params"]["language"] == "fr"

    def test_none_provenance_is_noop(self, episode_dir):
        vid = save_version(episode_dir, "polished", SAMPLE_SEGMENTS, None)
        assert vid == ""
        assert version_count(episode_dir, "polished") == 0

    def test_input_hash_stored(self, episode_dir):
        save_version(
            episode_dir,
            "polished",
            SAMPLE_SEGMENTS,
            {**_prov(), "input_hash": "sha256:abcdef1234567890"},
        )
        entries = list_versions(episode_dir, "polished")
        assert entries[0]["input_hash"] == "sha256:abcdef1234567890"


class TestListVersions:
    def test_empty_when_no_versions(self, episode_dir):
        assert list_versions(episode_dir, "polished") == []

    def test_returns_entries(self, episode_dir):
        save_version(episode_dir, "polished", SAMPLE_SEGMENTS, _prov())
        entries = list_versions(episode_dir, "polished")
        assert len(entries) == 1


class TestLoadVersion:
    def test_load_existing(self, episode_dir):
        vid = save_version(episode_dir, "polished", SAMPLE_SEGMENTS, _prov())
        segments = load_version(episode_dir, "polished", vid)
        assert len(segments) == 2
        assert segments[0]["text"] == "Hello"

    def test_load_missing_raises(self, episode_dir):
        with pytest.raises(FileNotFoundError):
            load_version(episode_dir, "polished", "nonexistent")


class TestLoadLatest:
    def test_returns_none_when_empty(self, episode_dir):
        assert load_latest(episode_dir, "polished") is None

    def test_returns_latest(self, episode_dir):
        save_version(
            episode_dir,
            "polished",
            [{"text": "old"}],
            _prov(model="v1"),
        )
        save_version(
            episode_dir,
            "polished",
            [{"text": "new"}],
            _prov(model="v2"),
        )
        segments = load_latest(episode_dir, "polished")
        assert segments == [{"text": "new"}]


class TestVersionCount:
    def test_zero_when_empty(self, episode_dir):
        assert version_count(episode_dir, "polished") == 0

    def test_counts_correctly(self, episode_dir):
        for i in range(3):
            save_version(
                episode_dir,
                "polished",
                SAMPLE_SEGMENTS,
                _prov(model=f"m{i}"),
            )
        assert version_count(episode_dir, "polished") == 3


class TestPruneVersions:
    def test_prune_keeps_newest(self, episode_dir):
        ids = []
        for i in range(5):
            vid = save_version(
                episode_dir,
                "polished",
                SAMPLE_SEGMENTS,
                _prov(model=f"m{i}"),
            )
            ids.append(vid)

        removed = prune_versions(episode_dir, "polished", keep=2)
        assert removed == 3

        remaining = list_versions(episode_dir, "polished")
        assert len(remaining) == 2
        # Newest are kept
        assert remaining[0]["id"] == ids[-1]
        assert remaining[1]["id"] == ids[-2]

        # Old files deleted
        sdir = episode_dir.parent / ".versions" / "polished"
        for old_id in ids[:3]:
            assert not (sdir / f"{old_id}.json").exists()

    def test_prune_noop_when_under_limit(self, episode_dir):
        save_version(episode_dir, "polished", SAMPLE_SEGMENTS, _prov())
        removed = prune_versions(episode_dir, "polished", keep=5)
        assert removed == 0
        assert version_count(episode_dir, "polished") == 1


class TestDifferentSteps:
    def test_steps_isolated(self, episode_dir):
        save_version(episode_dir, "polished", SAMPLE_SEGMENTS, _prov())
        save_version(
            episode_dir,
            "english",
            SAMPLE_SEGMENTS,
            _prov(step="english"),
        )

        assert version_count(episode_dir, "polished") == 1
        assert version_count(episode_dir, "english") == 1


class TestHasVersion:
    def test_false_when_empty(self, episode_dir):
        assert has_version(episode_dir, "polished") is False

    def test_true_when_exists(self, episode_dir):
        save_version(episode_dir, "polished", SAMPLE_SEGMENTS, _prov())
        assert has_version(episode_dir, "polished") is True


class TestHasMatchingVersion:
    def test_no_versions(self, episode_dir):
        assert (
            has_matching_version(episode_dir, "polished", {"model": "gpt-4o"}) is False
        )

    def test_matching_model(self, episode_dir):
        save_version(
            episode_dir,
            "polished",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"mode": "api"}),
        )
        assert (
            has_matching_version(episode_dir, "polished", {"model": "gpt-4o"}) is True
        )

    def test_different_model(self, episode_dir):
        save_version(
            episode_dir,
            "polished",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"mode": "api"}),
        )
        assert (
            has_matching_version(episode_dir, "polished", {"model": "claude"}) is False
        )

    def test_matching_params(self, episode_dir):
        save_version(
            episode_dir,
            "polished",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"mode": "api", "provider": "openai"}),
        )
        assert (
            has_matching_version(
                episode_dir,
                "polished",
                {"model": "gpt-4o", "mode": "api", "provider": "openai"},
            )
            is True
        )

    def test_partial_param_mismatch(self, episode_dir):
        save_version(
            episode_dir,
            "polished",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"mode": "api", "provider": "openai"}),
        )
        # Different provider
        assert (
            has_matching_version(
                episode_dir,
                "polished",
                {"model": "gpt-4o", "mode": "api", "provider": "anthropic"},
            )
            is False
        )

    def test_empty_params_matches_any(self, episode_dir):
        save_version(episode_dir, "polished", SAMPLE_SEGMENTS, _prov())
        assert has_matching_version(episode_dir, "polished", {}) is True

    def test_multiple_versions_one_matches(self, episode_dir):
        save_version(
            episode_dir,
            "polished",
            SAMPLE_SEGMENTS,
            _prov(model="old-model"),
        )
        save_version(
            episode_dir,
            "polished",
            SAMPLE_SEGMENTS,
            _prov(model="new-model"),
        )
        assert (
            has_matching_version(episode_dir, "polished", {"model": "old-model"})
            is True
        )
        assert (
            has_matching_version(episode_dir, "polished", {"model": "new-model"})
            is True
        )
        assert (
            has_matching_version(episode_dir, "polished", {"model": "other"}) is False
        )
