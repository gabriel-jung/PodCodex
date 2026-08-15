"""Tests for podcodex.core.versions — generation versioning."""

import json
import pytest

from podcodex.core.versions import (
    compute_hash,
    delete_version,
    has_matching_version,
    has_version,
    list_versions,
    load_latest,
    load_version,
    resolve_verified_source,
    save_version,
    version_count,
    versions_dir,
)
from podcodex.core.pipeline_db import get_pipeline_db


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
    "step": "corrected",
    "type": "raw",
    "model": "gpt-4o",
    "params": {"llm_mode": "api"},
    "manual_edit": False,
}


def _prov(step="corrected", type_="raw", model=None, params=None, manual_edit=False):
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
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o"),
        )

        assert vid.endswith("_raw")

        # Segment file exists
        seg_path = versions_dir(episode_dir) / "corrected" / f"{vid}.json"
        assert seg_path.exists()
        segments = json.loads(seg_path.read_text())
        assert len(segments) == 2

    def test_multiple_versions(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(model="v1"))
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(type_="validated", manual_edit=True),
        )

        entries = list_versions(episode_dir, "corrected")
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
        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, None)
        assert vid == ""
        assert version_count(episode_dir, "corrected") == 0

    def test_input_hash_stored(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            {**_prov(), "input_hash": "sha256:abcdef1234567890"},
        )
        entries = list_versions(episode_dir, "corrected")
        assert entries[0]["input_hash"] == "sha256:abcdef1234567890"


class TestListVersions:
    def test_empty_when_no_versions(self, episode_dir):
        assert list_versions(episode_dir, "corrected") == []

    def test_returns_entries(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        entries = list_versions(episode_dir, "corrected")
        assert len(entries) == 1


class TestLoadVersion:
    def test_load_existing(self, episode_dir):
        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        segments = load_version(episode_dir, "corrected", vid)
        assert len(segments) == 2
        assert segments[0]["text"] == "Hello"

    def test_load_missing_raises(self, episode_dir):
        with pytest.raises(FileNotFoundError):
            load_version(episode_dir, "corrected", "nonexistent")


class TestLoadLatest:
    def test_returns_none_when_empty(self, episode_dir):
        assert load_latest(episode_dir, "corrected") is None

    def test_returns_latest(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            [{"text": "old"}],
            _prov(model="v1"),
        )
        save_version(
            episode_dir,
            "corrected",
            [{"text": "new"}],
            _prov(model="v2"),
        )
        segments = load_latest(episode_dir, "corrected")
        assert segments == [{"text": "new"}]


class TestVersionCount:
    def test_zero_when_empty(self, episode_dir):
        assert version_count(episode_dir, "corrected") == 0

    def test_counts_correctly(self, episode_dir):
        for i in range(3):
            save_version(
                episode_dir,
                "corrected",
                SAMPLE_SEGMENTS,
                _prov(model=f"m{i}"),
            )
        assert version_count(episode_dir, "corrected") == 3


class TestDifferentSteps:
    def test_steps_isolated(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        save_version(
            episode_dir,
            "english",
            SAMPLE_SEGMENTS,
            _prov(step="english"),
        )

        assert version_count(episode_dir, "corrected") == 1
        assert version_count(episode_dir, "english") == 1


class TestHasVersion:
    def test_false_when_empty(self, episode_dir):
        assert has_version(episode_dir, "corrected") is False

    def test_true_when_exists(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        assert has_version(episode_dir, "corrected") is True


class TestHasMatchingVersion:
    def test_no_versions(self, episode_dir):
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "gpt-4o"}) is False
        )

    def test_matching_model(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"llm_mode": "api"}),
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "gpt-4o"}) is True
        )

    def test_different_model(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"llm_mode": "api"}),
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "claude"}) is False
        )

    def test_matching_params(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"llm_mode": "api", "llm_provider": "openai"}),
        )
        assert (
            has_matching_version(
                episode_dir,
                "corrected",
                {"model": "gpt-4o", "llm_mode": "api", "llm_provider": "openai"},
            )
            is True
        )

    def test_partial_param_mismatch(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="gpt-4o", params={"llm_mode": "api", "llm_provider": "openai"}),
        )
        # Different provider
        assert (
            has_matching_version(
                episode_dir,
                "corrected",
                {"model": "gpt-4o", "llm_mode": "api", "llm_provider": "anthropic"},
            )
            is False
        )

    def test_empty_params_matches_any(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov())
        assert has_matching_version(episode_dir, "corrected", {}) is True

    def test_multiple_versions_one_matches(self, episode_dir):
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="old-model"),
        )
        save_version(
            episode_dir,
            "corrected",
            SAMPLE_SEGMENTS,
            _prov(model="new-model"),
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "old-model"})
            is True
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "new-model"})
            is True
        )
        assert (
            has_matching_version(episode_dir, "corrected", {"model": "other"}) is False
        )


# ──────────────────────────────────────────────
# Dual-write & DB→file fallback
# ──────────────────────────────────────────────


class TestSaveAndLoad:
    """Verify save_version writes both file and DB, and load_latest uses the DB."""

    def test_save_writes_both_file_and_db(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

        version_id = save_version(
            episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE
        )

        # File half
        version_file = versions_dir(episode_dir) / "corrected" / f"{version_id}.json"
        assert version_file.exists()
        assert json.loads(version_file.read_text()) == SAMPLE_SEGMENTS

        # DB half — pipeline.db lives at show level (base.parent.parent)
        show_dir = episode_dir.parent.parent
        assert (show_dir / "pipeline.db").exists()
        db = get_pipeline_db(show_dir)
        meta = db.get_latest_version(episode_dir.name, "corrected")
        assert meta is not None
        assert meta["id"] == version_id
        assert meta["content_hash"] == compute_hash(SAMPLE_SEGMENTS)
        close_pipeline_db(show_dir)

    def test_load_latest_returns_none_when_db_empty(self, episode_dir):
        """No versions saved → load_latest returns None (no filesystem fallback)."""
        assert load_latest(episode_dir, "corrected") is None

    def test_load_latest_round_trip(self, episode_dir):
        """Save then load returns the same segments."""
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        loaded = load_latest(episode_dir, "corrected")
        assert loaded == SAMPLE_SEGMENTS


class TestResolveVerifiedSource:
    """Verify resolve_verified_source helper + delete cleanup."""

    def test_returns_none_when_no_pointer(self, episode_dir):
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        assert resolve_verified_source(episode_dir) is None

    def test_returns_pointer_when_set(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "corrected", vid)
        resolved = resolve_verified_source(episode_dir)
        assert resolved is not None
        step, version_id, path = resolved
        assert step == "corrected"
        assert version_id == vid
        assert path.exists()
        close_pipeline_db(show_dir)

    def test_returns_none_when_file_missing(self, episode_dir):
        """Stale pointer (file deleted out-of-band) resolves to None."""
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "corrected", vid)
        # Wipe file behind the pointer; DB still holds the reference.
        (versions_dir(episode_dir) / "corrected" / f"{vid}.json").unlink()
        assert resolve_verified_source(episode_dir) is None
        close_pipeline_db(show_dir)

    def test_returns_none_for_non_verifiable_step(self, episode_dir):
        """A pointer at e.g. 'synthesize' is not a valid verified source."""
        from podcodex.core.pipeline_db import close_pipeline_db

        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "english", "v-1")
        assert resolve_verified_source(episode_dir) is None
        close_pipeline_db(show_dir)

    def test_delete_clears_pointer_when_target_removed(self, episode_dir):
        """Deleting the verified version clears the pointer via refresh hook."""
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "corrected", vid)
        delete_version(episode_dir, "corrected", vid)
        assert db.get_verified(episode_dir.name) is None
        close_pipeline_db(show_dir)

    def test_delete_keeps_pointer_when_other_version_removed(self, episode_dir):
        """Deleting a non-verified version leaves the pointer intact."""
        from podcodex.core.pipeline_db import close_pipeline_db

        v1 = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        v2 = save_version(
            episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(model="other")
        )
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.set_verified(episode_dir.name, "corrected", v1)
        delete_version(episode_dir, "corrected", v2)
        ptr = db.get_verified(episode_dir.name)
        assert ptr == {"step": "corrected", "version_id": v1}
        close_pipeline_db(show_dir)


class TestBackfillFromDisk:
    """Rebuilding pipeline.db must make on-disk versions reachable again."""

    def test_restores_rows_for_orphaned_files(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db
        from podcodex.core.versions import backfill_versions_from_disk

        vid = save_version(
            episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(type_="validated")
        )
        show_dir = episode_dir.parent.parent
        close_pipeline_db(show_dir)
        (show_dir / "pipeline.db").unlink()

        assert backfill_versions_from_disk(show_dir) == 1
        rows = get_pipeline_db(show_dir).list_versions(episode_dir.name, "corrected")
        assert [r["id"] for r in rows] == [vid]
        # The type suffix survives, so an edited version still reads as edited
        # rather than silently demoting to a raw model output.
        assert rows[0]["type"] == "validated"
        assert rows[0]["segment_count"] == len(SAMPLE_SEGMENTS)
        assert load_version(episode_dir, "corrected", vid) == SAMPLE_SEGMENTS
        close_pipeline_db(show_dir)

    def test_is_idempotent(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db
        from podcodex.core.versions import backfill_versions_from_disk

        save_version(
            episode_dir, "transcript", SAMPLE_SEGMENTS, _prov(step="transcript")
        )
        show_dir = episode_dir.parent.parent
        # Rows already exist, so a second pass must not duplicate them.
        assert backfill_versions_from_disk(show_dir) == 0
        close_pipeline_db(show_dir)


class TestStatusDemotionOnDelete:
    """Deleting the last version of a step demotes its pipeline_db flag."""

    def test_last_delete_demotes_flag(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, corrected=True)
        delete_version(episode_dir, "corrected", vid)
        assert db.get_episode(episode_dir.name)["corrected"] is False
        close_pipeline_db(show_dir)

    def test_demote_skipped_when_a_version_survives(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        v1 = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(model="other"))
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, corrected=True)
        delete_version(episode_dir, "corrected", v1)
        assert db.get_episode(episode_dir.name)["corrected"] is True
        close_pipeline_db(show_dir)

    def test_demote_is_atomic_against_a_racing_save(self, episode_dir):
        """A version registered mid-delete must win over the demotion.

        Pipeline steps run in spawned subprocesses writing to this same DB,
        so the emptiness check and the flag write have to be one transaction.
        Simulated here by registering a new version from inside the check.
        """
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, SAMPLE_PROVENANCE)
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, corrected=True)

        # The racing save lands after the delete removes the only row, but
        # before the demotion commits.
        original = db.demote_step_if_no_versions

        def racing_demote(stem, step, flag):
            save_version(episode_dir, "corrected", SAMPLE_SEGMENTS, _prov(model="race"))
            return original(stem, step, flag)

        db.demote_step_if_no_versions = racing_demote
        try:
            delete_version(episode_dir, "corrected", vid)
        finally:
            db.demote_step_if_no_versions = original

        assert list_versions(episode_dir, "corrected"), "racing save should survive"
        assert db.get_episode(episode_dir.name)["corrected"] is True
        close_pipeline_db(show_dir)

    def test_last_translation_delete_drops_the_language(self, episode_dir):
        from podcodex.core.pipeline_db import close_pipeline_db

        vid = save_version(
            episode_dir, "english", SAMPLE_SEGMENTS, _prov(step="english")
        )
        show_dir = episode_dir.parent.parent
        db = get_pipeline_db(show_dir)
        db.mark(episode_dir.name, translations=["english", "french"])
        delete_version(episode_dir, "english", vid)
        assert db.get_episode(episode_dir.name)["translations"] == ["french"]
        close_pipeline_db(show_dir)
