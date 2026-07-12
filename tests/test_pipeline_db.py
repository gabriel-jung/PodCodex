"""Tests for podcodex.core.pipeline_db — per-show SQLite pipeline status."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from podcodex.core.pipeline_db import PipelineDB, get_pipeline_db, close_pipeline_db


@pytest.fixture
def db():
    """In-memory PipelineDB for fast tests."""
    d = PipelineDB(":memory:")
    yield d
    d.close()


# ── Basic CRUD ────────────────────────────────────────────


def test_empty_db(db):
    assert db.all_episodes() == []
    assert db.get_episode("nonexistent") is None
    assert db.episode_count() == 0


def test_mark_creates_row(db):
    """mark() on a non-existent stem creates the row."""
    db.mark("ep1", transcribed=True)
    row = db.get_episode("ep1")
    assert row["transcribed"] is True
    assert row["corrected"] is False


def test_mark_updates_row(db):
    db.mark("ep1", transcribed=True)
    db.mark("ep1", corrected=True)
    row = db.get_episode("ep1")
    assert row["transcribed"] is True
    assert row["corrected"] is True


def test_mark_invalid_column(db):
    with pytest.raises(ValueError, match="Unknown columns"):
        db.mark("ep1", bogus=True)


def test_mark_empty_is_noop(db):
    db.mark("ep1")
    assert db.get_episode("ep1") is None


# ── Translations ──────────────────────────────────────────


def test_mark_translations(db):
    db.mark("ep1", translations=["english", "french"])
    row = db.get_episode("ep1")
    assert row["translations"] == ["english", "french"]


def test_mark_translations_overwrite(db):
    db.mark("ep1", translations=["english"])
    db.mark("ep1", translations=["english", "french", "german"])
    row = db.get_episode("ep1")
    assert row["translations"] == ["english", "french", "german"]


# ── Bulk populate ─────────────────────────────────────────


@dataclass
class FakeEpisode:
    stem: str
    audio_path: Path | None = None
    transcribed: bool = False
    corrected: bool = False
    indexed: bool = False
    synthesized: bool = False
    translations: list[str] = field(default_factory=list)


def test_populate_from_scan(db):
    episodes = [
        FakeEpisode(stem="ep1", transcribed=True, corrected=True, translations=["en"]),
        FakeEpisode(stem="ep2", audio_path=Path("/a/ep2.mp3"), indexed=True),
        FakeEpisode(stem="ep3"),
    ]
    db.populate_from_scan(episodes)
    assert db.episode_count() == 3

    ep1 = db.get_episode("ep1")
    assert ep1["transcribed"] is True
    assert ep1["corrected"] is True
    assert ep1["translations"] == ["en"]

    ep2 = db.get_episode("ep2")
    assert ep2["audio_path"] == "/a/ep2.mp3"
    assert ep2["indexed"] is True

    ep3 = db.get_episode("ep3")
    assert ep3["transcribed"] is False


def test_populate_upserts(db):
    """populate_from_scan updates existing rows."""
    db.mark("ep1", transcribed=True)
    episodes = [FakeEpisode(stem="ep1", transcribed=True, corrected=True)]
    db.populate_from_scan(episodes)
    row = db.get_episode("ep1")
    assert row["corrected"] is True


# ── all_episodes ordering ────────────────────────────────


def test_all_episodes_sorted(db):
    db.mark("c", transcribed=True)
    db.mark("a", corrected=True)
    db.mark("b", indexed=True)
    stems = [row["stem"] for row in db.all_episodes()]
    assert stems == ["a", "b", "c"]


# ── Module-level cache ────────────────────────────────────


def test_get_pipeline_db_caches(tmp_path):
    db1 = get_pipeline_db(tmp_path)
    db2 = get_pipeline_db(tmp_path)
    assert db1 is db2
    close_pipeline_db(tmp_path)


def test_close_pipeline_db(tmp_path):
    db = get_pipeline_db(tmp_path)
    db.mark("ep", transcribed=True)
    close_pipeline_db(tmp_path)
    # Re-open — data persists.
    db2 = get_pipeline_db(tmp_path)
    assert db2.get_episode("ep")["transcribed"] is True
    close_pipeline_db(tmp_path)


# ── Provenance ───────────────────────────────────────────


def test_provenance_stored_and_read(db):
    prov = {
        "transcript": {
            "step": "transcript",
            "model": "large-v3",
            "params": {"diarize": True},
        }
    }
    db.mark("ep1", transcribed=True, provenance=prov)
    row = db.get_episode("ep1")
    assert row["provenance"]["transcript"]["model"] == "large-v3"
    assert row["provenance"]["transcript"]["params"]["diarize"] is True


def test_provenance_merge_across_steps(db):
    """Each step key merges into the existing provenance dict."""
    db.mark("ep1", transcribed=True, provenance={"transcript": {"model": "large-v3"}})
    db.mark("ep1", corrected=True, provenance={"corrected": {"model": "qwen3:4b"}})
    db.mark("ep1", provenance={"english": {"model": "gpt-4o"}})
    row = db.get_episode("ep1")
    assert row["provenance"]["transcript"]["model"] == "large-v3"
    assert row["provenance"]["corrected"]["model"] == "qwen3:4b"
    assert row["provenance"]["english"]["model"] == "gpt-4o"


def test_provenance_overwrite_same_step(db):
    """Writing the same step key overwrites it."""
    db.mark("ep1", provenance={"transcript": {"model": "small"}})
    db.mark("ep1", provenance={"transcript": {"model": "large-v3"}})
    row = db.get_episode("ep1")
    assert row["provenance"]["transcript"]["model"] == "large-v3"


def test_provenance_empty_by_default(db):
    db.mark("ep1", transcribed=True)
    row = db.get_episode("ep1")
    assert row["provenance"] == {}


def test_provenance_in_populate(db):
    """populate_from_scan creates rows with empty provenance."""
    episodes = [FakeEpisode(stem="ep1", transcribed=True)]
    db.populate_from_scan(episodes)
    row = db.get_episode("ep1")
    assert row["provenance"] == {}


# ── Step statuses ────────────────────────────────────────


def _make_status_row(
    transcribed=False,
    corrected=False,
    translations=None,
    provenance=None,
    verified=None,
):
    """Build a minimal status dict like PipelineDB.all_episodes() returns."""
    return {
        "transcribed": transcribed,
        "corrected": corrected,
        "indexed": False,
        "synthesized": False,
        "translations": translations or [],
        "provenance": provenance or {},
        "verified": verified,
    }


class TestStepStatuses:
    """Test the _step_statuses() comparison logic from shows.py."""

    @staticmethod
    def _step_statuses(st, provenance, effective):
        from podcodex.api.routes.shows import _step_statuses
        from podcodex.core.translate import clean_translations

        return _step_statuses(
            st, provenance, effective, clean_translations(st.get("translations", []))
        )

    def test_none_when_not_done(self):
        st = _make_status_row()
        result = self._step_statuses(st, {}, {"model_size": "large-v3"})
        assert result["transcribe_status"] == "none"
        assert result["correct_status"] == "none"
        assert result["translate_status"] == "none"

    def test_done_when_matching(self):
        prov = {
            "transcript": {
                "model": "large-v3",
                "type": "validated",
                "params": {"diarize": True},
            },
            "corrected": {
                "model": "qwen3:4b",
                "type": "validated",
                "params": {"llm_mode": "ollama", "llm_provider": ""},
            },
        }
        st = _make_status_row(transcribed=True, corrected=True, provenance=prov)
        effective = {"model_size": "large-v3", "diarize": True, "llm_mode": "ollama"}
        result = self._step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "done"
        assert result["correct_status"] == "done"

    def test_outdated_model_mismatch(self):
        prov = {"transcript": {"model": "small", "params": {"diarize": True}}}
        st = _make_status_row(transcribed=True, provenance=prov)
        effective = {"model_size": "large-v3", "diarize": True}
        result = self._step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "outdated"

    def test_outdated_diarize_mismatch(self):
        prov = {"transcript": {"model": "large-v3", "params": {"diarize": False}}}
        st = _make_status_row(transcribed=True, provenance=prov)
        effective = {"model_size": "large-v3", "diarize": True}
        result = self._step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "outdated"

    def test_verified_transcript_overrides_outdated(self):
        """Verified pointer on transcript forces 'done' regardless of model drift."""
        prov = {"transcript": {"model": "small", "params": {"diarize": False}}}
        st = _make_status_row(
            transcribed=True,
            provenance=prov,
            verified={"step": "transcript", "version_id": "v-1"},
        )
        effective = {"model_size": "large-v3", "diarize": True}
        result = self._step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "done"

    def test_verified_corrected_overrides_outdated(self):
        prov = {
            "corrected": {
                "model": "qwen3:4b",
                "params": {"llm_mode": "ollama"},
            }
        }
        st = _make_status_row(
            corrected=True,
            provenance=prov,
            verified={"step": "corrected", "version_id": "v-2"},
        )
        effective = {"llm_mode": "api", "llm_model": "gpt-4o"}
        result = self._step_statuses(st, prov, effective)
        assert result["correct_status"] == "done"

    def test_verified_on_one_step_does_not_affect_other(self):
        """Verified on transcript leaves the correct step's normal status alone."""
        prov = {
            "transcript": {"model": "small", "params": {}},
            "corrected": {
                "model": "qwen3:4b",
                "params": {"llm_mode": "ollama"},
            },
        }
        st = _make_status_row(
            transcribed=True,
            corrected=True,
            provenance=prov,
            verified={"step": "transcript", "version_id": "v-1"},
        )
        effective = {"model_size": "large-v3", "llm_mode": "api", "llm_model": "gpt-4o"}
        result = self._step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "done"
        assert result["correct_status"] == "outdated"

    def test_outdated_correct_provider_mismatch(self):
        prov = {
            "corrected": {
                "model": "qwen3:4b",
                "params": {"llm_mode": "ollama", "llm_provider": ""},
            }
        }
        st = _make_status_row(corrected=True, provenance=prov)
        effective = {"llm_mode": "api", "llm_provider": "openai"}
        result = self._step_statuses(st, prov, effective)
        assert result["correct_status"] == "outdated"

    def test_done_no_provenance(self):
        """Episodes without provenance default to 'done' (pre-existing episodes)."""
        st = _make_status_row(transcribed=True, corrected=True)
        result = self._step_statuses(st, {}, {"model_size": "large-v3"})
        assert result["transcribe_status"] == "done"
        assert result["correct_status"] == "done"

    def test_done_no_defaults(self):
        """No defaults configured → everything is 'done'."""
        prov = {"transcript": {"model": "small", "type": "validated", "params": {}}}
        st = _make_status_row(transcribed=True, provenance=prov)
        result = self._step_statuses(st, prov, {})
        assert result["transcribe_status"] == "done"

    def test_translate_target_lang(self):
        prov = {
            "english": {
                "model": "gpt-4o",
                "params": {"llm_mode": "api", "llm_provider": "openai"},
            }
        }
        st = _make_status_row(translations=["english"], provenance=prov)
        effective = {
            "target_lang": "english",
            "llm_mode": "api",
            "llm_provider": "openai",
        }
        result = self._step_statuses(st, prov, effective)
        assert result["translate_status"] == "done"

    def test_translate_missing_target_lang(self):
        """Target lang configured but not translated → 'none'."""
        st = _make_status_row(translations=["french"])
        effective = {"target_lang": "english"}
        result = self._step_statuses(st, {}, effective)
        assert result["translate_status"] == "none"

    def test_translate_outdated_model(self):
        prov = {
            "english": {
                "model": "old-model",
                "params": {"llm_mode": "api", "llm_provider": "openai"},
            }
        }
        st = _make_status_row(translations=["english"], provenance=prov)
        effective = {
            "target_lang": "english",
            "llm_mode": "api",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
        }
        result = self._step_statuses(st, prov, effective)
        assert result["translate_status"] == "outdated"

    def test_edited_beats_outdated_transcript(self):
        """User-validated transcript stays 'done' even if model defaults changed."""
        prov = {
            "transcript": {
                "model": "small",
                "type": "validated",
                "manual_edit": True,
                "params": {"diarize": False},
            }
        }
        st = _make_status_row(transcribed=True, provenance=prov)
        effective = {"model_size": "large-v3", "diarize": True}
        result = self._step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "done"

    def test_edited_beats_outdated_corrected(self):
        prov = {
            "corrected": {
                "model": "qwen3:4b",
                "manual_edit": True,
                "params": {"llm_mode": "ollama", "llm_provider": ""},
            }
        }
        st = _make_status_row(corrected=True, provenance=prov)
        effective = {"llm_mode": "api", "llm_provider": "openai"}
        result = self._step_statuses(st, prov, effective)
        assert result["correct_status"] == "done"

    def test_edited_beats_outdated_translate(self):
        prov = {
            "english": {
                "model": "old-model",
                "type": "validated",
                "params": {"llm_mode": "api", "llm_provider": "openai"},
            }
        }
        st = _make_status_row(translations=["english"], provenance=prov)
        effective = {
            "target_lang": "english",
            "llm_mode": "api",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
        }
        result = self._step_statuses(st, prov, effective)
        assert result["translate_status"] == "done"


# ── Resolve defaults ─────────────────────────────────────


class TestResolveDefaults:
    """Test the _resolve_defaults() merging logic from shows.py."""

    @staticmethod
    def _resolve_defaults(app_defaults, show_meta):
        from podcodex.api.routes.shows import _resolve_defaults

        return _resolve_defaults(app_defaults, show_meta)

    def test_app_defaults_only(self):
        result = self._resolve_defaults(
            {"model_size": "large-v3", "diarize": True}, None
        )
        assert result["model_size"] == "large-v3"
        assert result["diarize"] is True

    def test_show_overrides_app(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults(model_size="small"))
        result = self._resolve_defaults({"model_size": "large-v3"}, show)
        assert result["model_size"] == "small"

    def test_show_empty_falls_back(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults())  # all defaults
        result = self._resolve_defaults(
            {"model_size": "large-v3", "llm_mode": "ollama"}, show
        )
        assert result["model_size"] == "large-v3"
        assert result["llm_mode"] == "ollama"

    def test_show_diarize_false_overrides(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults(diarize=False))
        result = self._resolve_defaults({"diarize": True}, show)
        assert result["diarize"] is False

    def test_llm_model_resolved_per_mode(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(
            name="test",
            pipeline=PipelineDefaults(
                llm_mode="ollama",
                llm_models_by_mode={"ollama": "qwen3:4b", "api": "gpt-4o"},
            ),
        )
        result = self._resolve_defaults({}, show)
        assert result["llm_mode"] == "ollama"
        assert result["llm_model"] == "qwen3:4b"

    def test_llm_model_does_not_leak_across_modes(self):
        """A model set under ollama must not surface when mode is manual."""
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(
            name="test",
            pipeline=PipelineDefaults(
                llm_mode="manual",
                llm_models_by_mode={"ollama": "qwen3:4b"},
            ),
        )
        result = self._resolve_defaults({}, show)
        assert result["llm_mode"] == "manual"
        assert "llm_model" not in result or not result["llm_model"]

    def test_app_models_by_mode_used_when_show_unset(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults(llm_mode="api"))
        result = self._resolve_defaults(
            {"llm_models_by_mode": {"api": "gpt-4o", "ollama": "qwen3"}},
            show,
        )
        assert result["llm_mode"] == "api"
        assert result["llm_model"] == "gpt-4o"

    def test_show_models_override_app_per_mode(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(
            name="test",
            pipeline=PipelineDefaults(
                llm_mode="ollama",
                llm_models_by_mode={"ollama": "show-ollama"},
            ),
        )
        result = self._resolve_defaults(
            {"llm_models_by_mode": {"ollama": "app-ollama", "api": "app-api"}},
            show,
        )
        assert result["llm_model"] == "show-ollama"


# ── Verified pointer ──────────────────────────────────────


def test_verified_unset_by_default(db):
    db.mark("ep1", transcribed=True)
    assert db.get_verified("ep1") is None
    row = db.get_episode("ep1")
    assert row["verified"] is None


def test_set_get_clear_verified(db):
    db.set_verified("ep1", "corrected", "v-123")
    ptr = db.get_verified("ep1")
    assert ptr == {"step": "corrected", "version_id": "v-123"}
    row = db.get_episode("ep1")
    assert row["verified"] == {"step": "corrected", "version_id": "v-123"}

    db.clear_verified("ep1")
    assert db.get_verified("ep1") is None


def test_set_verified_singleton_replaces(db):
    db.set_verified("ep1", "transcript", "v-1")
    db.set_verified("ep1", "corrected", "v-2")
    ptr = db.get_verified("ep1")
    assert ptr == {"step": "corrected", "version_id": "v-2"}


def test_stems_with_verified(db):
    db.set_verified("ep1", "transcript", "v-1")
    db.set_verified("ep2", "corrected", "v-2")
    db.mark("ep3", transcribed=True)
    assert db.stems_with_verified() == {"ep1", "ep2"}


def test_verified_pointers_bulk(db):
    db.set_verified("ep1", "transcript", "v-1")
    db.set_verified("ep2", "corrected", "v-2")
    pointers = db.verified_pointers()
    assert pointers == {
        "ep1": {"step": "transcript", "version_id": "v-1"},
        "ep2": {"step": "corrected", "version_id": "v-2"},
    }


def test_version_ids_by_stem(db):
    db.insert_version(
        "ep1",
        "transcript",
        {
            "id": "v-1",
            "timestamp": "2026-05-28T10:00:00Z",
            "type": "raw",
            "content_hash": "sha256:abc",
            "segment_count": 5,
        },
    )
    db.insert_version(
        "ep1",
        "transcript",
        {
            "id": "v-2",
            "timestamp": "2026-05-28T11:00:00Z",
            "type": "raw",
            "content_hash": "sha256:def",
            "segment_count": 6,
        },
    )
    db.insert_version(
        "ep2",
        "transcript",
        {
            "id": "v-3",
            "timestamp": "2026-05-28T12:00:00Z",
            "type": "raw",
            "content_hash": "sha256:ghi",
            "segment_count": 7,
        },
    )
    ids = db.version_ids_by_stem("transcript")
    assert ids == {"ep1": {"v-1", "v-2"}, "ep2": {"v-3"}}
