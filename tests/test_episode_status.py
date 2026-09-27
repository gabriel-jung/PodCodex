"""Tests for podcodex.core.episode_status: per-step status derivation and
the reconcile of status flags against the version files on disk."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path


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
    """Per-step status: done, outdated against the effective defaults, or none."""

    @staticmethod
    def step_statuses(st, provenance, effective):
        from podcodex.core.episode_status import step_statuses
        from podcodex.core.versions import clean_translations

        return step_statuses(
            st, provenance, effective, clean_translations(st.get("translations", []))
        )

    def test_none_when_not_done(self):
        st = _make_status_row()
        result = self.step_statuses(st, {}, {"model_size": "large-v3"})
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
        result = self.step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "done"
        assert result["correct_status"] == "done"

    def test_outdated_model_mismatch(self):
        prov = {"transcript": {"model": "small", "params": {"diarize": True}}}
        st = _make_status_row(transcribed=True, provenance=prov)
        effective = {"model_size": "large-v3", "diarize": True}
        result = self.step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "outdated"

    def test_outdated_diarize_mismatch(self):
        prov = {"transcript": {"model": "large-v3", "params": {"diarize": False}}}
        st = _make_status_row(transcribed=True, provenance=prov)
        effective = {"model_size": "large-v3", "diarize": True}
        result = self.step_statuses(st, prov, effective)
        assert result["transcribe_status"] == "outdated"

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
        result = self.step_statuses(st, prov, effective)
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
        result = self.step_statuses(st, prov, effective)
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
        result = self.step_statuses(st, prov, effective)
        assert result["correct_status"] == "outdated"

    def test_done_no_provenance(self):
        """Episodes without provenance default to 'done' (pre-existing episodes)."""
        st = _make_status_row(transcribed=True, corrected=True)
        result = self.step_statuses(st, {}, {"model_size": "large-v3"})
        assert result["transcribe_status"] == "done"
        assert result["correct_status"] == "done"

    def test_done_no_defaults(self):
        """No defaults configured → everything is 'done'."""
        prov = {"transcript": {"model": "small", "type": "validated", "params": {}}}
        st = _make_status_row(transcribed=True, provenance=prov)
        result = self.step_statuses(st, prov, {})
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
        result = self.step_statuses(st, prov, effective)
        assert result["translate_status"] == "done"

    def test_translate_missing_target_lang(self):
        """Target lang configured but not translated → 'none'."""
        st = _make_status_row(translations=["french"])
        effective = {"target_lang": "english"}
        result = self.step_statuses(st, {}, effective)
        assert result["translate_status"] == "none"

    def test_translate_multi_word_target_lang(self):
        """Translations are stored under normalize_lang (spaces → underscores);
        the lookup must normalize the same way, or a multi-word target
        reports 'none'."""
        prov = {
            "brazilian_portuguese": {
                "model": "gpt-4o",
                "params": {"llm_mode": "api", "llm_provider": "openai"},
            }
        }
        st = _make_status_row(translations=["brazilian_portuguese"], provenance=prov)
        effective = {
            "target_lang": "Brazilian Portuguese",
            "llm_mode": "api",
            "llm_provider": "openai",
            "llm_model": "gpt-4o",
        }
        result = self.step_statuses(st, prov, effective)
        assert result["translate_status"] == "done"

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
        result = self.step_statuses(st, prov, effective)
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
        result = self.step_statuses(st, prov, effective)
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
        result = self.step_statuses(st, prov, effective)
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
        result = self.step_statuses(st, prov, effective)
        assert result["translate_status"] == "done"


# ── Resolve defaults ─────────────────────────────────────


class TestResolveDefaults:
    """App defaults merged with a show's own pipeline settings."""

    @staticmethod
    def resolve_defaults(app_defaults, show_meta):
        from podcodex.core.episode_status import resolve_defaults

        return resolve_defaults(app_defaults, show_meta)

    def test_show_overrides_app(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults(model_size="small"))
        result = self.resolve_defaults({"model_size": "large-v3"}, show)
        assert result["model_size"] == "small"

    def test_show_empty_falls_back(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults())  # all defaults
        result = self.resolve_defaults(
            {"model_size": "large-v3", "llm_mode": "ollama"}, show
        )
        assert result["model_size"] == "large-v3"
        assert result["llm_mode"] == "ollama"

    def test_show_diarize_false_overrides(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults(diarize=False))
        result = self.resolve_defaults({"diarize": True}, show)
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
        result = self.resolve_defaults({}, show)
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
        result = self.resolve_defaults({}, show)
        assert result["llm_mode"] == "manual"
        assert "llm_model" not in result or not result["llm_model"]

    def test_app_models_by_mode_used_when_show_unset(self):
        from podcodex.ingest.show import ShowMeta, PipelineDefaults

        show = ShowMeta(name="test", pipeline=PipelineDefaults(llm_mode="api"))
        result = self.resolve_defaults(
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
        result = self.resolve_defaults(
            {"llm_models_by_mode": {"ollama": "app-ollama", "api": "app-api"}},
            show,
        )
        assert result["llm_model"] == "show-ollama"


# ── Status context and the file-scan reconcile ───────────────────────────


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


def test_srt_is_reimportable_but_not_batch_importable(tmp_path):
    """`subtitle_files` and `has_subtitles` answer different questions.

    The episode panel's manual reimport parses .vtt and .srt alike, so an
    .srt shows up in `subtitle_files`. The *batch* subtitle source is gated
    on `has_subtitles`, and `_batch_transcribe_from_subs` only ever reads a
    cached `{stem}.subtitles.{lang}.vtt` — so reporting .srt there would
    select episodes the batch run cannot process.
    """
    from podcodex.core.episode_status import build_status_out, load_status_context

    show = tmp_path / "show"
    (show / "ep").mkdir(parents=True)
    (show / "ep" / "ep.srt").touch()
    ctx = load_status_context(show)

    out = build_status_out(
        stem="ep",
        audio_path=None,
        output_dir=show / "ep",
        st={},
        ep_files=ctx.episode_files.get("ep", []),
        ctx=ctx,
    )

    assert out["subtitle_files"] == ["ep/ep.srt"]
    assert out["has_subtitles"] is False


def test_has_subtitles_only_counts_what_the_batch_can_read(tmp_path):
    """The batch reads ``{stem}.subtitles.{lang}.vtt``; a hand-uploaded
    ``{stem}.subtitles.vtt`` has no language code, so its glob never finds
    it, and the flag must not promise otherwise."""
    from podcodex.core.episode_status import build_status_out, load_status_context

    cases = {
        "ep.subtitles.en.vtt": True,
        "ep.subtitles.pt-BR.VTT": True,
        "ep.subtitles.vtt": False,
        "ep.subtitles.srt": False,
        "ep.vtt": False,
    }
    for name, expected in cases.items():
        show = tmp_path / name
        (show / "ep").mkdir(parents=True)
        (show / "ep" / name).touch()
        ctx = load_status_context(show)
        out = build_status_out(
            stem="ep",
            audio_path=None,
            output_dir=show / "ep",
            st={},
            ep_files=ctx.episode_files.get("ep", []),
            ctx=ctx,
        )
        assert out["has_subtitles"] is expected, name


def test_status_context_lists_episode_dirs_that_hold_no_files(tmp_path):
    """`episode_dirs` carries empty directories; `episode_files` cannot.

    It is what `unified_episodes` unions into the stem listing, so a suffixed
    but still-empty episode directory stays matchable.
    """
    from podcodex.core.episode_status import load_status_context

    show = tmp_path / "show"
    (show / "my_episode").mkdir(parents=True)

    ctx = load_status_context(show)

    assert "my_episode" in ctx.episode_dirs
    assert "my_episode" not in ctx.episode_files


def test_status_reconcile_keeps_flags_bootstrapped_from_disk(tmp_path):
    """A DB built from a filesystem scan must survive the reconcile pass.

    `_populate_from_scan` derives transcribed/synthesized from the step
    directories and writes no `versions` rows, so reconciling against rows
    alone would undo the bootstrap in the same call, and a DB rebuilt from
    scan would report a whole library as not started.
    """
    from podcodex.core.episode_status import load_status_context
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    show = tmp_path / "show"
    (show / "ep1" / "transcript").mkdir(parents=True)
    (show / "ep1" / "transcript" / "20260101T000000000000Z_raw.json").write_text(
        '[{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi"}]'
    )

    ctx = load_status_context(show)
    assert ctx.status_map["ep1"]["transcribed"] is True
    # And it must not have been persisted as False either.
    assert get_pipeline_db(show).get_episode("ep1")["transcribed"] is True
    close_pipeline_db(show)


def test_status_reconcile_rebuilds_the_translations_list(tmp_path):
    """Languages are the per-language equivalent of the step flags.

    A rebuilt DB restores the language versions but not this list, so the
    episode would report "not started" with its translation on disk. The
    rebuild also drops pipeline-step names legacy rows leaked in.
    """
    from podcodex.core.episode_status import load_status_context
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    show = tmp_path / "show"
    (show / "ep1" / "french").mkdir(parents=True)
    (show / "ep1" / "french" / "20260101T000000000000Z_raw.json").write_text("[]")
    get_pipeline_db(show).mark("ep1", translations=["segments", "spanish"])

    ctx = load_status_context(show)
    assert ctx.status_map["ep1"]["translations"] == ["french"]
    close_pipeline_db(show)


def test_status_reconcile_demotes_when_nothing_is_left(tmp_path):
    """The demote half still fires when neither a row nor a file remains."""
    from podcodex.core.episode_status import load_status_context
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    show = tmp_path / "show"
    (show / "ep1").mkdir(parents=True)
    get_pipeline_db(show).mark("ep1", transcribed=True)

    ctx = load_status_context(show)
    assert ctx.status_map["ep1"]["transcribed"] is False
    close_pipeline_db(show)


def test_status_reconcile_demotes_a_row_whose_file_is_gone(tmp_path):
    """Backfill keeps such a row through its grace period; every reader
    skips it, so the flag must not keep claiming the step is done."""
    from podcodex.core.episode_status import load_status_context
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db
    from podcodex.core.versions import save_version, version_path

    show = tmp_path / "show"
    (show / "ep1").mkdir(parents=True)
    base = show / "ep1" / "ep1"
    vid = save_version(
        base,
        "transcript",
        [{"speaker": "A", "start": 0.0, "end": 1.0, "text": "hi"}],
        {"step": "transcript", "type": "raw", "model": "m", "params": {}},
    )
    get_pipeline_db(show).mark("ep1", transcribed=True)
    version_path(base, "transcript", vid).unlink()

    ctx = load_status_context(show)
    assert ctx.status_map["ep1"]["transcribed"] is False
    close_pipeline_db(show)


def test_status_reconcile_skips_stems_it_could_not_scan(tmp_path):
    """An unreadable episode dir must not be read as "nothing on disk".

    Otherwise a transient EACCES on a network mount demotes the step flags
    and wipes the language list for that request.
    """
    import os

    from podcodex.core.episode_status import _EPISODE_FILES_CACHE, load_status_context
    from podcodex.core.pipeline_db import close_pipeline_db

    show = tmp_path / "show"
    (show / "ep1" / "transcript").mkdir(parents=True)
    (show / "ep1" / "transcript" / "20260101T000000000000Z_raw.json").write_text("[]")
    (show / "ep1" / "french").mkdir()
    (show / "ep1" / "french" / "20260101T000000000000Z_raw.json").write_text("[]")

    ctx = load_status_context(show)
    assert ctx.status_map["ep1"]["transcribed"] is True
    assert ctx.status_map["ep1"]["translations"] == ["french"]

    _EPISODE_FILES_CACHE.clear()
    os.chmod(show / "ep1", 0o000)
    try:
        ctx = load_status_context(show)
        assert ctx.status_map["ep1"]["transcribed"] is True
        assert ctx.status_map["ep1"]["translations"] == ["french"]
    finally:
        os.chmod(show / "ep1", 0o755)
        close_pipeline_db(show)


def test_status_reconcile_skips_everything_when_the_show_folder_is_unreadable(
    tmp_path, monkeypatch
):
    """A failed listing of the show folder is "unknown", not "empty"."""
    from podcodex.core import episode_status as status_mod
    from podcodex.core.pipeline_db import close_pipeline_db

    show = tmp_path / "show"
    (show / "ep1" / "transcript").mkdir(parents=True)
    (show / "ep1" / "transcript" / "20260101T000000000000Z_raw.json").write_text("[]")
    (show / "ep1" / "french").mkdir()
    (show / "ep1" / "french" / "20260101T000000000000Z_raw.json").write_text("[]")
    ctx = status_mod.load_status_context(show)
    assert ctx.status_map["ep1"]["transcribed"] is True

    def unreadable(folder, local_audio):
        return {}, set(), None

    monkeypatch.setattr(status_mod, "_scan_episode_files", unreadable)
    try:
        ctx = status_mod.load_status_context(show)
        assert ctx.status_map["ep1"]["transcribed"] is True
        assert ctx.status_map["ep1"]["translations"] == ["french"]
    finally:
        close_pipeline_db(show)


def test_episode_file_scan_reports_an_unlistable_folder(tmp_path):
    from podcodex.core.episode_status import _scan_episode_files

    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    files, dirs, incomplete = _scan_episode_files(not_a_dir, {})
    assert (files, dirs, incomplete) == ({}, set(), None)


def test_status_reconcile_keeps_indexed_flags_when_the_index_is_unreadable(
    tmp_path, monkeypatch
):
    from podcodex.core import episode_status as status_mod
    from podcodex.core.pipeline_db import close_pipeline_db, get_pipeline_db

    show = tmp_path / "show"
    (show / "ep1" / "transcript").mkdir(parents=True)
    (show / "ep1" / "transcript" / "20260101T000000000000Z_raw.json").write_text("[]")
    monkeypatch.setattr(status_mod, "lance_indexed_stems", lambda _p: {"ep1"})
    assert status_mod.load_status_context(show).status_map["ep1"]["indexed"]

    monkeypatch.setattr(status_mod, "lance_indexed_stems", lambda _p: None)
    try:
        ctx = status_mod.load_status_context(show)
        assert ctx.status_map["ep1"]["indexed"]
        rows = {r["stem"]: r for r in get_pipeline_db(show).all_episodes()}
        assert rows["ep1"]["indexed"]
    finally:
        close_pipeline_db(show)


def test_episode_file_scan_sees_nested_writes(tmp_path):
    """Caching is keyed on the whole recorded directory tree, not just the top.

    A version landing in `ep/transcript/` leaves `ep/`'s own mtime untouched,
    so validating only the episode directory would serve a stale file list.
    """
    from podcodex.core.episode_status import _scan_episode_files

    show = _make_episode_tree(tmp_path)
    _age_dirs(show, 60)
    assert _scan_episode_files(show, {})[0]["ep"] == [
        "ep/ep.vtt",
        "ep/transcript/v1.json",
    ]

    versions = show / "ep" / "transcript"
    (versions / "v2.json").touch()
    # Age only the subdirectory: `show/` and `ep/` keep the mtimes just
    # recorded, so nothing but the nested change can trip the cache.
    stamp = time.time() - 30
    os.utime(versions, (stamp, stamp))
    assert _scan_episode_files(show, {})[0]["ep"] == [
        "ep/ep.vtt",
        "ep/transcript/v1.json",
        "ep/transcript/v2.json",
    ]

    os.remove(versions / "v1.json")
    stamp = time.time() - 20
    os.utime(versions, (stamp, stamp))
    assert _scan_episode_files(show, {})[0]["ep"] == [
        "ep/ep.vtt",
        "ep/transcript/v2.json",
    ]

    # A removed episode directory drops out of the result and the cache.
    shutil.rmtree(show / "ep")
    assert _scan_episode_files(show, {})[0] == {}


def test_episode_file_scan_lists_the_llm_failures_file(tmp_path):
    """The walk must keep listing llm_failures.json: the llm_failed_steps
    gate reads the cached listing before touching disk, so a future trim of
    the walk's filters would silently blank the failure markers."""
    from podcodex.core.episode_status import _scan_episode_files
    from podcodex.core.llm_failures import FAILURES_FILENAME

    show = _make_episode_tree(tmp_path)
    (show / "ep" / FAILURES_FILENAME).write_text("{}", encoding="utf-8")
    _age_dirs(show, 60)

    assert f"ep/{FAILURES_FILENAME}" in _scan_episode_files(show, {})[0]["ep"]


def test_episode_file_scan_reuses_settled_results(tmp_path, monkeypatch):
    """An untouched, settled tree is served from cache instead of re-walked."""
    from podcodex.core import episode_status as status_mod

    show = _make_episode_tree(tmp_path)
    _age_dirs(show, 60)
    status_mod._scan_episode_files(show, {})  # records settled mtimes

    def _fail(*args):
        raise AssertionError("settled directories should not be re-walked")

    monkeypatch.setattr(status_mod, "_walk_episode_dir", _fail)
    assert status_mod._scan_episode_files(show, {})[0]["ep"] == [
        "ep/ep.vtt",
        "ep/transcript/v1.json",
    ]


def test_episode_file_scan_rewalks_recent_changes(tmp_path):
    """A just-written directory is never trusted, whatever its mtime says.

    FAT32 rounds mtimes to 2s, so a write can land in the same tick as the
    scan that recorded the mtime and leave it looking unchanged.
    """
    from podcodex.core.episode_status import _scan_episode_files

    show = _make_episode_tree(tmp_path)
    _scan_episode_files(show, {})

    versions = show / "ep" / "transcript"
    recorded = os.stat(versions).st_mtime
    (versions / "v2.json").touch()
    os.utime(versions, (recorded, recorded))  # simulate a same-tick write

    assert "ep/transcript/v2.json" in _scan_episode_files(show, {})[0]["ep"]
