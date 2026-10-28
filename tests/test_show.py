"""Tests for podcodex.ingest.show — show.toml persistence."""

from __future__ import annotations

from podcodex.ingest.show import ShowMeta, load_show_meta, save_show_meta


def test_pipeline_keys_roundtrip(tmp_path):
    """Top-level scalars and the [pipeline] table must round-trip together.

    Guards the TOML ordering invariant: scalars are emitted before any table
    header, otherwise tomllib silently parses them as table keys and the
    setting stops persisting.
    """
    from podcodex.ingest.show import PipelineDefaults

    meta = ShowMeta(
        name="S",
        language="French",
        broadcast_number_pattern=r"\((\d+)\)",
        pipeline=PipelineDefaults(rag_model="bge-m3", rag_chunker="semantic"),
    )
    save_show_meta(tmp_path, meta)
    loaded = load_show_meta(tmp_path)
    assert loaded.name == "S"
    assert loaded.language == "French"
    assert loaded.broadcast_number_pattern == r"\((\d+)\)"
    assert loaded.pipeline.rag_model == "bge-m3"
    assert loaded.pipeline.rag_chunker == "semantic"


def test_control_characters_round_trip(tmp_path):
    """A CR from a pasted feed title used to write a show.toml that no longer
    parsed, which then lost the show its id."""
    from podcodex.ingest.show import PipelineDefaults

    hostile = 'a\r\nb\tc"d\\e\x01f\x7fg 😀'
    meta = ShowMeta(
        name=hostile,
        speakers=[hostile],
        pipeline=PipelineDefaults(context=hostile, llm_models_by_mode={"api": hostile}),
    )
    save_show_meta(tmp_path, meta)
    loaded = load_show_meta(tmp_path)
    assert loaded is not None
    assert loaded.id == meta.id
    assert loaded.name == hostile
    assert loaded.speakers == [hostile]
    assert loaded.pipeline.context == hostile
    assert loaded.pipeline.llm_models_by_mode == {"api": hostile}


def test_unreadable_show_toml_is_never_minted_over(tmp_path):
    """The file still holds the show's id; overwriting it orphans the index."""
    import pytest

    from podcodex.ingest.show import ShowMetaUnreadableError, ensure_show_id

    path = tmp_path / "show.toml"
    path.write_text('id = "kept_1234abcd"\nname = "broken', encoding="utf-8")
    with pytest.raises(ShowMetaUnreadableError):
        ensure_show_id(tmp_path)
    with pytest.raises(ShowMetaUnreadableError):
        save_show_meta(tmp_path, ShowMeta(name="fresh"))
    assert "kept_1234abcd" in path.read_text(encoding="utf-8")


def test_loaded_meta_is_a_copy(tmp_path):
    """An unsaved mutation must not leak to the next reader through the cache."""
    import os
    import time

    save_show_meta(tmp_path, ShowMeta(name="Original"))
    past = time.time() - 60
    os.utime(tmp_path / "show.toml", (past, past))  # settled, so it is cached
    for _ in range(2):  # the first read parses, the second is a cache hit
        load_show_meta(tmp_path).name = "Mutated"
        assert load_show_meta(tmp_path).name == "Original"


def test_failed_write_leaves_no_phantom_id(tmp_path, monkeypatch):
    """The id is assigned only once show.toml is on disk."""
    import pytest

    import podcodex.core._utils as utils

    def _fail(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(utils, "atomic_write", _fail)
    meta = ShowMeta(name="S")
    with pytest.raises(OSError):
        save_show_meta(tmp_path, meta)
    assert meta.id == ""
    assert load_show_meta(tmp_path) is None


def test_a_lock_on_show_toml_is_not_reported_as_corruption(tmp_path, monkeypatch):
    """A sync client's lock is transient; telling the user to remove the file
    could cost them the id."""
    from pathlib import Path

    import pytest

    from podcodex.ingest.show import ShowMetaUnreadableError

    save_show_meta(tmp_path, ShowMeta(name="S"))
    real = Path.read_text

    def _locked(self, *a, **k):
        if self.name == "show.toml":
            raise PermissionError("locked")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", _locked)
    with pytest.raises(OSError) as exc:
        save_show_meta(tmp_path, ShowMeta(name="fresh"))
    assert not isinstance(exc.value, ShowMetaUnreadableError)


def test_a_lock_that_hid_the_id_does_not_mint_over_it(tmp_path):
    """load_show_meta returned None during a lock; by save time the file
    reads fine and holds an id, which a fresh meta must not replace."""
    import pytest

    save_show_meta(tmp_path, ShowMeta(name="S"))
    original = load_show_meta(tmp_path).id
    with pytest.raises(OSError):
        save_show_meta(tmp_path, ShowMeta(name="fallback"))
    assert load_show_meta(tmp_path).id == original


def test_salvage_reads_the_id_line_of_a_broken_file(tmp_path):
    from podcodex.ingest.show import salvage_show_id

    (tmp_path / "show.toml").write_text('id = "kept_1234abcd"\nname = "broken')
    assert salvage_show_id(tmp_path) == "kept_1234abcd"


def test_a_string_speaker_roster_is_one_speaker(tmp_path):
    (tmp_path / "show.toml").write_text('id = "s_1"\nname = "S"\nspeakers = "Alice"\n')
    assert load_show_meta(tmp_path).speakers == ["Alice"]
