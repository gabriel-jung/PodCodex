"""Tests for podcodex.ingest.show — show.toml persistence."""

from __future__ import annotations

from podcodex.ingest.show import ShowMeta, load_show_meta, save_show_meta


def test_speaker_aliases_roundtrip(tmp_path):
    meta = ShowMeta(name="My Show", speaker_aliases={"Raf": "Rafik"})
    save_show_meta(tmp_path, meta)
    loaded = load_show_meta(tmp_path)
    assert loaded is not None
    assert loaded.speaker_aliases == {"Raf": "Rafik"}


def test_broadcast_pattern_roundtrip(tmp_path):
    meta = ShowMeta(name="Total Trax", broadcast_number_pattern=r"^\((\d+)\)")
    save_show_meta(tmp_path, meta)
    loaded = load_show_meta(tmp_path)
    assert loaded is not None
    assert loaded.broadcast_number_pattern == r"^\((\d+)\)"


def test_alias_table_does_not_swallow_pipeline_keys(tmp_path):
    from podcodex.ingest.show import PipelineDefaults

    meta = ShowMeta(
        name="S",
        speaker_aliases={"Raf": "Rafik"},
        pipeline=PipelineDefaults(rag_model="bge-m3", rag_chunker="semantic"),
    )
    save_show_meta(tmp_path, meta)
    loaded = load_show_meta(tmp_path)
    assert loaded.pipeline.rag_model == "bge-m3"
    assert loaded.pipeline.rag_chunker == "semantic"
    assert loaded.speaker_aliases == {"Raf": "Rafik"}
