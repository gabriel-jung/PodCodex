"""Tests for podcodex.ingest.show — show.toml persistence."""

from __future__ import annotations

from podcodex.ingest.show import ShowMeta, load_show_meta, save_show_meta


def test_broadcast_pattern_roundtrip(tmp_path):
    meta = ShowMeta(name="Total Trax", broadcast_number_pattern=r"^\((\d+)\)")
    save_show_meta(tmp_path, meta)
    loaded = load_show_meta(tmp_path)
    assert loaded is not None
    assert loaded.broadcast_number_pattern == r"^\((\d+)\)"


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
