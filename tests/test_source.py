"""`core/source.build_index_transcript` and `scan_show_stems`.

`build_index_transcript` is the only code that copies episode metadata
(`rss_pub_date` and the rest) into what the indexer stores; the reindex
tests stub it out, and sparse metadata silently breaks RAG date filters
(CLAUDE.md, episode metadata flow).
"""

from __future__ import annotations

from podcodex.core.source import build_index_transcript, scan_show_stems
from podcodex.core.versions import save_version
from podcodex.ingest.rss import RSSEpisode, save_episode_meta
from podcodex.ingest.show import ShowMeta, save_show_meta
from tests.fixtures.episodes import make_episode, prov

SEGS = [{"speaker": "A", "text": "Episode twelve starts.", "start": 0.0, "end": 1.0}]


def _episode(tmp_path, *, meta: RSSEpisode | None = None):
    base, audio = make_episode(tmp_path)
    if meta is not None:
        save_episode_meta(base.parent, meta)
    return base.parent.parent, str(audio)


def test_episode_metadata_reaches_the_index_meta(tmp_path):
    show, audio = _episode(
        tmp_path,
        meta=RSSEpisode(
            guid="g",
            title="#12 - The river",
            pub_date="2026-03-01T10:00:00+00:00",
            description="About a river.",
            audio_url="https://example.com/ep.mp3",
            episode_number=4,
            artwork_url="https://example.com/art.jpg",
            youtube_id="abc123",
        ),
    )
    save_show_meta(show, ShowMeta(name="Show", broadcast_number_pattern=r"#(\d+)"))

    meta = build_index_transcript(audio, "Show", "ep", segments=SEGS)["meta"]

    assert meta["show"] == "Show" and meta["episode"] == "ep"
    assert meta["rss_title"] == "#12 - The river"
    assert meta["rss_pub_date"] == "2026-03-01T10:00:00+00:00"
    assert meta["episode_number"] == 4
    assert meta["rss_description"] == "About a river."
    assert meta["rss_audio_url"] == "https://example.com/ep.mp3"
    assert meta["rss_artwork_url"] == "https://example.com/art.jpg"
    assert meta["youtube_id"] == "abc123"
    assert meta["broadcast_number"] == 12


def test_absent_metadata_is_omitted_not_invented(tmp_path):
    _show, audio = _episode(tmp_path)
    meta = build_index_transcript(audio, "Show", "ep", segments=SEGS)["meta"]
    assert set(meta) == {"show", "episode", "source"}


def test_segments_resolve_from_the_version_db(tmp_path):
    show, audio = _episode(tmp_path)
    save_version(
        show / "ep" / "ep",
        "transcript",
        SEGS,
        prov("transcript"),
    )
    transcript = build_index_transcript(audio, "Show", "ep")
    assert transcript["segments"] == SEGS


def test_scan_show_stems_separates_dirs_from_downloaded_audio(tmp_path):
    show = tmp_path / "show"
    (show / "only-dir").mkdir(parents=True)
    (show / ".hidden").mkdir()
    (show / "ep1.mp3").write_bytes(b"x")
    (show / "notes.txt").write_text("x")

    stems, audio = scan_show_stems(show)

    assert stems == {"only-dir", "ep1"}
    assert audio == {"ep1"}


def test_a_correction_pin_must_be_a_transcript(tmp_path):
    """A stale picker value naming a translation used to be corrected as if
    it were the transcript."""
    import pytest

    from podcodex.core.source import load_source, resolve_source_ref

    base, audio = make_episode(tmp_path)
    french = save_version(base, "french", SEGS, prov("french"))

    with pytest.raises(ValueError, match="not transcript"):
        load_source(str(audio), None, french, step="transcript")
    assert resolve_source_ref(str(audio), None, french, step="transcript") is None
    assert load_source(str(audio), None, french).step == "french"


def test_the_ref_names_the_version_load_source_reads(tmp_path):
    from podcodex.core.source import load_source, resolve_source_ref

    base, audio = make_episode(tmp_path)
    save_version(base, "transcript", SEGS, prov("transcript"))
    save_version(base, "corrected", SEGS, prov("corrected"))

    for step in ("auto", "transcript"):
        loaded = load_source(str(audio), None, step=step)
        ref = resolve_source_ref(str(audio), None, step=step)
        assert (ref.step, ref.version_id) == (loaded.step, loaded.version_id)
