"""Tests for podcodex.ingest.folder — filesystem mocked with tmp_path."""


# ──────────────────────────────────────────────
# Audio file detection
# ──────────────────────────────────────────────


def test_scan_folder_finds_audio_only(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    (tmp_path / "ep02.wav").touch()
    (tmp_path / "notes.json").touch()
    (tmp_path / "subdir").mkdir()

    result = scan_folder(tmp_path)

    assert {ep.stem for ep in result} == {"ep01", "ep02"}


def test_scan_folder_ignores_non_audio(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    (tmp_path / "ep01.transcript.json").touch()
    (tmp_path / "ep01.segments.parquet").touch()
    (tmp_path / "image.png").touch()
    (tmp_path / "README.md").touch()

    result = scan_folder(tmp_path)

    assert len(result) == 1
    assert result[0].stem == "ep01"


def test_scan_folder_all_audio_extensions(tmp_path):
    from podcodex.ingest.folder import scan_folder

    for name in ("a.mp3", "b.wav", "c.m4a", "d.ogg", "e.flac"):
        (tmp_path / name).touch()

    result = scan_folder(tmp_path)

    assert len(result) == 5


# ──────────────────────────────────────────────
# output_dir derivation
# ──────────────────────────────────────────────


def test_scan_folder_output_dir_derivation(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    result = scan_folder(tmp_path)

    assert result[0].output_dir == tmp_path / "ep01"


# ──────────────────────────────────────────────
# transcribed flag
# ──────────────────────────────────────────────


def test_scan_folder_transcribed_true(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "ep01.transcript.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].transcribed is True


def test_scan_folder_transcribed_via_raw(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "ep01.transcript.raw.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].transcribed is True
    assert result[0].raw_transcript is True
    assert result[0].validated_transcript is False


def test_scan_folder_transcribed_false(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    result = scan_folder(tmp_path)

    assert result[0].transcribed is False


def test_scan_folder_validated_transcript(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "ep01.transcript.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].validated_transcript is True
    assert result[0].raw_transcript is False


# ──────────────────────────────────────────────
# polished flags
# ──────────────────────────────────────────────


def test_scan_folder_polished_validated(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "ep01.polished.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].polished is True
    assert result[0].validated_polished is True
    assert result[0].raw_polished is False


def test_scan_folder_polished_raw_only(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "ep01.polished.raw.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].polished is True
    assert result[0].raw_polished is True
    assert result[0].validated_polished is False


# ──────────────────────────────────────────────
# translations flags
# ──────────────────────────────────────────────


def test_scan_folder_translation_validated(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "ep01.english.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].translations == ["english"]
    assert result[0].validated_translations == ["english"]
    assert result[0].raw_translations == []


def test_scan_folder_translation_raw_only(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "ep01.english.raw.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].translations == ["english"]
    assert result[0].raw_translations == ["english"]
    assert result[0].validated_translations == []


def test_scan_folder_translation_both_raw_and_validated(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "ep01.english.json").touch()
    (ep_dir / "ep01.english.raw.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].translations == ["english"]
    assert result[0].validated_translations == ["english"]
    assert result[0].raw_translations == []  # validated exists, so not raw-only


def test_scan_folder_internal_suffixes_excluded(tmp_path):
    """transcript.json, polished.json etc. are not treated as translations."""
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / "ep01.transcript.json").touch()
    (ep_dir / "ep01.polished.json").touch()
    (ep_dir / "ep01.speaker_map.json").touch()
    (ep_dir / "ep01.segments.meta.json").touch()
    (ep_dir / "ep01.diarization.meta.json").touch()

    result = scan_folder(tmp_path)

    assert result[0].translations == []


# ──────────────────────────────────────────────
# indexed flag
# ──────────────────────────────────────────────


def test_scan_folder_indexed_true(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / ".rag_indexed").touch()

    result = scan_folder(tmp_path)

    assert result[0].indexed is True


def test_scan_folder_indexed_false(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    result = scan_folder(tmp_path)

    assert result[0].indexed is False


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────


def test_scan_folder_empty(tmp_path):
    from podcodex.ingest.folder import scan_folder

    result = scan_folder(tmp_path)

    assert result == []


def test_scan_folder_sorted(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "zzz.mp3").touch()
    (tmp_path / "aaa.wav").touch()
    (tmp_path / "mmm.flac").touch()

    result = scan_folder(tmp_path)

    assert [ep.stem for ep in result] == ["aaa", "mmm", "zzz"]


def test_scan_folder_no_output_dir_yet(tmp_path):
    """Episodes with no output_dir yet should have all flags False."""
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    result = scan_folder(tmp_path)
    ep = result[0]

    assert ep.transcribed is False
    assert ep.polished is False
    assert ep.indexed is False
    assert ep.translations == []
    assert ep.raw_transcript is False
    assert ep.validated_transcript is False
