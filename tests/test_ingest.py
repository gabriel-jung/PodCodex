"""Tests for podcodex.ingest.folder — filesystem mocked with tmp_path."""

from unittest.mock import patch


def _make_status(exported=False):
    return {
        "transcribed": False,
        "diarized": False,
        "assigned": False,
        "mapped": False,
        "exported": exported,
    }


# ──────────────────────────────────────────────
# Audio file detection
# ──────────────────────────────────────────────


def test_scan_folder_finds_audio_only(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    (tmp_path / "ep02.wav").touch()
    (tmp_path / "notes.json").touch()
    (tmp_path / "subdir").mkdir()

    with patch("podcodex.ingest.folder.processing_status", return_value=_make_status()):
        result = scan_folder(tmp_path)

    assert {ep.stem for ep in result} == {"ep01", "ep02"}


def test_scan_folder_ignores_non_audio(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    (tmp_path / "ep01.transcript.json").touch()
    (tmp_path / "ep01.segments.parquet").touch()
    (tmp_path / "image.png").touch()
    (tmp_path / "README.md").touch()

    with patch("podcodex.ingest.folder.processing_status", return_value=_make_status()):
        result = scan_folder(tmp_path)

    assert len(result) == 1
    assert result[0].stem == "ep01"


def test_scan_folder_all_audio_extensions(tmp_path):
    from podcodex.ingest.folder import scan_folder

    for name in ("a.mp3", "b.wav", "c.m4a", "d.ogg", "e.flac"):
        (tmp_path / name).touch()

    with patch("podcodex.ingest.folder.processing_status", return_value=_make_status()):
        result = scan_folder(tmp_path)

    assert len(result) == 5


# ──────────────────────────────────────────────
# output_dir derivation
# ──────────────────────────────────────────────


def test_scan_folder_output_dir_derivation(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    with patch("podcodex.ingest.folder.processing_status", return_value=_make_status()):
        result = scan_folder(tmp_path)

    assert result[0].output_dir == tmp_path / "ep01"


# ──────────────────────────────────────────────
# transcribed flag
# ──────────────────────────────────────────────


def test_scan_folder_transcribed_true(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    with patch(
        "podcodex.ingest.folder.processing_status",
        return_value=_make_status(exported=True),
    ):
        result = scan_folder(tmp_path)

    assert result[0].transcribed is True


def test_scan_folder_transcribed_false(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    with patch(
        "podcodex.ingest.folder.processing_status",
        return_value=_make_status(exported=False),
    ):
        result = scan_folder(tmp_path)

    assert result[0].transcribed is False


# ──────────────────────────────────────────────
# indexed flag
# ──────────────────────────────────────────────


def test_scan_folder_indexed_true(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()
    ep_dir = tmp_path / "ep01"
    ep_dir.mkdir()
    (ep_dir / ".rag_indexed").touch()

    with patch("podcodex.ingest.folder.processing_status", return_value=_make_status()):
        result = scan_folder(tmp_path)

    assert result[0].indexed is True


def test_scan_folder_indexed_false(tmp_path):
    from podcodex.ingest.folder import scan_folder

    (tmp_path / "ep01.mp3").touch()

    with patch("podcodex.ingest.folder.processing_status", return_value=_make_status()):
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

    with patch("podcodex.ingest.folder.processing_status", return_value=_make_status()):
        result = scan_folder(tmp_path)

    assert [ep.stem for ep in result] == ["aaa", "mmm", "zzz"]
