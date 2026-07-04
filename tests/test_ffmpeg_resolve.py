"""Tests for the winget package-payload ffmpeg fallback scan."""

from pathlib import Path

from podcodex.core._ffmpeg import _winget_ffmpeg_payload_dirs


def _make_payload(root: Path, pkg: str, build: str) -> Path:
    bin_dir = root / pkg / build / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ffmpeg.exe").write_bytes(b"")
    return bin_dir


def test_finds_gyan_layout(tmp_path):
    bin_dir = _make_payload(
        tmp_path,
        "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
        "ffmpeg-7.1-full_build",
    )
    assert _winget_ffmpeg_payload_dirs(tmp_path) == [str(bin_dir)]


def test_newest_build_first(tmp_path):
    pkg = "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    old = _make_payload(tmp_path, pkg, "ffmpeg-6.1-full_build")
    new = _make_payload(tmp_path, pkg, "ffmpeg-7.1-full_build")
    assert _winget_ffmpeg_payload_dirs(tmp_path) == [str(new), str(old)]


def test_flat_bin_layout(tmp_path):
    pkg_dir = tmp_path / "Gyan.FFmpeg.Essentials_Source"
    bin_dir = pkg_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ffmpeg.exe").write_bytes(b"")
    assert _winget_ffmpeg_payload_dirs(tmp_path) == [str(bin_dir)]


def test_ignores_unrelated_packages(tmp_path):
    _make_payload(tmp_path, "SomeOther.Tool_Source", "tool-1.0")
    assert _winget_ffmpeg_payload_dirs(tmp_path) == []


def test_missing_root_returns_empty(tmp_path):
    assert _winget_ffmpeg_payload_dirs(tmp_path / "nope") == []
