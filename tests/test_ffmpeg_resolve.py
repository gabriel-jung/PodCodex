"""Tests for ffmpeg resolution: override order and the Windows fallbacks."""

import os
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


# ── Resolution order ─────────────────────────────────────────────────────


def _exe(tmp_path, name):
    path = tmp_path / name / "ffmpeg"
    path.parent.mkdir(parents=True)
    path.write_text("")
    return str(path)


def _setup(monkeypatch, *, env="", config="", which=None, extra=()):
    from podcodex.core import _ffmpeg
    from podcodex.core.app_config import AppConfig

    if env:
        monkeypatch.setenv(_ffmpeg.PODCODEX_FFMPEG_EXE_ENV, env)
    else:
        monkeypatch.delenv(_ffmpeg.PODCODEX_FFMPEG_EXE_ENV, raising=False)
    monkeypatch.setattr(
        _ffmpeg, "load_config", lambda: AppConfig(ffmpeg_exe_override=config)
    )

    def fake_which(cmd, path=None):
        if path is None:
            return which
        for d in path.split(os.pathsep):
            if os.path.isfile(os.path.join(d, cmd)):
                return os.path.join(d, cmd)
        return None

    monkeypatch.setattr(_ffmpeg.shutil, "which", fake_which)
    monkeypatch.setattr(_ffmpeg, "_windows_extra_dirs", lambda: tuple(extra))
    return _ffmpeg


def test_env_override_beats_config_and_path(tmp_path, monkeypatch):
    env, cfg = _exe(tmp_path, "env"), _exe(tmp_path, "cfg")
    ff = _setup(monkeypatch, env=env, config=cfg, which="/usr/bin/ffmpeg")
    assert ff.ffmpeg_exe() == env


def test_config_override_beats_path_and_is_quote_tolerant(tmp_path, monkeypatch):
    cfg = _exe(tmp_path, "cfg")
    ff = _setup(monkeypatch, config=f'  "{cfg}"  ', which="/usr/bin/ffmpeg")
    assert ff.ffmpeg_exe() == cfg


def test_an_override_that_does_not_exist_falls_through(tmp_path, monkeypatch):
    ff = _setup(monkeypatch, env=str(tmp_path / "gone"), which="/usr/bin/ffmpeg")
    assert ff.ffmpeg_exe() == "/usr/bin/ffmpeg"


def test_nothing_found_returns_the_bare_command(monkeypatch):
    ff = _setup(monkeypatch)
    assert ff.ffmpeg_exe() == "ffmpeg"
    assert not ff.ffmpeg_available()


def test_a_fallback_hit_prepends_its_dir_to_path_once(tmp_path, monkeypatch):
    """whisperx and faster-whisper call a bare "ffmpeg"; they must find the
    binary the fallback found, and repeated calls must not grow PATH."""
    found = _exe(tmp_path, "winget")
    ff = _setup(monkeypatch, extra=[os.path.dirname(found)])
    monkeypatch.setenv("PATH", "/usr/bin")

    assert ff.ffmpeg_exe() == found
    assert ff.ffmpeg_exe() == found
    parts = os.environ["PATH"].split(os.pathsep)
    assert parts[0] == os.path.dirname(found)
    assert parts.count(os.path.dirname(found)) == 1
