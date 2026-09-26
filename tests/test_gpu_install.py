"""GPU backend install: the path that replaces the executed sidecar binary.

Archives are served from ``file://`` URLs, which ``urllib`` opens like any
other, so the real download, sha check, extraction and markers all run.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from podcodex.api import gpu_backend

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the fake sidecar is a POSIX shell script"
)


def _tar(path: Path, files: dict[str, tuple[bytes, int]]) -> str:
    with tarfile.open(path, "w:gz") as tar:
        for name, (data, mode) in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A fake release dir and an install dir, with bundle checks bypassed."""
    install = tmp_path / "install"
    monkeypatch.setattr(gpu_backend, "gpu_install_dir", lambda: install)
    monkeypatch.setattr(gpu_backend, "_ensure_bundle_mode", lambda: None)
    monkeypatch.setattr(gpu_backend, "_ensure_platform_supported", lambda: None)
    monkeypatch.setattr(gpu_backend, "_app_version", lambda: "1.2.3")
    rel = tmp_path / "release"
    rel.mkdir()

    def build(server_reports="1.2.3", manifest_server="1.2.3", bad_sha=False):
        script = f"#!/bin/sh\necho podcodex {server_reports}\n".encode()
        server_sha = _tar(
            rel / "server-core.tar.gz", {"podcodex-server-gpu": (script, 0o755)}
        )
        libs_sha = _tar(rel / "cuda-libs-cu1.tar.gz", {"libcuda.so": (b"x", 0o644)})
        manifest = {
            "version": "cu1",
            "archive": "cuda-libs-cu1.tar.gz",
            "sha256": libs_sha,
            "server_sha256": "0" * 64 if bad_sha else server_sha,
        }
        if manifest_server is not None:
            manifest["server_version"] = manifest_server
        (rel / "cuda-libs.json").write_text(json.dumps(manifest))
        return (rel / "cuda-libs.json").as_uri()

    return install, build


def _cb(*_a):
    return None


def test_a_clean_install_is_trusted_and_keeps_activation(release):
    install, build = release
    install.mkdir()
    (install / "activated").touch()
    result = gpu_backend.download_and_install(_cb, manifest_url=build())
    assert result["downloaded_server"] and result["downloaded_libs"]
    assert gpu_backend.installed_manifest()["version"] == "cu1"
    assert gpu_backend.installed_server_core_version() == "1.2.3"
    assert (install / "activated").is_file()
    assert not (install / ".installing").exists()


def test_a_sha_mismatch_leaves_the_install_untouched(release):
    install, build = release
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        gpu_backend.download_and_install(_cb, manifest_url=build(bad_sha=True))
    assert not (install / "podcodex-server-gpu").exists()
    assert gpu_backend.installed_manifest() is None


def test_a_manifest_for_another_release_is_refused(release):
    install, build = release
    with pytest.raises(RuntimeError, match="PodCodex 9.9.9"):
        gpu_backend.download_and_install(
            _cb, manifest_url=build(manifest_server="9.9.9")
        )
    assert not (install / "podcodex-server-gpu").exists()


def test_a_server_core_of_another_version_leaves_the_install_alone(release):
    """Older manifests carry no server_version; the extracted binary decides,
    in staging, before anything overwrites the working install."""
    install, build = release
    gpu_backend.download_and_install(_cb, manifest_url=build())
    (install / "activated").touch()

    url = build(server_reports="1.2.2", manifest_server=None)
    # Force the server core to look stale so it is downloaded again.
    (install / "podcodex-server-gpu").write_text("#!/bin/sh\necho podcodex 1.2.1\n")
    (install / "podcodex-server-gpu").chmod(0o755)
    with pytest.raises(RuntimeError, match="1.2.2"):
        gpu_backend.download_and_install(_cb, manifest_url=url)
    assert (install / "activated").exists()
    assert not (install / ".installing").exists()
    assert gpu_backend.installed_server_core_version() == "1.2.1"


def test_an_interrupted_install_is_never_trusted(release):
    install, build = release
    gpu_backend.download_and_install(_cb, manifest_url=build())
    gpu_backend._begin_extraction()  # a crash right after extraction began
    assert not (install / "activated").exists()
    assert gpu_backend.installed_manifest() is None
    assert gpu_backend.installed_server_core_version() is None
    with pytest.raises(RuntimeError, match="did not finish"):
        gpu_backend.activate()


def test_an_interrupted_download_is_retried(release, monkeypatch):
    """A dropped connection retries instead of failing the whole install."""
    import http.client

    install, build = release
    url = build()
    real = gpu_backend._download_once
    calls: list[str] = []

    def flaky(u, dest, *a, **k):
        calls.append(u)
        if calls.count(u) == 1 and u.endswith("cuda-libs-cu1.tar.gz"):
            dest.write_bytes(Path(u.removeprefix("file://")).read_bytes()[:5])
            raise http.client.IncompleteRead(b"", 10)
        return real(u, dest, *a, **k)

    monkeypatch.setattr(gpu_backend, "_download_once", flaky)
    monkeypatch.setattr(gpu_backend, "_DOWNLOAD_BACKOFF_CAP_S", 0.0)
    gpu_backend.download_and_install(_cb, manifest_url=url)
    assert sum(u.endswith("cuda-libs-cu1.tar.gz") for u in calls) == 2
    assert gpu_backend.installed_manifest()["version"] == "cu1"


def test_drops_that_keep_making_progress_do_not_exhaust_the_retries(
    tmp_path, monkeypatch
):
    """Resumed attempts that move bytes reset the budget; only stalls count."""
    import http.client

    monkeypatch.setattr(gpu_backend, "_DOWNLOAD_BACKOFF_CAP_S", 0.0)
    attempts = {"n": 0}

    def drop_after_progress(url, dest, *_a, **_k):
        attempts["n"] += 1
        with open(dest, "ab") as f:
            f.write(b"x")
        if attempts["n"] < 2 * gpu_backend._DOWNLOAD_ATTEMPTS:
            raise http.client.IncompleteRead(b"", 1)

    monkeypatch.setattr(gpu_backend, "_download_once", drop_after_progress)
    dest = tmp_path / "big.tar.gz"
    gpu_backend._stream_download("u", dest, _cb, None, 0.0, 1.0, "cuda libs")
    assert dest.stat().st_size == 2 * gpu_backend._DOWNLOAD_ATTEMPTS


def test_stalled_attempts_give_up_with_a_sentence(tmp_path, monkeypatch):
    import urllib.error

    monkeypatch.setattr(gpu_backend, "_DOWNLOAD_BACKOFF_CAP_S", 0.0)

    def never(*_a, **_k):
        raise urllib.error.URLError("network is unreachable")

    monkeypatch.setattr(gpu_backend, "_download_once", never)
    with pytest.raises(RuntimeError, match="kept failing"):
        gpu_backend._stream_download(
            "u", tmp_path / "f", _cb, None, 0.0, 1.0, "cuda libs"
        )


def test_activate_and_uninstall_refuse_while_a_download_runs(release):
    install, build = release
    gpu_backend.download_and_install(_cb, manifest_url=build())
    with gpu_backend._INSTALL_LOCK:
        with pytest.raises(RuntimeError, match="being downloaded"):
            gpu_backend.uninstall()
        with pytest.raises(RuntimeError, match="being downloaded"):
            gpu_backend.activate()
    assert (install / "podcodex-server-gpu").exists()
