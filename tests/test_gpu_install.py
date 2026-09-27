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
import threading
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
    """A fake release dir and an install root, with bundle checks bypassed.

    ``build`` writes a schema-2 release (server core, torch runtime, cuda
    libs) and returns its manifest URL; ``app["version"]`` is the running
    app's version.
    """
    root = tmp_path / "gpu"
    app = {"version": "1.2.3"}
    monkeypatch.setattr(gpu_backend, "gpu_install_dir", lambda: root)
    monkeypatch.setattr(gpu_backend, "_ensure_bundle_mode", lambda: None)
    monkeypatch.setattr(gpu_backend, "_ensure_platform_supported", lambda: None)
    monkeypatch.setattr(gpu_backend, "_app_version", lambda: app["version"])
    rel = tmp_path / "release"
    rel.mkdir()

    def build(
        server_reports=None,
        manifest_server="",
        bad_sha=(),
        torch=b"torch-1",
        cuda=b"cuda-1",
    ):
        server_reports = server_reports or app["version"]
        manifest_server = app["version"] if manifest_server == "" else manifest_server
        script = f"#!/bin/sh\necho podcodex {server_reports}\n".encode()
        specs = {
            "server-core": (
                "server-core.tar.gz",
                {"podcodex-server-gpu": (script, 0o755)},
                server_reports,
            ),
            "torch-runtime": (
                f"torch-runtime-{hashlib.sha256(torch).hexdigest()[:8]}.tar.gz",
                {"_internal/torch/lib/libtorch_cuda.so": (torch, 0o644)},
                hashlib.sha256(torch).hexdigest()[:8],
            ),
            "cuda-libs": (
                f"cuda-libs-{hashlib.sha256(cuda).hexdigest()[:8]}.tar.gz",
                {"_internal/torch/lib/libcudnn.so": (cuda, 0o644)},
                hashlib.sha256(cuda).hexdigest()[:8],
            ),
        }
        parts = []
        for name, (archive, files, tag) in specs.items():
            sha = _tar(rel / archive, files)
            parts.append(
                {
                    "name": name,
                    "tag": tag,
                    "archive": archive,
                    "sha256": "0" * 64 if name in bad_sha else sha,
                }
            )
        manifest = {"schema": 2, "parts": parts, "version": parts[2]["tag"]}
        if manifest_server is not None:
            manifest["server_version"] = manifest_server
        (rel / "cuda-libs.json").write_text(json.dumps(manifest))
        return (rel / "cuda-libs.json").as_uri()

    return root, build, app


def _cb(*_a):
    return None


def _active(root: Path) -> Path:
    return root / (root / "current").read_text().strip()


def _no_leftovers(root: Path) -> bool:
    return not any(e.name.startswith(".") for e in root.iterdir())


def test_a_clean_install_is_trusted_and_keeps_activation(release):
    root, build, _app = release
    root.mkdir()
    (root / "activated").touch()

    result = gpu_backend.download_and_install(_cb, manifest_url=build())

    assert result["downloaded"] == ["server-core", "torch-runtime", "cuda-libs"]
    assert gpu_backend.active_install_dir() == _active(root)
    assert gpu_backend.installed_server_core_version() == "1.2.3"
    assert gpu_backend.is_gpu_activated()
    assert (_active(root) / "_internal/torch/lib/libcudnn.so").read_bytes() == b"cuda-1"
    assert _no_leftovers(root)


def test_an_app_update_downloads_only_the_server_core(release, monkeypatch):
    root, build, app = release
    gpu_backend.download_and_install(_cb, manifest_url=build())
    old = _active(root)
    # The running GPU sidecar is the old install: it must survive the prune,
    # and nothing may be written through the links it shares.
    monkeypatch.setattr(sys, "executable", str(old / "podcodex-server-gpu"))

    app["version"] = "1.2.4"
    result = gpu_backend.download_and_install(_cb, manifest_url=build())

    new = _active(root)
    assert new != old
    assert result["downloaded"] == ["server-core"]
    assert result["reused"] == ["torch-runtime", "cuda-libs"]
    assert gpu_backend.installed_server_core_version() == "1.2.4"
    lib = "_internal/torch/lib/libcudnn.so"
    assert (new / lib).stat().st_ino == (old / lib).stat().st_ino  # linked
    assert gpu_backend._probe_server_core_version(old) == "1.2.3"  # untouched
    assert _no_leftovers(root)


def test_a_torch_bump_downloads_torch_but_not_cuda(release):
    root, build, _app = release
    gpu_backend.download_and_install(_cb, manifest_url=build())

    result = gpu_backend.download_and_install(_cb, manifest_url=build(torch=b"torch-2"))

    assert result["downloaded"] == ["torch-runtime"]
    new = _active(root)
    assert (new / "_internal/torch/lib/libtorch_cuda.so").read_bytes() == b"torch-2"
    assert (new / "_internal/torch/lib/libcudnn.so").read_bytes() == b"cuda-1"
    # The old install is gone: nothing runs from it.
    assert [e.name for e in root.iterdir() if e.name.startswith("v-")] == [new.name]


def test_up_to_date_downloads_nothing(release):
    _root, build, _app = release
    url = build()
    gpu_backend.download_and_install(_cb, manifest_url=url)

    assert gpu_backend.download_and_install(_cb, manifest_url=url)["skipped"]


def test_a_failed_update_leaves_the_working_install_active(release):
    root, build, _app = release
    gpu_backend.download_and_install(_cb, manifest_url=build())
    (root / "activated").touch()
    before = _active(root)

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        gpu_backend.download_and_install(
            _cb, manifest_url=build(cuda=b"cuda-2", bad_sha=("cuda-libs",))
        )

    assert _active(root) == before
    assert (before / "_internal/torch/lib/libcudnn.so").read_bytes() == b"cuda-1"
    assert gpu_backend.is_gpu_activated()
    assert _no_leftovers(root)


def test_a_cancelled_update_leaves_the_working_install_active(release):
    root, build, _app = release
    gpu_backend.download_and_install(_cb, manifest_url=build())
    before = _active(root)
    cancel = threading.Event()

    def cb(_frac, msg):
        if msg.startswith("Verifying"):
            cancel.set()  # the user cancels once the download finished

    cb.cancel_event = cancel
    from podcodex.api.tasks import TaskCancelled

    with pytest.raises(TaskCancelled, match="unchanged"):
        gpu_backend.download_and_install(cb, manifest_url=build(torch=b"torch-2"))

    assert _active(root) == before
    assert _no_leftovers(root)


def test_a_sha_mismatch_on_a_fresh_install_installs_nothing(release):
    root, build, _app = release
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        gpu_backend.download_and_install(
            _cb, manifest_url=build(bad_sha=("server-core",))
        )
    assert gpu_backend.installed_manifest() is None
    assert not (root / "current").exists()
    assert _no_leftovers(root)


def test_a_manifest_for_another_release_is_refused(release):
    root, build, _app = release
    with pytest.raises(RuntimeError, match="PodCodex 9.9.9"):
        gpu_backend.download_and_install(
            _cb, manifest_url=build(manifest_server="9.9.9")
        )
    assert not (root / "current").exists()


def test_a_server_core_of_another_version_leaves_the_install_alone(release):
    """Older manifests carry no server_version; the extracted binary decides,
    before it becomes an install."""
    root, build, app = release
    gpu_backend.download_and_install(_cb, manifest_url=build())
    before = _active(root)

    app["version"] = "1.2.4"
    with pytest.raises(RuntimeError, match="1.2.2"):
        gpu_backend.download_and_install(
            _cb, manifest_url=build(server_reports="1.2.2", manifest_server=None)
        )
    assert _active(root) == before
    assert gpu_backend._probe_server_core_version(before) == "1.2.3"


def test_an_unsafe_archive_name_is_refused(release, tmp_path):
    _root, build, _app = release
    url = build()
    path = Path(url.removeprefix("file://"))
    manifest = json.loads(path.read_text())
    manifest["parts"][1]["archive"] = "../evil.tar.gz"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Unsafe name"):
        gpu_backend.download_and_install(_cb, manifest_url=url)


# ── Legacy flat layout ──────────────────────────────────────────────────


def _legacy_install(root: Path, version="1.2.3") -> None:
    """The layout before install folders: everything flat in the root."""
    (root / "_internal/torch/lib").mkdir(parents=True)
    binary = root / "podcodex-server-gpu"
    binary.write_text(f"#!/bin/sh\necho podcodex {version}\n")
    binary.chmod(0o755)
    (root / "_internal/torch/lib/libcudnn.so").write_bytes(b"old-cuda")
    (root / "cuda-libs.json").write_text(
        json.dumps(
            {
                "version": "cu128-v1",
                "archive": "cuda-libs-cu128-v1.tar.gz",
                "sha256": "a" * 64,
                "server_sha256": "b" * 64,
            }
        )
    )
    (root / "activated").touch()


@pytest.mark.legacy("gpu-flat-layout")
def test_a_legacy_flat_install_is_still_read(release):
    root, _build, _app = release
    _legacy_install(root)

    assert gpu_backend.active_install_dir() == root
    assert gpu_backend.installed_server_core_version() == "1.2.3"
    assert gpu_backend.is_gpu_activated()


@pytest.mark.legacy("gpu-flat-layout")
def test_installing_over_a_legacy_install_moves_to_a_folder(release):
    root, build, app = release
    _legacy_install(root)
    app["version"] = "1.2.4"

    gpu_backend.download_and_install(_cb, manifest_url=build())

    active = _active(root)
    assert gpu_backend.installed_server_core_version() == "1.2.4"
    assert (active / "_internal/torch/lib/libcudnn.so").read_bytes() == b"cuda-1"
    assert gpu_backend.is_gpu_activated()
    # The flat files are pruned; only the layout entries and the folder stay.
    assert sorted(e.name for e in root.iterdir()) == sorted(
        ["activated", "current", active.name]
    )


@pytest.mark.legacy("gpu-flat-layout")
def test_an_interrupted_legacy_install_is_never_trusted(release):
    root, _build, _app = release
    _legacy_install(root)
    (root / ".installing").touch()  # a crash mid-extraction, old style
    assert gpu_backend.installed_manifest() is None
    assert gpu_backend.installed_server_core_version() is None
    with pytest.raises(RuntimeError, match="did not finish"):
        gpu_backend.activate()


def test_an_interrupted_download_is_retried(release, monkeypatch):
    """A dropped connection retries instead of failing the whole install."""
    import http.client

    _root, build, _app = release
    url = build()
    real = gpu_backend._download_once
    calls: list[str] = []

    def flaky(u, dest, *a, **k):
        calls.append(u)
        if calls.count(u) == 1 and "cuda-libs-" in u:
            dest.write_bytes(Path(u.removeprefix("file://")).read_bytes()[:5])
            raise http.client.IncompleteRead(b"", 10)
        return real(u, dest, *a, **k)

    monkeypatch.setattr(gpu_backend, "_download_once", flaky)
    monkeypatch.setattr(gpu_backend, "_DOWNLOAD_BACKOFF_CAP_S", 0.0)
    gpu_backend.download_and_install(_cb, manifest_url=url)
    assert sum("cuda-libs-" in u for u in calls) == 2
    assert gpu_backend.installed_server_core_version() == "1.2.3"


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
    root, build, _app = release
    gpu_backend.download_and_install(_cb, manifest_url=build())
    with gpu_backend._INSTALL_LOCK:
        with pytest.raises(RuntimeError, match="being downloaded"):
            gpu_backend.uninstall()
        with pytest.raises(RuntimeError, match="being downloaded"):
            gpu_backend.activate()
    assert (_active(root) / "podcodex-server-gpu").exists()


def test_a_damaged_part_is_downloaded_again_instead_of_failing(release):
    """A file quarantined by an antivirus must not fail every later update."""
    root, build, app = release
    gpu_backend.download_and_install(_cb, manifest_url=build())
    (_active(root) / "_internal/torch/lib/libcudnn.so").unlink()

    app["version"] = "1.2.4"
    result = gpu_backend.download_and_install(_cb, manifest_url=build())

    assert result["downloaded"] == ["server-core", "cuda-libs"]
    assert (_active(root) / "_internal/torch/lib/libcudnn.so").read_bytes() == b"cuda-1"


def test_a_drive_relative_archive_name_is_refused(release):
    """Windows joins "C:x" outside the download folder."""
    _root, build, _app = release
    url = build()
    path = Path(url.removeprefix("file://"))
    manifest = json.loads(path.read_text())
    manifest["parts"][2]["archive"] = "C:cuda.tar.gz"
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Unsafe name"):
        gpu_backend.download_and_install(_cb, manifest_url=url)


@pytest.mark.legacy("gpu-flat-layout")
def test_a_legacy_carry_over_drops_the_old_server_dist_info(release):
    """A two-archive release whose cuda tag matches carries the whole legacy
    tree. Its dist-info has the version in the directory name; the old one
    left beside the new makes --version report the older release."""
    root, build, app = release
    _legacy_install(root)
    (root / "_internal/podcodex-1.2.3.dist-info").mkdir()
    (root / "_internal/podcodex-1.2.3.dist-info/METADATA").write_text("old")
    app["version"] = "1.2.4"
    url = build()
    path = Path(url.removeprefix("file://"))
    parts = {p["name"]: p for p in json.loads(path.read_text())["parts"]}
    path.write_text(
        json.dumps(
            {
                "server_version": "1.2.4",
                "version": "cu128-v1",  # what the legacy install holds
                "archive": parts["cuda-libs"]["archive"],
                "sha256": parts["cuda-libs"]["sha256"],
                "server_sha256": parts["server-core"]["sha256"],
            }
        )
    )

    result = gpu_backend.download_and_install(_cb, manifest_url=url)

    active = _active(root)
    assert result["downloaded"] == ["server-core"]
    assert gpu_backend._probe_server_core_version(active) == "1.2.4"
    assert not (active / "_internal/podcodex-1.2.3.dist-info").exists()
    # The reused legacy files came along.
    assert (active / "_internal/torch/lib/libcudnn.so").read_bytes() == b"old-cuda"


# ── The sidecar archive is verified, never best-effort ──────────────────


def test_gpu_install_refuses_manifest_without_server_hash(tmp_path, monkeypatch):
    """server-core.tar.gz becomes the executed sidecar, so its hash is required.

    It comes from the manifest, not an optional ``<archive>.sha256`` sidecar
    fetch, where a 404 or a network blip would downgrade the integrity check
    on the one archive that carries code to a log warning.
    """
    import json

    from podcodex.api import gpu_backend

    monkeypatch.setattr(gpu_backend, "_ensure_bundle_mode", lambda: None)
    monkeypatch.setattr(gpu_backend, "_ensure_platform_supported", lambda: None)
    monkeypatch.setattr(gpu_backend, "gpu_install_dir", lambda: tmp_path / "gpu")
    monkeypatch.setattr(
        gpu_backend,
        "_fetch_text",
        lambda url, **kw: json.dumps(
            {
                "version": "cu128-v1",
                "archive": "cuda-libs-cu128-v1.tar.gz",
                "sha256": "a" * 64,
            }
        ),
    )

    with pytest.raises(RuntimeError, match="server_sha256"):
        gpu_backend.download_and_install(lambda *a: None, "https://x/cuda-libs.json")


def test_gpu_packager_publishes_every_part_hash(tmp_path):
    """The packager must emit what the installer requires: every part with its
    sha256 (the server core's included) and content tags that only move when
    the part's files do."""
    import importlib.util
    import tarfile
    from pathlib import Path

    from podcodex.api.gpu_backend import _manifest_parts

    spec = importlib.util.spec_from_file_location(
        "package_gpu", Path("packaging/package_gpu.py")
    )
    package_gpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package_gpu)

    build = tmp_path / "onedir"
    for rel, data in {
        "podcodex-server-gpu.exe": b"exe",
        "_internal/podcodex/app.py": b"code",
        "_internal/torch/lib/torch_cuda.dll": b"torch",
        "_internal/torch/lib/cudnn64_9.dll": b"cudnn",
    }.items():
        (build / rel).parent.mkdir(parents=True, exist_ok=True)
        (build / rel).write_bytes(data)

    first = package_gpu.package(build, tmp_path / "out1", ">=2.7")
    parts = {p.name: p for p in _manifest_parts(first)}
    assert set(parts) == {"server-core", "torch-runtime", "cuda-libs"}
    assert first["server_sha256"] == parts["server-core"].sha256
    with tarfile.open(tmp_path / "out1" / parts["torch-runtime"].archive) as tar:
        assert tar.getnames() == ["_internal/torch/lib/torch_cuda.dll"]

    (build / "_internal/podcodex/app.py").write_bytes(b"new code")
    second = {
        p.name: p
        for p in _manifest_parts(package_gpu.package(build, tmp_path / "out2", ">=2.7"))
    }
    assert second["cuda-libs"].tag == parts["cuda-libs"].tag
    assert second["torch-runtime"].tag == parts["torch-runtime"].tag
