"""Split a GPU PyInstaller --onedir build into three release archives.

Takes ``packaging/dist/podcodex-server-gpu/`` (produced by
``build_server.py --gpu``) and splits it into parts that change at
different rates, so an app update downloads only what changed:

  1. ``server-core.tar.xz``            : PodCodex and its pure-Python
                                         dependencies. Changes every release.
  2. ``torch-runtime-<tag>.tar.xz``    : torch's own binaries
                                         (``torch/lib``: torch_cuda, torch_cpu,
                                         ...). Changes with the torch version.
  3. ``cuda-libs-<tag>.tar.xz``        : NVIDIA runtime libraries (cuDNN,
                                         cuBLAS, ...). Changes with the CUDA
                                         toolkit.
  4. ``cuda-libs.json``                : manifest listing the parts with their
                                         sha256 and tag, consumed by the
                                         runtime installer
                                         (``src/podcodex/api/gpu_backend.py``).

Each tag is a hash of the part's file contents, never a hand-set label: a
torch bump that brings a new cuDNN changes the cuda-libs tag by itself, and
a release that changes nothing in a part keeps its tag, so installs reuse it.

Usage:
    .venv/bin/python packaging/package_gpu.py
    .venv/bin/python packaging/package_gpu.py --output release-assets/

NVIDIA libs are classified by name rather than path: PyInstaller places
them differently depending on the torch version (``nvidia/`` subdirectories
on older torch, ``_internal/torch/lib/`` on torch 2.10+, sometimes top-level).
Everything else under ``torch/lib/`` is the torch runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "packaging" / "dist" / "podcodex-server-gpu"
DEFAULT_OUTPUT = REPO_ROOT / "packaging" / "release-assets"

# DLL/.so name prefixes that identify NVIDIA CUDA runtime libraries.
# Same shapes on Linux (.so) and Windows (.dll) — torch wheels use the same
# names with platform-appropriate extensions.
NVIDIA_LIB_PREFIXES = (
    "cublas",
    "cublaslt",
    "cudart",
    "cudnn",
    "cufft",
    "cufftw",
    "curand",
    "cusolver",
    "cusolvermg",
    "cusparse",
    "nvjitlink",
    "nvrtc",
    "nccl",
    "caffe2_nvrtc",
)

# NVIDIA libraries the Windows wheel ships that nothing in PodCodex loads,
# left out of the download (about 300 MB unpacked). On the 0.3.0 build no
# bundled binary imports them or names them for a runtime load, and torch on
# Windows loads whichever DLLs torch/lib holds, so a missing one is skipped.
# referenced_left_out re-checks the first half on every build. Windows
# only: on Linux torch links libcurand directly and preloads the rest.
#   cusolverMg: the multi-GPU cuSOLVER; torch links the plain one.
#   curand: the host library; torch's generators are its own, and the
#           curand device code is compiled into the kernels that use it.
#   nvrtc*.alt: a second NVRTC build; caffe2_nvrtc loads the regular one.
#   cufftw: the FFTW-compatible wrapper over cuFFT.
UNUSED_NVIDIA_PREFIXES = ("cusolvermg", "curand", "cufftw")

# Release archives: xz packs the CUDA binaries tighter than gzip, and the
# stdlib reads and writes it. The installer detects the compression.
ARCHIVE_SUFFIX = ".tar.xz"

# Files to keep in the server core even if their paths or names match
# NVIDIA prefixes — these are small Python stubs, not the heavy runtime libs.
NVIDIA_KEEP_IN_CORE = {
    "torch/cuda/nccl.py",
    "torch/_inductor/codegen/cuda/cutlass_lib_extensions/cutlass_mock_imports/cuda/cudart.py",
}


def is_nvidia_file(rel_path: str) -> bool:
    """Return True for files belonging to the NVIDIA CUDA runtime split."""
    rel_lower = rel_path.lower().replace("\\", "/")

    if rel_lower in NVIDIA_KEEP_IN_CORE:
        return False

    # Files anywhere under an `nvidia/` directory tree (older torch layout).
    if rel_lower.startswith("nvidia/") or "/nvidia/" in rel_lower:
        if rel_lower.endswith((".dll", ".so")):
            return True
        for part in rel_lower.split("/"):
            if part == "nvidia":
                return True

    # NVIDIA shared libs anywhere in the tree. Handles three filename shapes:
    #   Windows:        cublas64_12.dll, cudnn64_9.dll
    #   Linux unversioned: libcublas.so, libcudnn.so
    #   Linux versioned:   libcudnn_graph.so.9.10.2  (suffix is the version,
    #                      not just .so — `endswith('.so')` misses these).
    name = rel_lower.rsplit("/", 1)[-1]
    is_shared_lib = name.endswith(".dll") or ".so" in name
    if not is_shared_lib:
        return False
    # Strip Linux `lib` prefix so the prefix list doesn't need `lib`-variants.
    bare = name[3:] if name.startswith("lib") else name
    # Get the library name without extension/version suffix.
    if ".so" in bare:
        bare_no_ext = bare.split(".so", 1)[0]
    elif bare.endswith(".dll"):
        bare_no_ext = bare[:-4]
    else:
        bare_no_ext = bare
    for prefix in NVIDIA_LIB_PREFIXES:
        if bare_no_ext.startswith(prefix):
            return True

    return False


def is_unused_nvidia_file(rel_path: str) -> bool:
    """True for a Windows NVIDIA DLL nothing loads (see UNUSED_NVIDIA_PREFIXES)."""
    name = rel_path.lower().replace("\\", "/").rsplit("/", 1)[-1]
    if not name.endswith(".dll"):
        return False
    if name.startswith("nvrtc") and name.endswith(".alt.dll"):
        return True
    return is_nvidia_file(rel_path) and name.startswith(UNUSED_NVIDIA_PREFIXES)


def referenced_left_out(
    left_out: list[tuple[str, Path]], kept: list[tuple[str, Path]]
) -> set[str]:
    """Names of the left-out libraries a kept binary refers to.

    A DLL's import table stores the names of the DLLs it links as plain
    strings, and a runtime LoadLibrary takes one as ASCII or UTF-16, so a
    byte search over every bundled .dll/.pyd catches both. Runs on every
    build: a torch or CTranslate2 update that starts loading one of them
    would otherwise ship a GPU backend that fails at ``import torch``.
    """
    names = {src.name.lower().removesuffix(".dll") for _, src in left_out}
    found: set[str] = set()
    for _rel, src in kept:
        if not src.name.lower().endswith((".dll", ".pyd")):
            continue
        data = src.read_bytes().lower()
        found |= {
            name
            for name in names - found
            if name.encode() in data or name.encode("utf-16-le") in data
        }
    return found


def is_torch_runtime_file(rel_path: str) -> bool:
    """True for torch's own binaries (not NVIDIA's), under ``torch/lib/``."""
    rel_lower = "/" + rel_path.lower().replace("\\", "/")
    return "/torch/lib/" in rel_lower and not is_nvidia_file(rel_path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def content_tag(files: list[tuple[str, Path]]) -> str:
    """Hash of a part's relative paths and file contents (12 hex chars).

    From the files, not the archive: compressed tar output is not
    reproducible (member mtimes, compressor versions), and an unchanged part
    must keep its tag so installs reuse it.
    """
    h = hashlib.sha256()
    for rel, src in sorted(files, key=lambda f: f[0].replace("\\", "/")):
        h.update(rel.replace("\\", "/").encode("utf-8") + b"\0")
        h.update(sha256_file(src).encode("ascii") + b"\n")
    return h.hexdigest()[:12]


def write_archive(archive_path: Path, files: list[tuple[str, Path]]) -> None:
    """Write *files* into an xz tar, stored relative to the archive root so
    extraction into the install dir reunites the onedir tree."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:xz") as tar:
        for rel, src in files:
            tar.add(src, arcname=rel)


def _print_files(files: list[tuple[str, Path]]) -> None:
    sized = sorted(((src.stat().st_size, rel) for rel, src in files), reverse=True)
    for size, rel in sized:
        print(f"  {size / 1024**2:8.1f} MB  {rel}")


def _build_id() -> str:
    """``build_server.build_id``: one definition, since the app and its
    launcher compare the id bundled there with the one stamped here."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_server", Path(__file__).with_name("build_server.py")
    )
    build_server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_server)
    return build_server.build_id()


def _app_version() -> str:
    """Version from pyproject.toml, the source every other version derives from."""
    import tomllib

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]


def package(onedir_path: Path, output_dir: Path, torch_compat: str) -> dict:
    if not onedir_path.is_dir():
        print(f"Error: input is not a directory: {onedir_path}", file=sys.stderr)
        print(
            "Expected the PyInstaller --onedir output from build_server.py --gpu.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, list[tuple[str, Path]]] = {
        "server-core": [],
        "torch-runtime": [],
        "cuda-libs": [],
    }
    left_out: list[tuple[str, Path]] = []
    for item in sorted(onedir_path.rglob("*")):
        if item.is_dir():
            continue
        rel = str(item.relative_to(onedir_path))
        if is_unused_nvidia_file(rel):
            left_out.append((rel, item))
        elif is_nvidia_file(rel):
            buckets["cuda-libs"].append((rel, item))
        elif is_torch_runtime_file(rel):
            buckets["torch-runtime"].append((rel, item))
        else:
            buckets["server-core"].append((rel, item))

    for part in ("cuda-libs", "torch-runtime"):
        if not buckets[part]:
            print(
                f"Error: no {part} files found in {onedir_path}.\n"
                "Refusing to write an empty archive. Did you build with --gpu?",
                file=sys.stderr,
            )
            sys.exit(1)

    app_version = _app_version()
    print(f"Input:        {onedir_path}")
    print(f"Output:       {output_dir}")
    # A left-out library some kept binary refers to goes back in: a bigger
    # download beats a backend that fails at import, or a blocked release.
    referenced = referenced_left_out(
        left_out, [f for files in buckets.values() for f in files]
    )
    for rel, src in left_out:
        if src.name.lower().removesuffix(".dll") in referenced:
            print(
                f"Warning: {rel} is referenced by a bundled binary, kept. "
                "Its UNUSED_NVIDIA_PREFIXES entry is out of date."
            )
            buckets["cuda-libs"].append((rel, src))
    left_out = [f for f in left_out if f not in buckets["cuda-libs"]]
    if left_out:
        print("\nLeft out, unused (see UNUSED_NVIDIA_PREFIXES):")
        _print_files(left_out)

    archives: dict[str, tuple[str, Path]] = {}
    for name, files in buckets.items():
        # The server core's tag is the app version: the launcher and the
        # installer both key it on ``--version``, not on content.
        tag = app_version if name == "server-core" else content_tag(files)
        archive = output_dir / (
            f"server-core{ARCHIVE_SUFFIX}"
            if name == "server-core"
            else f"{name}-{tag}{ARCHIVE_SUFFIX}"
        )
        size = sum(src.stat().st_size for _, src in files)
        print(f"\n{archive.name} ({len(files)} files, {size / 1024**2:.1f} MB)")
        if name != "server-core":
            # A few dozen libraries, listed so the next trim starts from data.
            _print_files(files)
        archives[name] = (tag, archive)

    # xz in the stdlib is single-threaded (about 19 minutes for the three
    # parts in a row), so each part compresses in its own thread, which runs
    # in parallel since the lzma compressor releases the GIL: the largest
    # part sets the time.
    print("\nCompressing ...")
    try:
        with ThreadPoolExecutor(max_workers=len(archives)) as pool:
            for future in [
                pool.submit(write_archive, archive, buckets[name])
                for name, (_tag, archive) in archives.items()
            ]:
                future.result()
    except BaseException:
        # The pool waited for the other parts; none of them may be uploaded
        # beside a missing or truncated one.
        for _tag, archive in archives.values():
            archive.unlink(missing_ok=True)
        raise

    by_name: dict[str, dict] = {}
    for name, (tag, archive) in archives.items():
        sha = sha256_file(archive)
        (output_dir / f"{archive.name}.sha256").write_text(f"{sha}  {archive.name}\n")
        print(f"  {archive.name}: {archive.stat().st_size / 1024**2:.1f} MB, tag {tag}")
        by_name[name] = {
            "name": name,
            "tag": tag,
            "archive": archive.name,
            "sha256": sha,
        }

    # ``server_version``: the app release the server core was built from. The
    # launcher only runs a GPU sidecar of its own version, so the installer
    # refuses a manifest stamped for another release. Every archive's sha256
    # travels here, the server core's included, since it becomes the executed
    # sidecar. The flat fields mirror the two-archive schema for installers
    # that read it (0.3.0 betas before parts); 0.2.x installers can't install
    # this release anyway (they fetch server-core.tar.gz and read gzip only,
    # and their launcher refuses another version's server core).
    manifest = {
        "schema": 2,
        "server_version": app_version,
        # Which build of that version: prereleases share server_version.
        "build_id": _build_id(),
        "torch_compat": torch_compat,
        "parts": list(by_name.values()),
        "version": by_name["cuda-libs"]["tag"],
        "archive": by_name["cuda-libs"]["archive"],
        "sha256": by_name["cuda-libs"]["sha256"],
        "server_sha256": by_name["server-core"]["sha256"],
    }
    manifest_path = output_dir / "cuda-libs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"PyInstaller --onedir output (default: {DEFAULT_INPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write archives + manifest (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--torch-compat",
        default=">=2.7.0,<2.11.0",
        help="Torch version range this cuda-libs archive supports (default: >=2.7.0,<2.11.0).",
    )
    args = parser.parse_args()

    package(args.input, args.output, args.torch_compat)


if __name__ == "__main__":
    main()
