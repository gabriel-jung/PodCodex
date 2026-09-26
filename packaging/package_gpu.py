"""Split a GPU PyInstaller --onedir build into three release archives.

Takes ``packaging/dist/podcodex-server-gpu/`` (produced by
``build_server.py --gpu``) and splits it into parts that change at
different rates, so an app update downloads only what changed:

  1. ``server-core.tar.gz``            : PodCodex and its pure-Python
                                         dependencies. Changes every release.
  2. ``torch-runtime-<tag>.tar.gz``    : torch's own binaries
                                         (``torch/lib``: torch_cuda, torch_cpu,
                                         ...). Changes with the torch version.
  3. ``cuda-libs-<tag>.tar.gz``        : NVIDIA runtime libraries (cuDNN,
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

    From the files, not the archive: gzip output is not reproducible, and an
    unchanged part must keep its tag so installs reuse it.
    """
    h = hashlib.sha256()
    for rel, src in sorted(files, key=lambda f: f[0].replace("\\", "/")):
        h.update(rel.replace("\\", "/").encode("utf-8") + b"\0")
        h.update(sha256_file(src).encode("ascii") + b"\n")
    return h.hexdigest()[:12]


def write_archive(archive_path: Path, files: list[tuple[str, Path]]) -> None:
    """Write *files* into a gzipped tar, stored relative to the archive root
    so extraction into the install dir reunites the onedir tree."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for rel, src in files:
            tar.add(src, arcname=rel)


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
    for item in sorted(onedir_path.rglob("*")):
        if item.is_dir():
            continue
        rel = str(item.relative_to(onedir_path))
        if is_nvidia_file(rel):
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

    by_name: dict[str, dict] = {}
    for name, files in buckets.items():
        # The server core's tag is the app version: the launcher and the
        # installer both key it on ``--version``, not on content.
        tag = app_version if name == "server-core" else content_tag(files)
        archive = output_dir / (
            "server-core.tar.gz" if name == "server-core" else f"{name}-{tag}.tar.gz"
        )
        size = sum(src.stat().st_size for _, src in files)
        print(
            f"\nWriting {archive.name} ({len(files)} files, {size / 1024**2:.1f} MB) ..."
        )
        write_archive(archive, files)
        sha = sha256_file(archive)
        (output_dir / f"{archive.name}.sha256").write_text(f"{sha}  {archive.name}\n")
        print(f"  size:    {archive.stat().st_size / 1024**2:.1f} MB")
        print(f"  tag:     {tag}")
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
    # sidecar. The flat fields mirror the two-archive schema, so an older
    # installer that falls back to this manifest reaches its "made for another
    # release" refusal instead of a missing-field error.
    manifest = {
        "schema": 2,
        "server_version": app_version,
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
