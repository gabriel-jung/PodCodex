"""Split a GPU PyInstaller --onedir build into two release archives.

Takes ``packaging/dist/podcodex-server-gpu/`` (produced by
``build_server.py --gpu``) and splits it into:

  1. ``server-core.tar.gz``     — server code + non-NVIDIA deps (~300 MB).
                                   Versioned with the app, redownloaded on
                                   every release.
  2. ``cuda-libs-<ver>.tar.gz`` — NVIDIA runtime libraries (~2 GB).
                                   Versioned independently; only redownloaded
                                   when the CUDA toolkit / torch major
                                   version changes.
  3. ``cuda-libs.json``         — manifest with version + sha256 + torch
                                   compatibility range, consumed by the
                                   runtime downloader (Phase M.4).

Usage:
    .venv/bin/python packaging/package_gpu.py
    .venv/bin/python packaging/package_gpu.py --output release-assets/
    .venv/bin/python packaging/package_gpu.py --cuda-libs-version cu128-v1

The split is by file: anything matching an NVIDIA library prefix goes
into the cuda-libs archive; everything else into server-core. PyInstaller
places NVIDIA libs in different locations depending on the torch version
(``nvidia/`` subdirectories on older torch, ``_internal/torch/lib/`` on
torch 2.10+, sometimes top-level), so we classify by name rather than path.
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_archive(archive_path: Path, files: list[tuple[str, Path]]) -> None:
    """Write a gzipped tar of (arcname, source_path) pairs. Files inside the
    archive are stored relative to the archive root so extraction to
    `<app_data>/backends/gpu/` puts them at the right level."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for arcname, src in files:
            tar.add(src, arcname=arcname)


def package(
    onedir_path: Path,
    output_dir: Path,
    cuda_libs_version: str,
    torch_compat: str,
) -> None:
    if not onedir_path.is_dir():
        print(f"Error: input is not a directory: {onedir_path}", file=sys.stderr)
        print("Expected the PyInstaller --onedir output from build_server.py --gpu.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    core_files: list[tuple[str, Path]] = []
    nvidia_files: list[tuple[str, Path]] = []
    for item in sorted(onedir_path.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(onedir_path)
        rel_str = str(rel)
        bucket = nvidia_files if is_nvidia_file(rel_str) else core_files
        bucket.append((rel_str, item))

    if not nvidia_files:
        print(
            f"Error: no NVIDIA files found in {onedir_path}.\n"
            "Refusing to write an empty cuda-libs archive. Did you build with --gpu?",
            file=sys.stderr,
        )
        sys.exit(1)

    core_size = sum(src.stat().st_size for _, src in core_files)
    nvidia_size = sum(src.stat().st_size for _, src in nvidia_files)

    print(f"Input:        {onedir_path}")
    print(f"Output:       {output_dir}")
    print(f"Core files:   {len(core_files):>5} ({core_size / 1024**2:>7.1f} MB)")
    print(f"NVIDIA files: {len(nvidia_files):>5} ({nvidia_size / 1024**2:>7.1f} MB)")

    # Server core
    core_archive = output_dir / "server-core.tar.gz"
    print(f"\nWriting {core_archive.name} ...")
    write_archive(core_archive, core_files)
    core_sha = sha256_file(core_archive)
    (output_dir / f"{core_archive.name}.sha256").write_text(f"{core_sha}  {core_archive.name}\n")
    print(f"  size:    {core_archive.stat().st_size / 1024**2:.1f} MB")
    print(f"  sha256:  {core_sha[:16]}...")

    # CUDA libs
    cuda_archive = output_dir / f"cuda-libs-{cuda_libs_version}.tar.gz"
    print(f"\nWriting {cuda_archive.name} ...")
    write_archive(cuda_archive, nvidia_files)
    cuda_sha = sha256_file(cuda_archive)
    (output_dir / f"{cuda_archive.name}.sha256").write_text(f"{cuda_sha}  {cuda_archive.name}\n")
    print(f"  size:    {cuda_archive.stat().st_size / 1024**2:.1f} MB")
    print(f"  sha256:  {cuda_sha[:16]}...")

    # Manifest — the runtime downloader fetches this first to know which
    # archive to pull, verify, and extract.
    manifest = {
        "version": cuda_libs_version,
        "torch_compat": torch_compat,
        "archive": cuda_archive.name,
        "sha256": cuda_sha,
    }
    manifest_path = output_dir / "cuda-libs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))

    total_in = core_size + nvidia_size
    total_out = core_archive.stat().st_size + cuda_archive.stat().st_size
    print(f"\nUncompressed input: {total_in / 1024**3:.2f} GB")
    print(f"Compressed total:   {total_out / 1024**3:.2f} GB")


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
        "--cuda-libs-version",
        default="cu128-v1",
        help="Version tag for the cuda-libs archive (default: cu128-v1). Bump only when the CUDA toolkit changes.",
    )
    parser.add_argument(
        "--torch-compat",
        default=">=2.7.0,<2.11.0",
        help="Torch version range this cuda-libs archive supports (default: >=2.7.0,<2.11.0).",
    )
    args = parser.parse_args()

    package(args.input, args.output, args.cuda_libs_version, args.torch_compat)


if __name__ == "__main__":
    main()
