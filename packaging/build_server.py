"""Build the bundled podcodex-server binary.

Drives PyInstaller directly via CLI args (no .spec file). All build config
lives here as module-level constants so the whole build is readable
top-to-bottom.

Usage:
    .venv/bin/python packaging/build_server.py            # CPU build (default)
    .venv/bin/python packaging/build_server.py --gpu      # GPU build (CUDA)
    .venv/bin/python packaging/build_server.py --clean    # wipe dist/ and build_work/ first

CPU vs GPU:
    CPU: --onefile, excludes nvidia.* (~500 MB sidecar). Output:
         dist/podcodex-server → src-tauri/binaries/podcodex-server-<triple>.
    GPU: --onedir, no nvidia excludes (~3 GB tree). Output:
         dist/podcodex-server-gpu/ ready for packaging/package_gpu.py to
         split into server-core + cuda-libs archives (Phase M.3).

Torch swap:
    On Linux/Windows hosts where the venv has GPU torch installed, the CPU
    build temporarily replaces torch + torchvision + torchaudio with the
    +cpu wheel before PyInstaller runs (~1.5 GB savings) and restores GPU
    torch in a finally block. All three packages move together — they
    share a C++ ABI and version drift breaks operator registration.
    macOS torch wheels are always CPU/MPS — no swap needed.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGING_DIR = REPO_ROOT / "packaging"
PYI_HOOKS_DIR = PACKAGING_DIR / "pyi_hooks"
SRC_DIR = REPO_ROOT / "src"
ENTRY_SCRIPT = SRC_DIR / "podcodex" / "api" / "server.py"
DIST_DIR = PACKAGING_DIR / "dist"
WORK_DIR = PACKAGING_DIR / "build_work"
SIDECAR_OUT_DIR = REPO_ROOT / "src-tauri" / "binaries"

# ── Build content ───────────────────────────────────────────────────────

HIDDEN_IMPORTS = [
    # Backend application
    "podcodex",
    "podcodex.api.app",
    "podcodex.api.routes",
    "podcodex.core",
    "podcodex.pipeline",
    "podcodex.rag",
    "podcodex.mcp",
    # Job entry points imported dynamically by subprocess_runner via
    # importlib.import_module(). PyInstaller cannot statically detect these
    # so they need an explicit hidden-import.
    "podcodex.core.transcribe_job",
    "podcodex.core.synthesize_job",
    "podcodex.rag.index_job",
    # ML stack
    "torch",
    "torchaudio",
    "transformers",
    "accelerate",
    "ctranslate2",
    "faster_whisper",
    "whisperx",
    "pyannote.audio",
    "lightning_fabric",
    "pytorch_lightning",
    "sentence_transformers",
    "FlagEmbedding",
    # Retrieval
    "lancedb",
    "pyarrow",
    # Audio
    "soundfile",
    "librosa",
    "numba",
    "llvmlite",
    # API + integrations
    "fastapi",
    "uvicorn",
    "uvicorn.lifespan.on",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "starlette",
    "yt_dlp",
    "mcp",
    "ollama",
    "openai",
    "qwen_tts",
    "feedparser",
    "loguru",
]

COPY_METADATA = [
    "torch",
    "transformers",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
    "tqdm",
    "requests",
    "numpy",
    "fastapi",
    "uvicorn",
    "yt-dlp",
    "lancedb",
    "torchcodec",
]

# Native .so / pickled data / lazy_loader stubs that --hidden-import alone misses.
COLLECT_ALL = [
    "ctranslate2",
    "faster_whisper",
    "whisperx",
    "pyannote",
    "lancedb",
    "lazy_loader",
    "librosa",
    "numba",
    "llvmlite",
    "soundfile",
    "yt_dlp",
    # jsonschema's rfc3987_syntax dep ships a .lark grammar file consumed at
    # import time; collect_all picks up the data file alongside the module.
    "rfc3987_syntax",
    "jsonschema",
    "jsonschema_specifications",
    "mcp",
    # transformers.audio_utils calls importlib.metadata.version("torchcodec")
    # at module-import time when torchcodec is installed. Without the .dist-info
    # in the bundle, that lookup raises PackageNotFoundError and every
    # `from transformers import Pipeline` fails. collect_all also pulls in the
    # native .so so the runtime call paths actually work.
    "torchcodec",
]

COLLECT_SUBMODULES = [
    "transformers",
    "torch",
    "pyannote",
]

# Order matters: pyi_rth_aaa_anaconda_sysver runs first because it patches
# sys.version before pyi_rth_nltk's auto-injected hook hits cloudpickle's
# strict version parser.
RUNTIME_HOOKS = [
    PYI_HOOKS_DIR / "pyi_rth_aaa_anaconda_sysver.py",
    PYI_HOOKS_DIR / "pyi_rth_numpy_compat.py",
    PYI_HOOKS_DIR / "pyi_rth_torch_compiler_disable.py",
]

# Always excluded — pulled in by transitive deps but never used at runtime.
DEV_ONLY_EXCLUDES = [
    "tkinter",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
]

# CUDA libs — excluded from CPU build only. Saves ~1.5 GB.
NVIDIA_EXCLUDES = [
    "nvidia",
    "nvidia.cublas",
    "nvidia.cuda_cupti",
    "nvidia.cuda_nvrtc",
    "nvidia.cuda_runtime",
    "nvidia.cudnn",
    "nvidia.cufft",
    "nvidia.curand",
    "nvidia.cusolver",
    "nvidia.cusparse",
    "nvidia.nccl",
    "nvidia.nvjitlink",
    "nvidia.nvtx",
]

# GPU build needs these explicitly; CPU build ignores them.
GPU_HIDDEN_IMPORTS = [
    "torch.cuda",
    "torch.backends.cudnn",
]

# ── Helpers ─────────────────────────────────────────────────────────────


def host_target_triple() -> str:
    out = subprocess.check_output(["rustc", "--print", "host-tuple"], text=True).strip()
    if not out:
        raise RuntimeError("rustc --print host-tuple returned empty")
    return out


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def find_openssl_libs_for_bundle() -> list[tuple[Path, str]]:
    """Linux conda/miniforge fix: bundle libssl/libcrypto from the active
    Python's prefix so they match what _ssl.so was linked against.

    On these hosts PyInstaller's binary auto-detector picks the system
    /lib/x86_64-linux-gnu/libcrypto.so.3 (older), and the sidecar then dies
    with `version OPENSSL_3.x.0 not found`. Prepending these via --add-binary
    makes PyInstaller's first-wins dedup keep them.

    Returns [] on macOS/Windows and on Linux hosts where the libs aren't
    in the active Python's prefix (e.g., system Python on Ubuntu).
    """
    if is_macos() or is_windows():
        return []
    pyprefix_lib = Path(sys.base_prefix) / "lib"
    out: list[tuple[Path, str]] = []
    for name in ("libssl.so.3", "libcrypto.so.3"):
        src = pyprefix_lib / name
        if src.is_file():
            out.append((src, "."))
    return out


# ── Torch swap (CPU build only on Linux/Windows) ────────────────────────


def _venv_torch_cuda_version() -> str:
    """Return torch.version.cuda from a subprocess (avoids importing torch
    in the build process). Empty string if torch is CPU-only or missing."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return result.stdout.strip()


def _pip_install_torch(index_url: str) -> None:
    """Reinstall torch + torchvision + torchaudio from the given index.

    All three must move in lockstep — torchvision and torchaudio compile
    against torch's C++ ABI, and even minor torch version drift breaks
    operator registration (e.g. ``torchvision::nms does not exist`` at
    import time). Uses `uv pip` because our venv is uv-managed and doesn't
    ship pip; `--reinstall` is uv's equivalent of pip's `--force-reinstall`.
    """
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            index_url,
            "--reinstall",
            "--python",
            sys.executable,
        ],
        check=True,
    )


def maybe_swap_torch_to_cpu() -> str | None:
    """If the venv has GPU torch on Linux/Windows, swap to torch+cpu before
    PyInstaller runs. Returns the original CUDA version (e.g. '12.8') so
    the caller can restore, or None if no swap was needed.
    """
    if is_macos():
        return None
    cuda = _venv_torch_cuda_version()
    if not cuda:
        return None
    print(f"GPU torch detected (cu{cuda.replace('.', '')}); swapping to torch+cpu...")
    _pip_install_torch("https://download.pytorch.org/whl/cpu")
    return cuda


def restore_torch_gpu(cuda_version: str) -> None:
    """Reinstall the GPU torch wheel matching the original cuda version
    (e.g. '12.8' → cu128 index)."""
    tag = "cu" + cuda_version.replace(".", "")
    print(f"Restoring GPU torch ({tag})...")
    _pip_install_torch(f"https://download.pytorch.org/whl/{tag}")


# ── PyInstaller invocation ──────────────────────────────────────────────


def build_pyinstaller_args(*, gpu: bool, clean: bool) -> list[str]:
    """Assemble the full PyInstaller command line for either build."""
    name = "podcodex-server-gpu" if gpu else "podcodex-server"
    args: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(ENTRY_SCRIPT),
        "--name",
        name,
        "--onedir" if gpu else "--onefile",
        "--paths",
        str(SRC_DIR),
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(WORK_DIR),
        "--noconfirm",
        "--additional-hooks-dir",
        str(PYI_HOOKS_DIR),
    ]
    if clean:
        args.append("--clean")
    if is_windows():
        # Build the sidecar with GUI subsystem on Windows so spawning it from
        # the Tauri shell doesn't pop a cmd.exe window. stdout/stderr pipes
        # still work — Tauri captures them from Rust's piped stdio handles
        # regardless of subsystem. macOS/Linux spawn semantics don't show a
        # console for child processes by default, so we keep console there
        # for terminal debugging.
        args.append("--noconsole")

    for mod in HIDDEN_IMPORTS:
        args.extend(["--hidden-import", mod])
    if gpu:
        for mod in GPU_HIDDEN_IMPORTS:
            args.extend(["--hidden-import", mod])

    for pkg in COPY_METADATA:
        args.extend(["--copy-metadata", pkg])

    for pkg in COLLECT_ALL:
        args.extend(["--collect-all", pkg])

    for pkg in COLLECT_SUBMODULES:
        args.extend(["--collect-submodules", pkg])

    for hook in RUNTIME_HOOKS:
        args.extend(["--runtime-hook", str(hook)])

    excludes = list(DEV_ONLY_EXCLUDES)
    if not gpu:
        excludes += NVIDIA_EXCLUDES
    for mod in excludes:
        args.extend(["--exclude-module", mod])

    for src, dest in find_openssl_libs_for_bundle():
        args.extend(["--add-binary", f"{src}{os.pathsep}{dest}"])

    return args


def run_pyinstaller(*, gpu: bool, clean: bool) -> Path:
    args = build_pyinstaller_args(gpu=gpu, clean=clean)
    print(f"$ {' '.join(args)}")
    subprocess.check_call(args, cwd=PACKAGING_DIR)

    if gpu:
        # --onedir: the directory itself is the artifact. The exe inside is
        # named podcodex-server-gpu(.exe); package_gpu.py walks the dir.
        out = DIST_DIR / "podcodex-server-gpu"
        if not out.is_dir():
            raise RuntimeError(f"PyInstaller --onedir output not found: {out}")
        return out

    # CPU --onefile: single binary at dist/podcodex-server[.exe]
    for c in (DIST_DIR / "podcodex-server", DIST_DIR / "podcodex-server.exe"):
        if c.is_file():
            return c
    raise RuntimeError(f"PyInstaller --onefile output not found in {DIST_DIR}")


def copy_to_sidecar(source: Path, triple: str) -> None:
    """Place the CPU binary at src-tauri/binaries/podcodex-server-<triple>.

    Tauri's externalBin convention requires the -<triple> suffix in the
    project tree; the .app/.msi bundler strips it on copy. GPU builds skip
    this — they're packaged separately by package_gpu.py.
    """
    SIDECAR_OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if "windows" in triple else ""
    dest = SIDECAR_OUT_DIR / f"podcodex-server-{triple}{suffix}"
    if dest.exists():
        dest.unlink()
    shutil.copy2(source, dest)
    dest.chmod(0o755)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"Sidecar binary written: {dest}  ({size_mb:.1f} MB)")


# ── Entry ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="build GPU sidecar (CUDA, --onedir)")
    parser.add_argument("--clean", action="store_true", help="wipe dist/ and build_work/ first")
    parser.add_argument(
        "--no-torch-swap",
        action="store_true",
        help="skip the auto torch+cpu swap on CPU builds (use whatever's in the venv)",
    )
    args = parser.parse_args()

    if args.clean:
        for d in (DIST_DIR, WORK_DIR):
            if d.exists():
                shutil.rmtree(d)

    triple = host_target_triple()
    mode = "GPU (--onedir)" if args.gpu else "CPU (--onefile)"
    print(f"Target triple: {triple}")
    print(f"Build mode:    {mode}")

    restore_cuda: str | None = None
    if not args.gpu and not args.no_torch_swap:
        restore_cuda = maybe_swap_torch_to_cpu()

    try:
        output = run_pyinstaller(gpu=args.gpu, clean=args.clean)
    finally:
        if restore_cuda is not None:
            restore_torch_gpu(restore_cuda)

    if args.gpu:
        print(f"GPU --onedir built at: {output}")
        print("Next: run packaging/package_gpu.py to split into server-core + cuda-libs archives.")
    else:
        copy_to_sidecar(output, triple)


if __name__ == "__main__":
    main()
