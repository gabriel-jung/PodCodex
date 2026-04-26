# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the bundled podcodex-server sidecar.

Build entry: src/podcodex/api/server.py (sets ML cache env then runs uvicorn).
Output: dist/podcodex-server (onefile) — copied to src-tauri/binaries/ by build_server.py.

CPU-only. CUDA build excluded; ship a separate spec later if GPU is needed.

Onefile mode: PyInstaller compresses everything into a single executable.
Cold-start cost (10-30 s for first launch each session while libs extract to
/tmp) traded for a ~500 MB sidecar instead of the ~1.2 GB onedir tree.
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

block_cipher = None

datas = []
binaries = []
hiddenimports = [
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

# Library metadata required at runtime (pkg_resources / importlib.metadata).
for pkg in (
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
):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass  # metadata missing in some envs; PyInstaller's discovery will flag if critical.

# Native .so / pickled data / lazy_loader stubs that --hidden-import alone misses.
for pkg in (
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
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Ensure transformers + torch submodules are walked so dynamic imports resolve.
hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("pyannote")

a = Analysis(
    ["../src/podcodex/api/server.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["pyi_hooks"],
    hooksconfig={},
    runtime_hooks=[
        # Sort first so it patches sys.version before the auto-injected
        # pyi_rth_nltk hook triggers cloudpickle's strict version parser.
        "pyi_hooks/pyi_rth_aaa_anaconda_sysver.py",
        "pyi_hooks/pyi_rth_numpy_compat.py",
        "pyi_hooks/pyi_rth_torch_compiler_disable.py",
    ],
    excludes=[
        # CUDA libs — CPU-only build. Saves ~1.5 GB.
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
        # Dev-only modules pulled by transitive deps; not used at runtime.
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="podcodex-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX breaks codesign on macOS, marginal gains elsewhere.
    runtime_tmpdir=None,
    console=True,  # Tauri captures stdout/stderr; --noconsole only for Windows GUI.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
