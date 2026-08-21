"""Application-startup hooks: monkey-patches and logging setup.

Importing ``podcodex`` is *passive* — you get ``__version__`` and nothing
else. Code paths that drive PodCodex as an application — the bundled
FastAPI sidecar, the MCP stdio entry, the dev uvicorn server, the
Discord bot — call exactly one ``bootstrap_for_*()`` function before
real work begins. The choice picks the logging destination and which
patches to install.

The patches mutate global Python state
(``subprocess.Popen.__init__``, ``transformers.utils.doc.*``,
``huggingface_hub.file_download.*``). Pulling them out of import-time
means tests, notebooks, and one-off scripts that touch ``podcodex.*``
don't silently inherit those mutations.

Order contract for the bundled sidecar / MCP entry:
    1. Set ``PODCODEX_DATA_DIR`` env var (caller's responsibility).
    2. Wire ML cache env vars (HF_HOME, TORCH_HOME, ...) BEFORE any
       torch / transformers import.
    3. Call the appropriate ``bootstrap_for_*()`` — installs patches,
       configures loguru.
    4. Import the rest of ``podcodex.*`` and run.

Steps 1-2 must precede (3): bootstrap consumes those env vars but does
not set them. Doing the cache-dir wiring inside bootstrap would couple
unrelated concerns (filesystem layout vs. monkey-patching).
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<level>{message}</level>"
)


# ── Public bootstrap entry points ───────────────────────────────────────


def bootstrap_for_bundled_sidecar() -> None:
    """For the PyInstaller-frozen FastAPI sidecar spawned by Tauri.

    Logs to ``<PODCODEX_DATA_DIR>/logs/server.log``. ``server.py`` has
    already redirected raw stdio to that same file before this is called,
    so we deliberately do NOT add an extra stderr sink (would write every
    line twice).
    """
    _install_eager_patches()
    _arm_ml_patch_hook()
    _check_system_ffmpeg()
    _setup_loguru_file_sink()
    _install_stdlib_intercept()
    _log_hf_cache_state()


def bootstrap_for_mcp_stdio() -> None:
    """For ``podcodex-server.exe --mcp`` spawned by Claude Desktop.

    Logs to stderr (Claude captures it). Stdout is the JSON-RPC channel
    and must stay clean — never touch it from here.
    """
    _install_eager_patches()
    _arm_ml_patch_hook()
    _check_system_ffmpeg()
    _setup_loguru_stderr_sink()
    _install_stdlib_intercept()


def bootstrap_for_dev() -> None:
    """For dev uvicorn (``make dev-api``), the Discord bot, and one-off
    scripts. Logs to stderr."""
    _install_all_patches()
    _check_system_ffmpeg()
    _setup_loguru_stderr_sink()
    _install_stdlib_intercept()


def bootstrap_for_subprocess_child() -> None:
    """For ``multiprocessing.spawn``'d children of the bundled sidecar.

    The child re-execs the frozen binary and goes straight from
    ``server.py:freeze_support`` into ``spawn_main`` → user entry
    function — it never reaches ``main()`` and so never runs
    ``bootstrap_for_bundled_sidecar``. Without this, the child loads
    transformers / FlagEmbedding without the doc patch and crashes
    on ``inspect.getsource`` while *defining* model classes.

    Installs only the patches; logging is handled by
    ``subprocess_runner._install_log_forwarder``, which forwards the
    child's loguru output to the parent over an IPC queue. Adding a
    file sink here would race the parent's ``enqueue=True`` sink for
    the same ``server.log``.
    """
    _patch_missing_stdio()
    _install_all_patches()
    _wire_ffmpeg_path()


def _wire_ffmpeg_path() -> None:
    """Prepend the resolved ffmpeg dir to PATH for bare-``"ffmpeg"`` shellouts.

    Spawned children normally inherit the parent's wired PATH, but if the
    parent failed to resolve ffmpeg at its startup (installed or repaired
    after launch), the child re-resolves independently here so whisperx /
    faster-whisper shellouts don't die with WinError 2.
    """
    from podcodex.core._ffmpeg import ffmpeg_override_dir

    try:
        ffmpeg_dir = ffmpeg_override_dir()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffmpeg wiring: resolution failed: {!r}", exc)
        return
    if not ffmpeg_dir:
        return
    existing = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_dir not in existing:
        os.environ["PATH"] = os.pathsep.join([ffmpeg_dir, *existing])
        logger.info("ffmpeg wiring: prepended {} to PATH", ffmpeg_dir)


def _patch_missing_stdio() -> None:
    """Windows --noconsole spawn children get ``sys.stdout``/``stderr`` = None.
    Libraries that call ``sys.stdout.write`` directly (torch.hub download
    progress, tqdm, etc.) crash with ``AttributeError`` before our loguru
    forwarder ever sees the failure. Redirect to devnull so writes are silent
    no-ops; loguru output still reaches the parent over the IPC queue.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        devnull = open(os.devnull, "w", buffering=1, encoding="utf-8")
    except Exception:
        return
    if sys.stdout is None:
        sys.stdout = devnull
    if sys.stderr is None:
        sys.stderr = devnull


# ── Patches ─────────────────────────────────────────────────────────────


def _install_all_patches() -> None:
    """Apply every monkey-patch eagerly, ML ones included.

    Used by the dev server, the bot, and multiprocessing children — all
    contexts that either start ML work immediately or want a broken CUDA
    wheel to fail loudly at startup rather than mid-request.

    ``_install_torch_from_numpy_patch`` runs before the transformers
    patches because both end up importing torch and we want the numpy ABI
    fallback installed before any caller hits ``torch.from_numpy``.
    """
    _install_eager_patches()
    _install_ml_patches()
    logger.info("bootstrap: all patches installed")


def _install_ml_patches() -> None:
    """The patches that need torch or transformers loaded."""
    _install_torch_patches()
    _install_transformers_patches()
    _check_cuda_kernels_or_degrade()


def _install_eager_patches() -> None:
    """The patches that cost nothing: no torch, no transformers, no model IO."""
    logger.info("bootstrap: installing platform patches (pid={})", os.getpid())
    _wire_ssl_certs()
    _apply_persisted_device_override()
    _install_hf_symlink_patch()
    _install_subprocess_console_patch()


def _install_transformers_patches() -> None:
    _install_transformers_doc_patch()
    _install_transformers_torch_check_patch()


def _install_torch_patches() -> None:
    """Fired once ``torch`` has finished importing.

    Deliberately does *not* run the kernel guard: this executes inside an
    ``import torch`` statement, where raising would make importlib evict
    torch and force a full multi-second re-import on the next try. The
    guard is instead owned by ``core.device``, which runs it lazily before
    reading the override and raises from ``resolve_device`` — the place
    that already validates a forced CUDA request.
    """
    _install_torch_from_numpy_patch()


class _PostExecLoader:
    """Loader proxy that runs ``callback`` after the wrapped loader execs.

    Everything not overridden is delegated, so PyInstaller's FrozenImporter
    (which is finder and loader both) keeps working as itself.
    """

    def __init__(self, inner, callback) -> None:
        self._inner = inner
        self._callback = callback

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module) -> None:
        self._inner.exec_module(module)
        try:
            self._callback()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "deferred callback failed after importing {}: {!r}",
                module.__name__,
                exc,
            )
        finally:
            # Step out of the way. Left in place this proxy stays as
            # ``torch.__loader__`` for the process lifetime, routing every
            # later loader call (get_source, get_data — linecache,
            # importlib.resources, and the inspect.getsource paths the
            # transformers doc patch exists for) through an extra
            # ``__getattr__`` hop, and pinning the callback.
            module.__loader__ = self._inner
            if module.__spec__ is not None:
                module.__spec__.loader = self._inner
            self._callback = None

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _DeferredImportFinder:
    """``sys.meta_path`` finder that runs a callback the moment a named
    module finishes importing.

    This is the single facility for "do X, but only once someone actually
    pays for Y". Three registrations use it today, all for the same reason:
    the setup work is cheap, the import that makes it necessary is not, and
    a launch pays for neither unless it uses the feature.

      torch          ~4 s in the frozen bundle
      transformers   ~3 s
      nltk           ~1.6 s

    Why a hook rather than an ``ensure_patches()`` call at each entry
    point: a forgotten call site does not fail visibly, it fails as an
    ``inspect.getsource`` OSError while a model class is being defined, or
    as a vmap RuntimeError inside a forward pass. Hooking the import makes
    the guarantee structural instead of a convention to remember.

    Ordering: a callback runs *after* the module body executes, so
    ``transformers``' own ``import torch`` fires the torch callback first
    and the transformers callback never sees a half-initialised package.

    The finder stays on ``sys.meta_path`` once its callbacks have fired
    rather than unhooking itself. ``importlib._bootstrap._find_spec``
    iterates the live list by index, so removing an entry from inside
    another module's ``exec_module`` can shift a concurrent importer's
    cursor past a finder it still needed — a spurious ModuleNotFoundError
    in an unrelated module, and the API serves imports from a threadpool.
    Staying costs one dict lookup per import.
    """

    def __init__(self) -> None:
        self._pending: dict[str, Callable[[], None]] = {}

    def register(self, module_name: str, callback: Callable[[], None]) -> None:
        self._pending[module_name] = callback

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._pending:
            return None

        # Delegate to the finders that actually know where the module is
        # (PathFinder in dev, PyInstaller's FrozenImporter when bundled).
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None

        # Bind the callback to *execution*, not resolution. transformers
        # probes for torch with a bare ``importlib.util.find_spec("torch")``
        # (that is how ``is_torch_available()`` works) and throws the spec
        # away — consuming the callback here would arm a trigger that never
        # fires and leave torch unpatched.
        spec.loader = _PostExecLoader(
            spec.loader, functools.partial(self._fire, fullname)
        )
        return spec

    def _fire(self, fullname: str) -> None:
        callback = self._pending.pop(fullname, None)
        if callback is None:
            return  # already fired; a re-import is not a second chance
        callback()


def _get_deferred_finder() -> _DeferredImportFinder:
    """The one finder instance, installed on first use."""
    for finder in sys.meta_path:
        if isinstance(finder, _DeferredImportFinder):
            return finder
    finder = _DeferredImportFinder()
    sys.meta_path.insert(0, finder)
    return finder


def defer_until_imported(module_name: str, callback: Callable[[], None]) -> None:
    """Run ``callback`` right after ``module_name`` next finishes importing.

    Calls it immediately instead when the module is already in
    ``sys.modules``, since the hook would then never fire. Public because
    the PyInstaller runtime hook for nltk uses it too — see
    ``packaging/pyi_hooks/rthooks/pyi_rth_nltk_lazy.py``.
    """
    if module_name in sys.modules:
        callback()
        return
    _get_deferred_finder().register(module_name, callback)


def _arm_ml_patch_hook() -> None:
    """Defer the torch / transformers patches to first use of either."""
    for name in ("torch", "transformers"):
        if name in sys.modules:
            # Already imported by something upstream — the hook would never
            # fire, so patch now and keep the eager guarantee. Only the ML
            # half: the caller ran _install_eager_patches one line ago, and
            # repeating it would re-wire ssl and log "installing platform
            # patches" twice, which reads as a bug in a support log.
            logger.info("bootstrap: {} already imported, patching eagerly", name)
            _install_ml_patches()
            return
    defer_until_imported("torch", _install_torch_patches)
    defer_until_imported("transformers", _install_transformers_patches)
    logger.info("bootstrap: ML patches deferred to first torch/transformers import")


def _wire_ssl_certs() -> None:
    """Point stdlib ``ssl`` at certifi's CA bundle.

    The frozen bundle's OpenSSL carries a compiled-in default cert path
    from the build machine that does not exist on user machines, so any
    stdlib HTTPS client (urllib, and therefore torch.hub align-model
    downloads, RSS enclosure fetches, cover-art fetches) fails with
    CERTIFICATE_VERIFY_FAILED. requests/httpx are unaffected: they use
    certifi on their own. ``setdefault`` keeps user-provided overrides
    (corporate CA bundles) intact.
    """
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def _apply_persisted_device_override() -> None:
    """Promote the persisted ``device_override`` setting to ``PODCODEX_DEVICE``.

    Env var wins when explicitly set (dev / CI / ``make dev-no-tauri-cpu``);
    otherwise the value the user picked in the desktop GPU panel is read from
    ``data_dir()/settings.json`` and exported so every downstream
    ``device.user_override()`` call sees it.
    """
    if os.environ.get("PODCODEX_DEVICE", "").strip():
        return
    try:
        from podcodex.core.user_settings import get_device_override

        value = get_device_override()
    except Exception as exc:  # noqa: BLE001
        logger.warning("device override: persisted setting load failed: {!r}", exc)
        return
    if value != "auto":
        os.environ["PODCODEX_DEVICE"] = value
        logger.info("device override: applied persisted setting {!r}", value)


def _check_cuda_kernels_or_degrade() -> None:
    """Run the device kernel guard eagerly and re-raise a forced-CUDA mismatch.

    The degrade logic itself lives in ``core.device.ensure_kernel_guard``
    so the lazy path (bundled sidecar / MCP, where torch is imported on
    first use) gets the same answer. This wrapper adds only the eager
    contract the dev server and multiprocessing children want: if the user
    forced ``PODCODEX_DEVICE=cuda`` on a wheel without kernels for their
    GPU, fail at startup rather than mid-pipeline.
    """
    from podcodex.core.device import (
        ensure_kernel_guard,
        kernel_guard_error,
        user_override,
    )

    ensure_kernel_guard()
    error = kernel_guard_error()
    if error is not None and user_override() == "cuda":
        raise error


def _check_system_ffmpeg() -> None:
    """Warn if no system ffmpeg is reachable. Frontend surfaces ``/health``
    capability so users see an install dialog before the first job fails."""
    from podcodex.core._ffmpeg import PODCODEX_FFMPEG_EXE_ENV, ffmpeg_available

    if ffmpeg_available():
        return
    logger.warning(
        "ffmpeg: not found on PATH. Transcription / synthesis / clip extraction "
        "will fail. Install from https://ffmpeg.org/download.html, or set "
        "{} to point at the binary.",
        PODCODEX_FFMPEG_EXE_ENV,
    )


def _install_hf_symlink_patch() -> None:
    """Force huggingface_hub to copy files instead of symlinking on Windows.

    HF Hub's cache uses ``os.symlink`` (snapshot → blob). On Windows that
    needs ``SeCreateSymbolicLinkPrivilege`` (Developer Mode or admin).
    Worse, ctranslate2's C++ ``fopen`` on the ``model.bin`` symlink fails
    with 'Unable to open file' even when the privilege is granted. Copy
    instead — loses dedup across model versions but every library can
    open the result.
    """
    if sys.platform != "win32":
        logger.debug("hf-symlink patch: skipped (not win32)")
        return
    try:
        from huggingface_hub import file_download as _hf_file_download
    except Exception as exc:  # noqa: BLE001
        logger.warning("hf-symlink patch: huggingface_hub import failed: {!r}", exc)
        return

    def _copy_instead_of_symlink(src, dst, new_blob: bool = False) -> None:
        try:
            os.remove(dst)
        except OSError:
            pass
        real_src = os.path.realpath(src)
        shutil.copy(real_src, dst)

    _hf_file_download._create_symlink = _copy_instead_of_symlink
    logger.info("hf-symlink patch: applied")


def _install_subprocess_console_patch() -> None:
    """OR ``CREATE_NO_WINDOW`` into every ``subprocess.Popen`` on Windows.

    whisperx → ffmpeg, faster-whisper, nvidia-smi, uv pip — without this
    each shell-out flashes a console window for a fraction of a second
    in a GUI-subsystem app. No-op on macOS/Linux.
    """
    if sys.platform != "win32":
        logger.debug("subprocess-console patch: skipped (not win32)")
        return
    import subprocess

    if not hasattr(subprocess, "CREATE_NO_WINDOW"):
        logger.warning("subprocess-console patch: CREATE_NO_WINDOW unavailable")
        return
    original_init = subprocess.Popen.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["creationflags"] = (
            kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        )
        return original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init
    logger.info("subprocess-console patch: applied")


def _install_torch_from_numpy_patch() -> None:
    """Make ``torch.from_numpy`` survive the numpy 2.x / torch ABI mismatch.

    Torch wheels compiled against numpy 1.x can fail in frozen bundles where
    the FrozenImporter loads numpy 2.x first; ``torch.from_numpy`` then raises
    ``RuntimeError: Numpy is not available`` even though numpy is importable.
    Wrap ``torch.from_numpy`` with a ctypes ``memmove`` fallback that bypasses
    the C-level numpy ABI version check. Adapted from voicebox.

    Runs synchronously after ``import torch`` finishes — never mid-init —
    which is the whole point of this function existing instead of the old
    polling-thread runtime hook.
    """
    try:
        import ctypes

        import numpy as np
        import torch
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch from_numpy patch: import failed: {!r}", exc)
        return

    if getattr(torch, "_podcodex_from_numpy_patched", False):
        return

    _orig = torch.from_numpy
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }

    def _safe_from_numpy(arr):
        try:
            return _orig(arr)
        except RuntimeError:
            a = np.ascontiguousarray(arr)
            key = str(a.dtype)
            if key not in dtype_map:
                raise TypeError(
                    f"torch from_numpy patch: unsupported numpy dtype {key!r}; "
                    "add an explicit mapping rather than silently copying bytes "
                    "into the wrong dtype."
                )
            out = torch.empty(list(a.shape), dtype=dtype_map[key])
            ctypes.memmove(out.data_ptr(), a.ctypes.data, a.nbytes)
            return out

    torch.from_numpy = _safe_from_numpy
    torch._podcodex_from_numpy_patched = True
    logger.info("torch from_numpy patch: applied")


def _install_transformers_torch_check_patch() -> None:
    """Make transformers' torch version gate work in a PyInstaller bundle.

    ``--copy-metadata torch`` doesn't always make the dist-info findable
    from _MEIPASS, so ``importlib.metadata.version("torch")`` raises and
    transformers' ``is_torch_greater_or_equal`` returns False for a
    perfectly working 2.8 build. That cascades into ``masking_utils``
    picking ``sdpa_mask_older_torch`` (vmap-based), which crashes Pplx's
    ``or_masks`` mask function (``RuntimeError: vmap`` on CPU,
    ``device-side assert`` on CUDA).

    Four layers of defense, all cheap:
      1. Replace ``is_torch_greater_or_equal`` to read ``torch.__version__``
         directly — covers other transformers call sites we haven't hit yet.
      2. Force-load ``masking_utils`` and overwrite its cached bools, in
         case it imported before our function patch lands.
      3. Rebind ``sdpa_mask`` and the ``AttentionMaskInterface`` registry —
         dispatch happens by reference at call time and was sealed at
         module import, so flipping the bool is not enough.
      4. Inject ``TransformGetItemToIndex`` into ``masking_utils``'s
         namespace. The module imports it conditionally, guarded by the
         same broken ``_is_torch_greater_or_equal_than_2_6`` flag — when
         the gate misfired at import time the symbol was never bound,
         and ``sdpa_mask_recent_torch``'s body resolves it via free-variable
         lookup, raising ``NameError`` at call time.
    """
    try:
        from packaging import version as _v
        from transformers.utils import import_utils as _iu

        def _patched(library_version: str, accept_dev: bool = False) -> bool:
            try:
                import torch

                base = _v.parse(getattr(torch, "__version__", "0.0")).base_version
                return _v.parse(base) >= _v.parse(library_version)
            except Exception:  # noqa: BLE001
                return False

        _iu.is_torch_greater_or_equal = _patched

        import torch
        import transformers.masking_utils as _mu

        # ``2.8.0+cpu`` / ``2.8.0+cu128`` parse fine, but compare oddly
        # against ``2.6``; the base version strips the local tag.
        base = _v.parse(getattr(torch, "__version__", "0.0")).base_version
        for attr, threshold in (
            ("_is_torch_greater_or_equal_than_2_5", "2.5"),
            ("_is_torch_greater_or_equal_than_2_6", "2.6"),
        ):
            if hasattr(_mu, attr):
                setattr(_mu, attr, _v.parse(base) >= _v.parse(threshold))

        rebind = "skipped"
        tgi_inject = "skipped"
        if _v.parse(base) >= _v.parse("2.6") and hasattr(_mu, "sdpa_mask_recent_torch"):
            _mu.sdpa_mask = _mu.sdpa_mask_recent_torch
            if hasattr(_mu, "AttentionMaskInterface"):
                _mu.AttentionMaskInterface._global_mapping["sdpa"] = (
                    _mu.sdpa_mask_recent_torch
                )
            rebind = "recent"
            if not hasattr(_mu, "TransformGetItemToIndex"):
                try:
                    from torch._dynamo._trace_wrapped_higher_order_op import (
                        TransformGetItemToIndex as _TGI,
                    )

                    _mu.TransformGetItemToIndex = _TGI
                    tgi_inject = "ok"
                except Exception as exc:  # noqa: BLE001
                    tgi_inject = f"failed: {exc!r}"
            else:
                tgi_inject = "already-present"

        logger.info(
            "transformers torch-check patch applied "
            "(torch={}, ge_2_6={}, sdpa_rebind={}, tgi_inject={})",
            base,
            getattr(_mu, "_is_torch_greater_or_equal_than_2_6", "?"),
            rebind,
            tgi_inject,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("transformers torch-check patch failed: {!r}", exc)


def _install_transformers_doc_patch() -> None:
    """Make transformers' docstring decorator survive a no-source bundle.

    ``transformers.utils.doc.get_docstring_indentation_level`` calls
    ``inspect.getsource(func)`` while *defining* a model class. PyInstaller
    bundles ship ``.pyc`` files but no ``.py`` source, so it raises
    ``OSError`` and the model module fails to import. Returning 0 indent
    is harmless — the value is only used to re-indent the docstring,
    which we never read at runtime.
    """
    try:
        from transformers.utils import doc as _doc

        _orig = _doc.get_docstring_indentation_level

        def _safe_get_indent(func):
            try:
                return _orig(func)
            except OSError:
                return 0

        _doc.get_docstring_indentation_level = _safe_get_indent
        logger.info("transformers doc patch: applied")
    except Exception as exc:  # noqa: BLE001
        logger.warning("transformers doc patch failed: {!r}", exc)


# ── Logging ─────────────────────────────────────────────────────────────


def _setup_loguru_file_sink() -> None:
    """Bundled sidecar: file sink at ``PODCODEX_DATA_DIR/logs/server.log``.

    Falls back to stderr if no data dir is set (someone ran the frozen
    binary directly from a checkout, etc.).
    """
    logger.remove()
    data_dir = os.environ.get("PODCODEX_DATA_DIR")
    if not data_dir:
        if sys.stderr is not None:
            try:
                logger.add(sys.stderr, format=_LOG_FORMAT, level="INFO")
            except Exception:  # noqa: BLE001
                pass
        return

    try:
        from podcodex.core.app_paths import server_log_path

        log_path = server_log_path(data_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format=_LOG_FORMAT,
            level="DEBUG",
            rotation="10 MB",
            retention=3,
            enqueue=True,
            backtrace=True,
            diagnose=False,
        )
        # Patch raw sys.stdout/stderr if missing — Windows --noconsole spawn
        # children get None for both, which crashes any library doing
        # sys.stdout.write directly (torch.hub progress, tqdm, etc.).
        if sys.stdout is None or sys.stderr is None:
            _stdio_fp = open(log_path, "a", buffering=1, encoding="utf-8")
            if sys.stdout is None:
                sys.stdout = _stdio_fp
            if sys.stderr is None:
                sys.stderr = _stdio_fp
        logger.info(
            "sidecar logging started — pid={}, frozen={}, data_dir={}",
            os.getpid(),
            getattr(sys, "frozen", False),
            data_dir,
        )
    except Exception as exc:  # noqa: BLE001
        try:
            fallback = Path.home() / "podcodex-startup-error.log"
            fallback.write_text(
                f"Failed to set up sidecar log file: {exc!r}\n",
                encoding="utf-8",
            )
        except Exception:
            pass


def _setup_loguru_stderr_sink() -> None:
    """MCP stdio + dev: stderr only. MCP must keep stdout clean for JSON-RPC."""
    logger.remove()
    if sys.stderr is None:
        return
    try:
        logger.add(sys.stderr, format=_LOG_FORMAT, level="INFO")
    except Exception:  # noqa: BLE001
        pass


def _log_hf_cache_state() -> None:
    """Log the HuggingFace cache contents at startup for diagnostics."""
    try:
        from podcodex.core._hf_logging import log_cached_models

        log_cached_models()
    except Exception as exc:  # noqa: BLE001
        logger.warning("hf-cache state log failed: {!r}", exc)


def _install_stdlib_intercept() -> None:
    """Route stdlib ``logging`` records into loguru for unified format.

    Sets named loggers (uvicorn, fastapi, mcp, ...) to ``propagate=False``
    with their own InterceptHandler so messages don't double-emit through
    root.
    """

    class _InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)
    for _name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "fastapi",
        "mcp",
        "FastMCP",
        "mcp.server.streamable_http_manager",
        "mcp.server.lowlevel",
    ):
        _lg = logging.getLogger(_name)
        _lg.handlers = [_InterceptHandler()]
        _lg.propagate = False
