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

import logging
import os
import shutil
import sys
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
    _install_all_patches()
    _setup_loguru_file_sink()
    _install_stdlib_intercept()
    _log_hf_cache_state()


def bootstrap_for_mcp_stdio() -> None:
    """For ``podcodex-server.exe --mcp`` spawned by Claude Desktop.

    Logs to stderr (Claude captures it). Stdout is the JSON-RPC channel
    and must stay clean — never touch it from here.
    """
    _install_all_patches()
    _setup_loguru_stderr_sink()
    _install_stdlib_intercept()


def bootstrap_for_dev() -> None:
    """For dev uvicorn (``make dev-api``), the Discord bot, and one-off
    scripts. Logs to stderr."""
    _install_all_patches()
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
    _install_all_patches()


# ── Patches ─────────────────────────────────────────────────────────────


def _install_all_patches() -> None:
    """Apply every monkey-patch we install at startup.

    Each ``_install_*`` helper is responsible for logging its own outcome
    (``logger.info`` on apply / skip, ``logger.warning`` on failure).
    Subprocess children rely on these logs to confirm bootstrap actually
    ran — silent failure is what hid the rc.9 torch-check bug.
    """
    logger.info("bootstrap: installing platform patches (pid={})", os.getpid())
    _install_hf_symlink_patch()
    _install_subprocess_console_patch()
    _install_transformers_doc_patch()
    _install_transformers_torch_check_patch()
    logger.info("bootstrap: all patches installed")


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
            if os.path.exists(dst):
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
         module import (lines 472 + 624 of masking_utils.py), so flipping
         the bool is not enough.
      4. Inject ``TransformGetItemToIndex`` into ``masking_utils``'s
         namespace. The module imports it conditionally at line 39-40,
         guarded by the same broken ``_is_torch_greater_or_equal_than_2_6``
         flag — when the gate misfired at import time the symbol was
         never bound, and ``sdpa_mask_recent_torch``'s body resolves it
         via free-variable lookup, raising ``NameError`` at call time.
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
        log_path = Path(data_dir) / "logs" / "server.log"
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

    logging.basicConfig(
        handlers=[_InterceptHandler()], level=logging.INFO, force=True
    )
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
