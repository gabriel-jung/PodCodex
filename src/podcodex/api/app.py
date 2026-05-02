"""PodCodex API — FastAPI application factory and entry point."""

from __future__ import annotations

import os

# Prevent multiprocessing/OpenMP deadlocks when PyTorch DataLoaders run
# inside ThreadPoolExecutor threads (used by the task runner).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv


import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from podcodex.api.routes import (
    api_keys,
    audio,
    batch,
    bot_access,
    bundle,
    config,
    episodes as episodes_route,
    export,
    filesystem,
    gpu,
    health,
    index,
    integrations,
    mcp_prompts as mcp_prompts_route,
    models,
    correct,
    provider_profiles,
    rss,
    search,
    shows,
    synthesize,
    transcribe,
    translate,
    ws,
    youtube,
)

load_dotenv()
# User-scoped secrets file — survives packaged installs and overrides repo .env.
from podcodex.core.app_paths import secrets_env_path as _secrets_env_path  # noqa: E402

_secrets_env = _secrets_env_path()
if _secrets_env.exists():
    load_dotenv(_secrets_env, override=True)


# ── Optional MCP (desktop extra) ────────────────────────────────────────
try:
    from podcodex.mcp.server import mcp as _mcp  # type: ignore[import-not-found]
except Exception as exc:
    _mcp = None
    _mcp_import_error: Exception | None = exc
else:
    _mcp_import_error = None


async def _watch_parent(parent_pid: int) -> None:
    """Self-terminate when the Tauri shell dies abruptly.

    The Rust shell injects ``PODCODEX_PARENT_PID`` and normally kills the
    sidecar process group on ``RunEvent::Exit``. That callback doesn't fire
    on SIGKILL / Force Quit / panic, so this poll is the fallback: every
    2s, check the parent still exists; on disappearance, raise SIGTERM at
    ourselves so uvicorn runs lifespan teardown.

    Windows already gets KILL_ON_JOB_CLOSE via ``command_group``, so the
    Rust shell skips setting the env on that platform path if desired —
    this watcher is a no-op when the var is unset.
    """
    while True:
        await asyncio.sleep(2.0)
        try:
            os.kill(parent_pid, 0)
        except ProcessLookupError:
            logger.warning(f"parent process {parent_pid} gone, shutting down sidecar")
            os.kill(os.getpid(), signal.SIGTERM)
            return
        except PermissionError:
            # Process exists but is owned by someone else — still alive.
            pass


def _warmup_caches_sync() -> None:
    """Pre-open LanceDB and per-show pipeline.db connections.

    Without this, the first ``GET /api/shows/{folder}/unified`` after a
    process boot pays the ``import lancedb`` + ``lancedb.connect()`` cost
    inside the request, making the user's first show open feel sluggish
    (~10s on cold OS cache). Both are process-wide singletons, so warming
    them once at startup eliminates that delay.
    """
    try:
        from podcodex.rag.index_store import get_index_store

        store = get_index_store()
        # Touch the metadata table so the connection actually loads.
        store.list_collections()
    except Exception:
        logger.opt(exception=True).debug("warmup: index store failed")

    try:
        from podcodex.api.routes.config import _load
        from podcodex.core.pipeline_db import get_pipeline_db

        cfg = _load()
        for folder in cfg.show_folders:
            p = Path(folder)
            if not p.is_dir():
                continue
            try:
                get_pipeline_db(p)
            except Exception:
                logger.opt(exception=True).debug(
                    f"warmup: pipeline_db open failed for {p}"
                )
    except Exception:
        logger.opt(exception=True).debug("warmup: pipeline_db pass failed")


def _make_lifespan(mcp_http):
    """Nest the mounted MCP sub-app's lifespan under FastAPI's."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        watcher_task: asyncio.Task | None = None
        parent_pid_raw = os.environ.get("PODCODEX_PARENT_PID", "").strip()
        if parent_pid_raw and sys.platform != "win32":
            try:
                parent_pid = int(parent_pid_raw)
            except ValueError:
                parent_pid = 0
            if parent_pid > 0:
                watcher_task = asyncio.create_task(_watch_parent(parent_pid))

        # Fire-and-forget on a worker thread. ``asyncio.to_thread`` cannot
        # actually interrupt the underlying thread, so there's no point trying
        # to cancel; on shutdown we just await whatever is left.
        warmup_task = asyncio.create_task(asyncio.to_thread(_warmup_caches_sync))

        try:
            if mcp_http is not None:
                async with mcp_http.router.lifespan_context(app):
                    yield
            else:
                yield
        finally:
            try:
                await warmup_task
            except Exception:
                pass
            if watcher_task is not None:
                watcher_task.cancel()
                try:
                    await watcher_task
                except (asyncio.CancelledError, Exception):
                    pass

    return lifespan


def _register_show_folder_resolver() -> None:
    """Wire a show-name → folder resolver into the IndexStore.

    Enables the ``episode_title`` backfill (``IndexStore._ensure_episode_title_backfill``)
    to locate each episode's ``.episode_meta.json`` when healing chunks whose
    RSS title never made it into the transcript meta.
    """
    try:
        from podcodex.api.routes.config import _load as _load_cfg
        from podcodex.ingest.show import load_show_meta
        from podcodex.rag.index_store import IndexStore
    except Exception:
        logger.opt(exception=True).debug("show folder resolver: import failed")
        return

    def resolve(show_name: str):
        try:
            cfg = _load_cfg()
        except Exception:
            return None
        target = (show_name or "").strip().lower()
        if not target:
            return None
        for folder_path in cfg.show_folders:
            child = Path(folder_path)
            if not child.is_dir():
                continue
            meta = load_show_meta(child)
            name = (meta.name if meta else None) or child.name
            if name.strip().lower() == target:
                return child
        return None

    IndexStore.set_show_folder_resolver(resolve)


# Frontend's `json()` and direct fetch sites must send this header on
# state-changing requests; the middleware below enforces it.
CSRF_HEADER = "X-PodCodex"
CSRF_VALUE = "1"
_CSRF_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_CSRF_EXEMPT_PREFIXES = ("/mcp",)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    mcp_http = _mcp.streamable_http_app() if _mcp is not None else None
    _register_show_folder_resolver()

    # Clean up atomic-write temp orphans left by any prior hard-crash.
    try:
        from podcodex.core.recovery import run_startup_recovery

        run_startup_recovery()
    except Exception:
        logger.opt(exception=True).debug("startup recovery failed")

    app = FastAPI(
        title="PodCodex",
        version="0.1.0",
        description="Podcast processing pipeline API",
        lifespan=_make_lifespan(mcp_http),
    )
    app.state.mcp_available = _mcp is not None
    if _mcp_import_error is not None:
        logger.warning(f"MCP extra unavailable: {_mcp_import_error}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:18811",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:18811",
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom header forces a CORS preflight that the origin allowlist rejects,
    # so a drive-by <form> on a malicious page can't reach mutating endpoints.
    @app.middleware("http")
    async def _csrf_guard(request: Request, call_next):
        if request.method in _CSRF_METHODS and not request.url.path.startswith(
            _CSRF_EXEMPT_PREFIXES
        ):
            if request.headers.get(CSRF_HEADER.lower()) != CSRF_VALUE:
                return JSONResponse(
                    {"detail": "CSRF token missing"},
                    status_code=403,
                )
        return await call_next(request)

    app.include_router(health.router, prefix="/api", tags=["system"])
    app.include_router(config.router, prefix="/api", tags=["config"])
    app.include_router(api_keys.router, prefix="/api/keys", tags=["api-keys"])
    app.include_router(
        provider_profiles.router,
        prefix="/api/provider-profiles",
        tags=["provider-profiles"],
    )
    app.include_router(audio.router, prefix="/api/audio", tags=["audio"])
    app.include_router(filesystem.router, prefix="/api/fs", tags=["filesystem"])
    app.include_router(shows.router, prefix="/api/shows", tags=["shows"])
    app.include_router(rss.router, prefix="/api/shows", tags=["rss"])
    app.include_router(youtube.router, prefix="/api/shows", tags=["youtube"])
    app.include_router(transcribe.router, prefix="/api/transcribe", tags=["transcribe"])
    app.include_router(correct.router, prefix="/api/correct", tags=["correct"])
    app.include_router(translate.router, prefix="/api/translate", tags=["translate"])
    app.include_router(synthesize.router, prefix="/api/synthesize", tags=["synthesize"])
    app.include_router(index.router, prefix="/api/index", tags=["index"])
    app.include_router(search.router, prefix="/api/search", tags=["search"])
    app.include_router(episodes_route.router, prefix="/api/episodes", tags=["episodes"])
    app.include_router(ws.router, prefix="/api", tags=["ws"])

    app.include_router(batch.router, prefix="/api/batch", tags=["batch"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])
    app.include_router(export.router, prefix="/api/export", tags=["export"])
    app.include_router(bundle.router, prefix="/api/bundle", tags=["bundle"])
    app.include_router(gpu.router, prefix="/api/gpu", tags=["gpu"])
    app.include_router(
        integrations.router, prefix="/api/integrations", tags=["integrations"]
    )
    app.include_router(mcp_prompts_route.router, prefix="/api/mcp", tags=["mcp"])
    app.include_router(bot_access.router, prefix="/api/bot-access", tags=["bot-access"])

    if mcp_http is not None:
        app.mount("/mcp", mcp_http)

    return app


_DEFAULT_API_PORT = 18811
_API_PORT = int(os.environ.get("PODCODEX_API_PORT", _DEFAULT_API_PORT))

app = create_app()
app.state.api_port = _API_PORT


def main() -> None:
    """Entry point for ``podcodex-api`` script."""
    from podcodex.bootstrap import bootstrap_for_dev

    bootstrap_for_dev()
    uvicorn.run(
        "podcodex.api.app:app",
        host="127.0.0.1",
        port=_API_PORT,
        reload=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
