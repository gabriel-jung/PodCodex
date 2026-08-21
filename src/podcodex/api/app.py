"""PodCodex API — FastAPI application factory and entry point."""

from __future__ import annotations

import os

# Prevent multiprocessing/OpenMP deadlocks when PyTorch DataLoaders run
# inside ThreadPoolExecutor threads (used by the task runner).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import asyncio
import secrets
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv


import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers, QueryParams
from starlette.websockets import WebSocketClose
from loguru import logger

from podcodex import __version__
from podcodex.bootstrap import defer_until_imported
from podcodex.core.api_token import (
    TOKEN_HEADER,
    TOKEN_QUERY_PARAM,
    get_or_create_api_token,
)
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


# ── Optional MCP (desktop extra), mounted lazily ────────────────────────
#
# Importing ``podcodex.mcp.server`` pulls in ``mcp.server.fastmcp``, which
# costs ~0.75 s (most of it jsonschema loading its rfc3987_syntax format
# checker) — measured as 45% of this module's whole import time. Nothing
# reaches /mcp during an app launch; only Claude Desktop or Claude Code
# connecting does, and that happens minutes later if at all. So the sub-app
# is built on the first request instead of at startup.


def _mcp_installed() -> bool:
    """Whether the MCP extra looks importable, without importing it.

    ``find_spec`` resolves the module without executing it, which is the
    whole point: answering "is MCP available?" must not cost what importing
    MCP costs. It is therefore optimistic — it cannot see a broken
    transitive dependency or an error inside ``podcodex.mcp.server``. The
    first ``/mcp`` request finds out for real and corrects
    ``app.state.mcp_available`` (see ``_run_mcp_when_requested``).
    """
    from importlib.util import find_spec

    try:
        return find_spec("mcp") is not None
    except (ImportError, ValueError):
        return False


class _LazyMCPMount:
    """ASGI app that hands off to an MCP sub-app built on first request.

    It does not build the sub-app itself. The streamable-http session
    manager runs on anyio cancel scopes, and anyio requires the task that
    *enters* a scope to be the task that exits it — so a request handler
    cannot open the sub-app's lifespan and leave shutdown to close it. All
    of that is owned by :func:`_run_mcp_when_requested`, one task spawned by
    the parent lifespan; this class only signals it and waits.

    Starlette's ``Mount`` does not forward lifespan scopes to sub-apps, so
    this only ever sees http/websocket scopes.
    """

    #: Generous: the deferred import alone is ~0.75 s and a cold filesystem
    #: makes it slower. Only a genuinely wedged startup should hit this.
    START_TIMEOUT_S = 30.0

    def __init__(self) -> None:
        self.requested = asyncio.Event()
        self.ready = asyncio.Event()
        self.app = None

    async def __call__(self, scope, receive, send) -> None:
        self.requested.set()
        if not self.ready.is_set():
            try:
                # Bounded so a sub-app that wedges during startup answers the
                # client instead of holding its connection open indefinitely.
                await asyncio.wait_for(self.ready.wait(), self.START_TIMEOUT_S)
            except TimeoutError:
                logger.warning("MCP sub-app did not start within the timeout")
        if self.app is None:
            await JSONResponse(
                {"detail": "MCP support is not available in this build"},
                status_code=503,
            )(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _build_mcp_app():
    """Import the MCP server and build its streamable-http ASGI app.

    Blocking and slow (the ~0.75 s import this whole dance exists to defer),
    so callers run it on a worker thread rather than stalling the loop.
    """
    from podcodex.mcp.server import mcp as _mcp

    return _mcp.streamable_http_app()


async def _run_mcp_when_requested(app: FastAPI, mount: _LazyMCPMount) -> None:
    """Own the MCP sub-app's whole life in a single task.

    Waits for the first /mcp request, builds the sub-app off-loop, enters
    its lifespan, and holds until the parent lifespan cancels this task —
    at which point the ``async with`` unwinds in the same task that opened
    it, which is what anyio's cancel scopes require. Ordering therefore
    matches the old eager nesting: the sub-app shuts down with the app.
    """
    await mount.requested.wait()
    try:
        sub = await asyncio.to_thread(_build_mcp_app)
        async with sub.router.lifespan_context(app):
            mount.app = sub
            mount.ready.set()
            logger.info("MCP sub-app mounted on first request")
            await asyncio.Event().wait()  # hold open until cancelled
    except Exception as exc:
        # Logged once and not retried: a missing extra will not appear
        # mid-process, and repeating the slow import per request would turn
        # a misconfiguration into a performance problem too. CancelledError
        # is a BaseException, so shutdown passes straight through here.
        #
        # Correct the advertised capability too. It was answered by
        # find_spec, which only proves the `mcp` package resolves — it
        # cannot see a broken transitive dep or an error inside
        # podcodex.mcp.server. Without this the Settings panel keeps
        # offering the Claude Desktop wiring for a surface that answers 503.
        app.state.mcp_available = False
        logger.warning(f"MCP sub-app unavailable: {exc!r}")
    finally:
        # Released on every exit path, including the failures above: a
        # waiter that never gets this event blocks its request forever, so
        # a failed startup has to end as a 503 rather than a hang. The
        # streamable-http session manager in particular refuses to start
        # twice in one process.
        mount.app = None
        mount.ready.set()


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


def _run_recovery_sync() -> None:
    """Delete atomic-write temp orphans left by a prior hard crash."""
    try:
        from podcodex.core.recovery import run_startup_recovery

        run_startup_recovery()
    except Exception:
        logger.opt(exception=True).debug("startup recovery failed")


def _warmup_caches_sync() -> None:
    """Pre-open LanceDB and the per-show pipeline.db connections.

    Runs on a worker thread, which is the only part of startup that is
    genuinely after the bind: uvicorn awaits ``lifespan.startup()`` *before*
    it opens the socket, so anything awaited there still counts against
    time-to-``/api/health``.

    Without this, the first ``GET /api/shows/{folder}/unified`` after a
    process boot pays the ``import lancedb`` + ``lancedb.connect()`` cost
    inside the request, making the user's first show open feel sluggish
    (~10s on cold OS cache). Both are process-wide singletons, so warming
    them once at startup eliminates that delay.
    """
    try:
        from podcodex.rag.index_store import get_index_store

        store = get_index_store()
        # Touch the metadata table so the connection actually loads. Full
        # info read (not just names) so the artwork_url backfill runs at
        # startup, with the show-folder resolver already registered.
        store.get_all_collection_info()
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Post-bind startup and shutdown.

    Everything here runs *after* uvicorn awaits it but before it binds,
    so keep the body to scheduling: the actual work belongs on the
    worker threads below, which are the only part that runs after the
    socket is listening.
    """
    # Task submission happens on the threadpool (routes are sync `def`),
    # so the progress broadcaster's loop has to be captured here, the one
    # place we are reliably in async context.
    from podcodex.api.tasks import task_manager

    task_manager.bind_loop()

    watcher_task: asyncio.Task | None = None
    parent_pid_raw = os.environ.get("PODCODEX_PARENT_PID", "").strip()
    if parent_pid_raw and sys.platform != "win32":
        try:
            parent_pid = int(parent_pid_raw)
        except ValueError:
            parent_pid = 0
        if parent_pid > 0:
            watcher_task = asyncio.create_task(_watch_parent(parent_pid))

    # Crash-recovery sweep, on a worker thread. It walks the show tree
    # to delete atomic-write temp orphans from a prior hard crash, which
    # measured 190 ms — a third of the whole startup — and it ran inside
    # create_app, delaying the uvicorn bind. Nothing needs it to serve a
    # request: the orphans it removes are files no reader looks at.
    recovery_task = asyncio.create_task(asyncio.to_thread(_run_recovery_sync))

    # Fire-and-forget on a worker thread. ``asyncio.to_thread`` cannot
    # actually interrupt the underlying thread, so there's no point trying
    # to cancel; on shutdown we just await whatever is left.
    warmup_task = asyncio.create_task(asyncio.to_thread(_warmup_caches_sync))

    mcp_task: asyncio.Task | None = None
    mcp_mount = getattr(app.state, "mcp_mount", None)
    if mcp_mount is not None:
        mcp_task = asyncio.create_task(_run_mcp_when_requested(app, mcp_mount))

    try:
        yield
    finally:
        try:
            await recovery_task
        except Exception:
            pass
        if mcp_task is not None:
            mcp_task.cancel()
            try:
                await mcp_task
            except (asyncio.CancelledError, Exception):
                pass
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


def _loopback_hosts(port: int) -> frozenset[str]:
    """Host-header values that identify a same-machine loopback request.

    The CORS allowlist and the CSRF header both fall to a DNS-rebinding
    attack, which makes a malicious page same-origin with the API (so CORS
    never applies and the page can set X-PodCodex freely). The only signal
    that still distinguishes rebinding from a real local request is the Host
    header: a rebound request carries the attacker's hostname, not a loopback
    name. Rejecting non-loopback hosts closes that hole for the default,
    loopback-only install.
    """
    return frozenset(
        {
            f"127.0.0.1:{port}",
            f"localhost:{port}",
            f"[::1]:{port}",
            "127.0.0.1",
            "localhost",
            "[::1]",
        }
    )


class LoopbackGuardMiddleware:
    """Pure-ASGI guard: loopback Host allowlist + API token, one place.

    Host check (anti DNS-rebinding): a rebound request carries the
    attacker's hostname, not a loopback name: the one browser vector CORS
    and the X-PodCodex header do not cover.

    Token check: loopback alone doesn't authenticate the caller; any local
    process or OS user can reach 127.0.0.1. The shared token is required on
    every /api route. Header for normal fetches; query param for
    <img>/<audio>/download URLs and the browser WebSocket, which can't send
    custom headers. Exempt: /api/health (boot probe, runs before the UI has
    the token), OPTIONS (CORS preflights can't carry custom headers; the
    CORSMiddleware sits outside this one and answers them before they get
    here, so the exemption is belt-and-braces for any non-CORS OPTIONS),
    /mcp (outside /api, own access story).

    Written as raw ASGI rather than ``@app.middleware("http")`` because
    BaseHTTPMiddleware never runs on the websocket scope; this way /api/ws
    (and any future websocket route) is covered by construction instead of
    re-implementing the checks per route.
    """

    def __init__(self, app, allowed_hosts: frozenset[str], api_token: str) -> None:
        self.app = app
        self.allowed_hosts = allowed_hosts
        self.api_token = api_token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if headers.get("host", "") not in self.allowed_hosts:
            await self._reject(scope, receive, send, 421, "Bad host header")
            return
        path = scope.get("path", "")
        exempt = path == "/api/health" or (
            scope["type"] == "http" and scope.get("method") == "OPTIONS"
        )
        if path.startswith("/api/") and not exempt:
            supplied = headers.get(TOKEN_HEADER) or QueryParams(
                scope.get("query_string", b"")
            ).get(TOKEN_QUERY_PARAM, "")
            if not secrets.compare_digest(
                supplied.encode("utf-8"), self.api_token.encode("utf-8")
            ):
                await self._reject(
                    scope, receive, send, 401, "Missing or bad API token"
                )
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(scope, receive, send, status: int, detail: str) -> None:
        if scope["type"] == "websocket":
            await WebSocketClose(code=1008)(scope, receive, send)
        else:
            await JSONResponse({"detail": detail}, status_code=status)(
                scope, receive, send
            )


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    # Attached to the import rather than called here: registering reaches
    # IndexStore, and importing that costs ~150 ms of pyarrow + numpy that
    # has no business on the startup path. Deferring it to a worker thread
    # was worse than slow — it races the first request, and
    # ``get_all_collection_info`` caches its result against the collections
    # mtime, so a request that wins the race pins an un-backfilled
    # ``artwork_url`` for the rest of the process. Binding to the import
    # means whoever loads index_store first has the resolver in place by the
    # time they get the module back.
    defer_until_imported("podcodex.rag.index_store", _register_show_folder_resolver)

    app = FastAPI(
        title="PodCodex",
        version=__version__,
        description="Podcast processing pipeline API",
        lifespan=lifespan,
    )
    app.state.mcp_available = _mcp_installed()
    if not app.state.mcp_available:
        logger.warning("MCP extra not installed; /mcp will answer 503")

    # ── Middleware stack ──
    # Added last runs outermost, so these read inside-out: CSRF guard, then
    # the loopback guard, then CORS on the outside. CORS *must* stay outermost
    # so that the guards' 403/401/421 rejections travel back out through it and
    # carry `Access-Control-Allow-Origin`. In the Tauri build the document
    # origin (tauri://localhost) differs from the API origin (127.0.0.1), so a
    # rejection without those headers is blocked by the webview: `fetch` raises
    # a network error instead of resolving with a status, and the first-boot
    # 401 token-refresh retry in `frontend/src/api/client.ts` can never run.

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

    # Host + token enforcement (see LoopbackGuardMiddleware). Outside the CSRF
    # guard: an unauthenticated request is rejected before anything downstream
    # reads it.
    api_token = get_or_create_api_token()
    # Test fixtures read the resolved token from here to build auth headers.
    app.state.api_token = api_token
    app.add_middleware(
        LoopbackGuardMiddleware,
        allowed_hosts=_loopback_hosts(_API_PORT),
        api_token=api_token,
    )

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

    if app.state.mcp_available:
        # Held on app.state so the lifespan can spawn the task that owns it.
        app.state.mcp_mount = _LazyMCPMount()
        app.mount("/mcp", app.state.mcp_mount)

    return app


_DEFAULT_API_PORT = 18811
_API_PORT = int(os.environ.get("PODCODEX_API_PORT", _DEFAULT_API_PORT))

app = create_app()
app.state.api_port = _API_PORT


def main() -> None:
    """Entry point for ``podcodex-api`` script."""
    from podcodex.bootstrap import bootstrap_for_dev

    bootstrap_for_dev()

    # Honour PODCODEX_FFMPEG_EXE in dev too: without this, whisperx /
    # faster-whisper invoke bare "ffmpeg" and only see the system PATH,
    # so the override silently fails for transcription.
    from podcodex.api.server import _wire_native_binaries
    from podcodex.core._ffmpeg import log_ffmpeg_status

    _wire_native_binaries()
    log_ffmpeg_status()

    uvicorn.run(
        "podcodex.api.app:app",
        host="127.0.0.1",
        port=_API_PORT,
        reload=False,
        log_config=None,
        # The API token rides in the query string for <img>/<audio>/download
        # and websocket URLs, which the access log would write out verbatim.
        # Matches `api/server.py` (the bundled sidecar).
        access_log=False,
    )


if __name__ == "__main__":
    main()
