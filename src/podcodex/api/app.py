"""PodCodex API — FastAPI application factory and entry point."""

from __future__ import annotations

import os

# Prevent multiprocessing/OpenMP deadlocks when PyTorch DataLoaders run
# inside ThreadPoolExecutor threads (used by the task runner).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv


import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from podcodex.api.routes import (
    audio,
    batch,
    bot_access,
    config,
    episodes as episodes_route,
    export,
    filesystem,
    health,
    index,
    integrations,
    mcp_prompts as mcp_prompts_route,
    models,
    correct,
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


# ── Optional MCP (desktop extra) ────────────────────────────────────────
try:
    from podcodex.mcp.server import mcp as _mcp  # type: ignore[import-not-found]
except Exception as exc:
    _mcp = None
    _mcp_import_error: Exception | None = exc
else:
    _mcp_import_error = None


def _make_lifespan(mcp_http):
    """Nest the mounted MCP sub-app's lifespan under FastAPI's."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if mcp_http is not None:
            async with mcp_http.router.lifespan_context(app):
                yield
        else:
            yield

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


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    mcp_http = _mcp.streamable_http_app() if _mcp is not None else None
    _register_show_folder_resolver()
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

    app.include_router(health.router, prefix="/api", tags=["system"])
    app.include_router(config.router, prefix="/api", tags=["config"])
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
    uvicorn.run(
        "podcodex.api.app:app",
        host="127.0.0.1",
        port=_API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
