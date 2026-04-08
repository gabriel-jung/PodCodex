"""PodCodex API — FastAPI application factory and entry point."""

from __future__ import annotations

from dotenv import load_dotenv


import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from podcodex.api.routes import (
    audio,
    batch,
    config,
    export,
    filesystem,
    health,
    index,
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


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="PodCodex",
        version="0.1.0",
        description="Podcast processing pipeline API",
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
    app.include_router(ws.router, prefix="/api", tags=["ws"])

    app.include_router(batch.router, prefix="/api/batch", tags=["batch"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])
    app.include_router(export.router, prefix="/api/export", tags=["export"])

    return app


app = create_app()


def main() -> None:
    """Entry point for ``podcodex-api`` script."""
    uvicorn.run(
        "podcodex.api.app:app",
        host="127.0.0.1",
        port=18811,
        reload=False,
    )


if __name__ == "__main__":
    main()
