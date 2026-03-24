"""PodCodex API — FastAPI application factory and entry point."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from podcodex.api.routes import (
    audio,
    config,
    filesystem,
    health,
    rss,
    shows,
    transcribe,
)


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
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(config.router, tags=["config"])
    app.include_router(audio.router)
    app.include_router(filesystem.router)
    app.include_router(shows.router, prefix="/api/shows", tags=["shows"])
    app.include_router(rss.router, prefix="/api/shows", tags=["rss"])
    app.include_router(transcribe.router, prefix="/api/transcribe", tags=["transcribe"])

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
