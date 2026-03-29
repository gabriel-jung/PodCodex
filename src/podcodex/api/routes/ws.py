"""WebSocket endpoint for real-time task progress."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from podcodex.api.tasks import task_manager

router = APIRouter()


@router.websocket("/ws")
async def websocket_progress(ws: WebSocket) -> None:
    await ws.accept()
    await task_manager.register_ws(ws)
    try:
        # Keep connection open — server pushes only
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        task_manager.unregister_ws(ws)
