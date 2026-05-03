"""GPU backend routes — status, download, activate.

Exposes the runtime side of Phase M's "small CPU MSI/DMG + optional CUDA
download" model. See ``podcodex.api.gpu_backend`` for the logic; this
module is just FastAPI plumbing.

Dev-mode behaviour: ``/status`` reports current torch backend and detected
hardware so the Settings UI can render correctly under ``make dev-no-tauri``;
``/download``, ``/activate``, ``/deactivate``, ``/uninstall`` return 400
because they only mean something inside the packaged app.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from podcodex.api import gpu_backend
from podcodex.api.routes._helpers import submit_task

router = APIRouter()

# Shared lock key for the GPU download task. submit_task uses this as the
# audio_path so duplicate POST /download calls return the existing task_id
# rather than racing.
_GPU_LOCK_KEY = "gpu_backend"


class DownloadRequest(BaseModel):
    manifest_url: str | None = None


@router.get("/status")
async def gpu_status() -> dict:
    """Return current GPU backend status — safe in dev or bundle mode."""
    return gpu_backend.status()


@router.post("/download")
async def gpu_download(req: DownloadRequest) -> dict:
    """Kick off the download+install task. Returns the task_id to poll."""
    if not gpu_backend.running_in_bundle():
        raise HTTPException(
            400,
            "GPU backend management is only available in the packaged desktop "
            "app. Your dev venv already uses whatever torch you have installed.",
        )

    manifest_url = (req.manifest_url or "").strip() or gpu_backend.default_manifest_url()

    def _run(progress_cb, url: str) -> dict:
        return gpu_backend.download_and_install(progress_cb, manifest_url=url)

    return submit_task("gpu_download", _GPU_LOCK_KEY, _run, manifest_url).model_dump()


@router.post("/activate")
async def gpu_activate() -> dict:
    """Mark the GPU backend as active. Sidecar respawn is the caller's job."""
    try:
        gpu_backend.activate()
    except gpu_backend.DevModeError as exc:
        raise HTTPException(400, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"activated": True, "restart_required": True}


@router.post("/deactivate")
async def gpu_deactivate() -> dict:
    """Revert to the bundled CPU sidecar on next launch."""
    try:
        gpu_backend.deactivate()
    except gpu_backend.DevModeError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"activated": False, "restart_required": True}


@router.post("/uninstall")
async def gpu_uninstall() -> dict:
    """Remove the on-disk GPU install entirely. Idempotent."""
    try:
        gpu_backend.uninstall()
    except gpu_backend.DevModeError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"installed": False}
