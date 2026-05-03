"""Health & system routes — capabilities, install/remove extras."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from podcodex.api.routes._helpers import submit_task

router = APIRouter(tags=["system"])


def _has(package: str) -> bool:
    """Check if a Python package is importable."""
    return importlib.util.find_spec(package) is not None


# Map each capability to the package that proves it's installed.
_CAPABILITY_CHECKS: dict[str, str] = {
    "whisperx": "whisperx",
    "soundfile": "soundfile",
    "ollama": "ollama",
    "openai": "openai",
    "tts": "qwen_tts",
    "bot": "discord",
    "torch": "torch",
    "embeddings": "sentence_transformers",
    "lancedb": "lancedb",
    "yt_dlp": "yt_dlp",
}

# Map installable extra names to their description.
INSTALLABLE_EXTRAS: dict[str, str] = {
    "pipeline": "Transcription, TTS, LLM correct/translate (whisperx, soundfile, ollama, etc.)",
    "rag": "Indexing & semantic search (torch, sentence-transformers, etc.)",
    "bot": "Discord bot integration",
    "youtube": "YouTube channel/playlist ingest (yt-dlp)",
}

# Map each extra to the capabilities that indicate it's installed.
# Single source of truth — used by list_extras, install, and remove.
_EXTRA_CAPS: dict[str, list[str]] = {
    "pipeline": ["whisperx", "soundfile", "ollama", "tts"],
    "rag": ["torch", "embeddings", "lancedb"],
    "bot": ["bot"],
    "youtube": ["yt_dlp"],
}


def _get_capabilities() -> dict[str, bool]:
    return {name: _has(pkg) for name, pkg in _CAPABILITY_CHECKS.items()}


def _installed_extras(caps: dict[str, bool] | None = None) -> set[str]:
    """Return the set of extras currently installed, based on capabilities."""
    if caps is None:
        caps = _get_capabilities()
    return {
        ext
        for ext, ext_caps in _EXTRA_CAPS.items()
        if all(caps.get(c, False) for c in ext_caps)
    }


@router.get("/health")
async def health() -> dict:
    """Return API status and detected capabilities."""
    from podcodex.core.app_paths import running_in_bundle

    return {
        "status": "ok",
        "capabilities": _get_capabilities(),
        # mode: "bundle" = frozen PyInstaller sidecar (no venv to manage);
        # "dev" = uvicorn from .venv (extras installable via uv sync).
        # Frontend uses this to hide tabs that only make sense in dev.
        "mode": "bundle" if running_in_bundle() else "dev",
    }


@router.get("/system/extras")
async def list_extras() -> dict:
    """List installable extras and their install status."""
    caps = _get_capabilities()
    return {
        "extras": {
            name: {
                "description": desc,
                "installed": all(caps.get(c, False) for c in _EXTRA_CAPS.get(name, [])),
                "capabilities": _EXTRA_CAPS.get(name, []),
            }
            for name, desc in INSTALLABLE_EXTRAS.items()
        },
        "capabilities": caps,
    }


@router.post("/system/free-vram")
async def free_vram_endpoint() -> dict:
    """Flush GPU VRAM — call before heavy pipeline steps if memory is tight."""
    from podcodex.core._utils import free_vram
    from podcodex.core.device import cuda_available

    free_vram()
    if cuda_available():
        import torch

        mem = torch.cuda.mem_get_info()
        return {
            "freed": True,
            "free_mb": mem[0] // (1024 * 1024),
            "total_mb": mem[1] // (1024 * 1024),
        }
    return {"freed": True}


@router.get("/system/device")
async def get_device_info() -> dict:
    """Return resolved device, dtype, GPU name, compute capability, env override."""
    from podcodex.core.device import device_info

    return device_info()


@router.get("/tasks/active")
async def get_active_task(
    audio_path: str | None = None,
) -> dict | None:
    """Return the active task for an audio path, if any."""
    from podcodex.api.tasks import task_manager

    if not audio_path:
        return None
    info = task_manager.get_active(audio_path)
    if not info:
        return None
    resp: dict = {
        "task_id": info.task_id,
        "status": info.status,
        "progress": info.progress,
        "message": info.message,
    }
    if info.steps:
        resp["steps"] = info.steps
    if info.log:
        resp["log"] = info.log
    return resp


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict | None:
    """Return current state of a task by ID (any status, until cleanup)."""
    from podcodex.api.tasks import task_manager

    info = task_manager.get(task_id)
    if not info:
        return None
    resp: dict = {
        "task_id": info.task_id,
        "status": info.status,
        "progress": info.progress,
        "message": info.message,
    }
    if info.steps:
        resp["steps"] = info.steps
    if info.log:
        resp["log"] = info.log
    if info.result is not None:
        resp["result"] = info.result
    if info.error is not None:
        resp["error"] = info.error
    return resp


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict:
    """Cancel a running or pending task."""
    from podcodex.api.tasks import task_manager

    if task_manager.cancel(task_id):
        return {"status": "cancelled", "task_id": task_id}
    raise HTTPException(404, f"No active task with id '{task_id}'")


def _uv_sync_cmd() -> list[str]:
    """Build the base ``uv sync`` command."""
    import shutil

    uv_bin = shutil.which("uv")
    return [uv_bin, "sync"] if uv_bin else [sys.executable, "-m", "uv", "sync"]


def _run_uv_sync(
    extras: set[str],
    progress_cb,
    label: str,
) -> dict:
    """Run ``uv sync --extra ...`` for *extras* and report progress."""
    progress_cb(0.0, f"{label}...")
    cmd = list(_uv_sync_cmd())
    for ext in sorted(extras):
        cmd.extend(["--extra", ext])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        lines.append(line)
        if line:
            progress_cb(0.1, line[:120])
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"uv sync failed (exit {proc.returncode}):\n" + "\n".join(lines[-20:])
        )
    progress_cb(0.95, "Verifying...")
    progress_cb(1.0, "Done! Restart the backend to activate.")
    return {"output": "\n".join(lines)}


class InstallExtraRequest(BaseModel):
    extra: str


@router.post("/system/install-extra")
async def install_extra(req: InstallExtraRequest) -> dict:
    """Install a Python extra via uv sync as a background task."""
    if req.extra not in INSTALLABLE_EXTRAS:
        raise HTTPException(
            400, f"Unknown extra: {req.extra}. Available: {list(INSTALLABLE_EXTRAS)}"
        )

    def run_install(progress_cb, extra_name):
        # Keep all currently installed extras + the new one + desktop (always needed).
        all_extras = _installed_extras() | {extra_name, "desktop"}
        return _run_uv_sync(all_extras, progress_cb, f"Installing '{extra_name}'")

    return submit_task("install", req.extra, run_install, req.extra).model_dump()


@router.post("/system/remove-extra")
async def remove_extra(req: InstallExtraRequest) -> dict:
    """Remove a Python extra via uv sync (omitting it) as a background task."""
    if req.extra not in INSTALLABLE_EXTRAS:
        raise HTTPException(
            400, f"Unknown extra: {req.extra}. Available: {list(INSTALLABLE_EXTRAS)}"
        )
    if req.extra == "desktop":
        raise HTTPException(
            400, "Cannot remove the desktop extra (required by the API)"
        )

    def run_remove(progress_cb, extra_name):
        # Keep all currently installed extras minus the one being removed.
        all_extras = (_installed_extras() | {"desktop"}) - {extra_name}
        return _run_uv_sync(all_extras, progress_cb, f"Removing '{extra_name}'")

    return submit_task("remove", req.extra, run_remove, req.extra).model_dump()
