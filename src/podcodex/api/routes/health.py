"""Health & system routes — capabilities, install extras."""

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
    "qdrant": "qdrant_client",
    "tts": "qwen_tts",
    "bot": "discord",
    "torch": "torch",
    "embeddings": "sentence_transformers",
    "bm25": "bm25s",
}

# Map installable extra names to their description.
INSTALLABLE_EXTRAS: dict[str, str] = {
    "pipeline": "Transcription, TTS, LLM polish/translate (whisperx, soundfile, ollama, etc.)",
    "rag": "Indexing & semantic search (torch, sentence-transformers, qdrant, etc.)",
    "bot": "Discord bot integration",
}


def _get_capabilities() -> dict[str, bool]:
    return {name: _has(pkg) for name, pkg in _CAPABILITY_CHECKS.items()}


@router.get("/api/health")
async def health() -> dict:
    """Return API status and detected capabilities."""
    return {
        "status": "ok",
        "capabilities": _get_capabilities(),
    }


@router.get("/api/system/extras")
async def list_extras() -> dict:
    """List installable extras and their install status."""
    caps = _get_capabilities()
    # Map extras to their key capabilities
    extra_caps = {
        "pipeline": ["whisperx", "soundfile", "ollama", "tts"],
        "rag": ["torch", "embeddings", "bm25", "qdrant"],
        "bot": ["bot"],
    }
    return {
        "extras": {
            name: {
                "description": desc,
                "installed": all(caps.get(c, False) for c in extra_caps.get(name, [])),
                "capabilities": extra_caps.get(name, []),
            }
            for name, desc in INSTALLABLE_EXTRAS.items()
        },
        "capabilities": caps,
    }


@router.get("/api/tasks/active")
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


class InstallExtraRequest(BaseModel):
    extra: str


@router.post("/api/system/install-extra")
async def install_extra(req: InstallExtraRequest) -> dict:
    """Install a Python extra via uv sync as a background task."""
    if req.extra not in INSTALLABLE_EXTRAS:
        raise HTTPException(
            400, f"Unknown extra: {req.extra}. Available: {list(INSTALLABLE_EXTRAS)}"
        )

    def run_install(progress_cb, extra_name):
        import shutil

        progress_cb(0.0, f"Installing '{extra_name}' extra...")

        # Find uv binary
        uv_bin = shutil.which("uv")
        base_cmd = [uv_bin, "sync"] if uv_bin else [sys.executable, "-m", "uv", "sync"]

        # Detect currently installed extras so we don't lose them.
        # uv sync replaces the environment to match exactly the requested extras,
        # so we must pass ALL desired extras, not just the new one.
        all_extras = {extra_name, "desktop"}  # desktop is always needed (we're the API)
        caps = _get_capabilities()
        extra_cap_map = {
            "pipeline": ["whisperx", "soundfile", "ollama", "tts"],
            "rag": ["torch", "embeddings", "bm25", "qdrant"],
            "bot": ["bot"],
        }
        for ext, ext_caps in extra_cap_map.items():
            if all(caps.get(c, False) for c in ext_caps):
                all_extras.add(ext)

        cmd = list(base_cmd)
        for ext in sorted(all_extras):
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
            # Show the last meaningful line as progress
            if line:
                progress_cb(0.1, line[:120])
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Install failed (exit {proc.returncode}):\n" + "\n".join(lines[-20:])
            )
        progress_cb(0.95, "Verifying installation...")
        progress_cb(1.0, "Done! Restart the backend to activate.")
        return {"output": "\n".join(lines), "extra": extra_name}

    return submit_task("install", req.extra, run_install, req.extra).model_dump()
