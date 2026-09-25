"""Minimal on-disk episodes for tests (explicit import; no root conftest).

The layout is the real one: ``{show}/{stem}.mp3`` for the audio and
``{show}/{stem}/{stem}`` as the version base (``core._utils.episode_base``).
"""

from __future__ import annotations

from pathlib import Path


def make_episode(root: Path, stem: str = "ep") -> tuple[Path, Path]:
    """Create ``root/show`` with one episode; return ``(base, audio_path)``."""
    show = root / "show"
    (show / stem).mkdir(parents=True, exist_ok=True)
    audio = show / f"{stem}.mp3"
    audio.touch()
    return show / stem / stem, audio


def prov(step: str, model: str | None = "m", **params: object) -> dict:
    """A raw-version provenance dict for ``save_version``."""
    return {"step": step, "type": "raw", "model": model, "params": params}
