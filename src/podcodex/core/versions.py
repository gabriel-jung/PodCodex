"""
podcodex.core.versions — Generation versioning for pipeline outputs.

Archives each pipeline save (transcribe, polish, translate) with full
provenance metadata so users can browse, compare, and restore past
generations.

Storage layout per episode::

    episode/
      .versions/
        transcript.json            # manifest (array of version entries)
        transcript/
          20260331T103000Z_raw.json
        polished.json
        polished/
          ...
        english.json               # translation step = language name
        english/
          ...

Active files ({stem}.transcript.raw.json etc.) are unchanged — the
archive is purely additive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


# ──────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────


@dataclass
class VersionMeta:
    """Provenance metadata for one archived generation."""

    step: str  # e.g. "transcript", "polished", "english"
    type: str  # "raw" or "validated"
    model: str | None = None
    params: dict = field(default_factory=dict)
    manual_edit: bool = False

    # Computed at archive time — not passed by caller
    id: str = ""
    timestamp: str = ""
    content_hash: str = ""
    segment_count: int = 0


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def compute_hash(segments: list[dict]) -> str:
    """Deterministic SHA-256 of segment content."""
    canonical = json.dumps(segments, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _versions_dir(base: Path) -> Path:
    """Return the .versions directory for an episode (next to episode files)."""
    return base.parent / ".versions"


def _manifest_path(base: Path, step: str) -> Path:
    """Return the manifest path for a given step."""
    return _versions_dir(base) / f"{step}.json"


def _step_dir(base: Path, step: str) -> Path:
    """Return the directory holding archived segment files for a step."""
    return _versions_dir(base) / step


def _read_manifest(base: Path, step: str) -> list[dict]:
    """Read the manifest for a step. Returns [] if missing."""
    mp = _manifest_path(base, step)
    if not mp.exists():
        return []
    try:
        data = json.loads(mp.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt manifest {}, starting fresh", mp)
        return []


def _write_manifest(base: Path, step: str, entries: list[dict]) -> None:
    """Write the manifest for a step."""
    mp = _manifest_path(base, step)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def maybe_archive(
    base: Path,
    segments: list[dict],
    provenance: dict | None,
    label: str = "",
) -> None:
    """Archive segments if provenance metadata is provided.

    This is the single entry point for all version archiving.  Call it
    after writing the active file — it silently no-ops when *provenance*
    is ``None`` and logs a warning (without raising) on any error.

    Args:
        base:       The AudioPaths.base path (episode stem path).
        segments:   The segment data to archive.
        provenance: Dict with keys ``step``, ``type``, ``model``, ``params``,
                    ``manual_edit``.  ``None`` → skip archiving.
        label:      Human-readable file label for the warning message.
    """
    if not provenance:
        return
    try:
        meta = VersionMeta(
            step=provenance.get("step", ""),
            type=provenance.get("type", "raw"),
            model=provenance.get("model"),
            params=provenance.get("params", {}),
            manual_edit=provenance.get("manual_edit", False),
        )
        archive_version(base, meta.step, segments, meta)
    except Exception:
        logger.opt(exception=True).warning(
            "Failed to archive version for {}", label or "unknown"
        )


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def archive_version(
    base: Path,
    step: str,
    segments: list[dict],
    meta: VersionMeta,
) -> str:
    """Archive a set of segments with provenance metadata.

    Args:
        base:     The AudioPaths.base path (episode stem without extension).
        step:     Pipeline step name (e.g. "transcript", "polished", "english").
        segments: The segment data to archive.
        meta:     Provenance metadata (model, params, etc.).

    Returns:
        The version id (timestamp + type string).
    """
    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y%m%dT%H%M%SZ")
    version_id = f"{ts_str}_{meta.type}"

    # Fill computed fields
    meta.id = version_id
    meta.timestamp = now.isoformat()
    meta.content_hash = compute_hash(segments)
    meta.segment_count = len(segments)
    meta.step = step

    # Write archived segments
    sdir = _step_dir(base, step)
    sdir.mkdir(parents=True, exist_ok=True)
    seg_path = sdir / f"{version_id}.json"
    seg_path.write_text(
        json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Append to manifest (newest first)
    entries = _read_manifest(base, step)
    entries.insert(0, asdict(meta))
    _write_manifest(base, step, entries)

    logger.debug(
        "Archived version {} for step '{}' ({} segments)",
        version_id,
        step,
        len(segments),
    )
    return version_id


def list_versions(base: Path, step: str) -> list[dict]:
    """List all archived versions for a step (newest first).

    Returns a list of version entry dicts from the manifest.
    Returns [] if no versions exist.
    """
    return _read_manifest(base, step)


def load_version(base: Path, step: str, version_id: str) -> list[dict]:
    """Load archived segments for a specific version.

    Args:
        base:       The AudioPaths.base path.
        step:       Pipeline step name.
        version_id: The version id (e.g. "20260331T103000Z_raw").

    Returns:
        List of segment dicts.

    Raises:
        FileNotFoundError: If the version file doesn't exist.
    """
    seg_path = _step_dir(base, step) / f"{version_id}.json"
    if not seg_path.exists():
        raise FileNotFoundError(f"Version {version_id} not found for step '{step}'")
    return json.loads(seg_path.read_text(encoding="utf-8"))


def version_count(base: Path, step: str) -> int:
    """Return the number of archived versions for a step."""
    return len(_read_manifest(base, step))


def prune_versions(base: Path, step: str, keep: int) -> int:
    """Remove old versions, keeping the newest *keep* entries.

    Args:
        base: The AudioPaths.base path.
        step: Pipeline step name.
        keep: Number of newest versions to retain.

    Returns:
        Number of versions removed.
    """
    entries = _read_manifest(base, step)
    if len(entries) <= keep:
        return 0

    to_remove = entries[keep:]
    sdir = _step_dir(base, step)
    removed = 0
    for entry in to_remove:
        vid = entry.get("id", "")
        seg_path = sdir / f"{vid}.json"
        if seg_path.exists():
            seg_path.unlink()
        removed += 1

    _write_manifest(base, step, entries[:keep])
    logger.info("Pruned {} versions for step '{}', kept {}", removed, step, keep)
    return removed
