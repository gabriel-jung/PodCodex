"""
podcodex.core.versions -- Generation versioning for pipeline outputs.

Every pipeline save (transcribe, polish, translate, manual edit) creates
a new version.  Segment data is stored as JSON files under ``.versions/``;
metadata (index) lives in the ``versions`` table of the show-level
``pipeline.db``.  The DB is the source of truth for lookups — there is
no filesystem fallback.

Storage layout per episode::

    episode/
      .versions/
        transcript/
          20260401T103000Z_raw.json
          20260401T120000Z_validated.json
        polished/
          ...
        english/
          ...

There are no "active" files -- the most recent version by timestamp is
the default.  Users can pick any version from the History dropdown.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass
class VersionMeta:
    """Provenance metadata for one version."""

    step: str  # e.g. "transcript", "polished", "english"
    type: str  # "raw" or "validated"
    model: str | None = None
    params: dict = field(default_factory=dict)
    manual_edit: bool = False
    input_hash: str | None = None  # hash of segments used as input (lineage)

    # Computed at save time -- not passed by caller
    id: str = ""
    timestamp: str = ""
    content_hash: str = ""
    segment_count: int = 0


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def compute_hash(segments: list[dict]) -> str:
    """Deterministic SHA-256 of segment content."""
    canonical = json.dumps(segments, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _versions_dir(base: Path) -> Path:
    """Return the .versions directory for an episode (next to episode files)."""
    return base.parent / ".versions"


def _step_dir(base: Path, step: str) -> Path:
    """Return the directory holding segment files for a step."""
    return _versions_dir(base) / step


def _version_path(base: Path, step: str, version_id: str) -> Path:
    """Return the path to a version's segment JSON file."""
    return _step_dir(base, step) / f"{version_id}.json"


def _get_db(base: Path):
    """Get the PipelineDB for the show containing this episode."""
    from podcodex.core.pipeline_db import get_pipeline_db

    show_dir = base.parent.parent
    return get_pipeline_db(show_dir)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def save_version(
    base: Path,
    step: str,
    segments: list[dict],
    provenance: dict | None,
) -> str:
    """Save segments as a new version.  Single entry point for all saves.

    1. Generate version ID from timestamp + type
    2. Compute content_hash
    3. Write segments JSON to .versions/{step}/{id}.json
    4. INSERT into versions table in pipeline.db
    5. Return version_id

    Args:
        base:       The AudioPaths.base path (episode stem path).
        step:       Pipeline step name ("transcript", "polished", "english", ...).
        segments:   The segment data to save.
        provenance: Dict with keys ``step``, ``type``, ``model``, ``params``,
                    ``manual_edit``, optionally ``input_hash``.
                    ``None`` -> skip (no-op).

    Returns:
        The version id string, or "" if provenance is None.
    """
    if not provenance:
        return ""

    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"
    vtype = provenance.get("type", "raw")
    version_id = f"{ts_str}_{vtype}"

    meta = VersionMeta(
        step=step,
        type=vtype,
        model=provenance.get("model"),
        params=provenance.get("params", {}),
        manual_edit=provenance.get("manual_edit", False),
        input_hash=provenance.get("input_hash"),
        id=version_id,
        timestamp=now.isoformat(),
        content_hash=compute_hash(segments),
        segment_count=len(segments),
    )

    # Write segment JSON file
    sdir = _step_dir(base, step)
    sdir.mkdir(parents=True, exist_ok=True)
    seg_path = sdir / f"{version_id}.json"
    seg_path.write_text(
        json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Insert metadata into DB
    db = _get_db(base)
    db.insert_version(base.name, step, asdict(meta))

    logger.debug(
        "Saved version {} for step '{}' ({} segments)",
        version_id,
        step,
        len(segments),
    )
    return version_id


def load_version(base: Path, step: str, version_id: str) -> list[dict]:
    """Load segments for a specific version.

    Raises:
        FileNotFoundError: If the version file doesn't exist.
    """
    seg_path = _version_path(base, step, version_id)
    if not seg_path.exists():
        raise FileNotFoundError(f"Version {version_id} not found for step '{step}'")
    return json.loads(seg_path.read_text(encoding="utf-8"))


def load_latest(base: Path, step: str) -> list[dict] | None:
    """Load segments from the most recent version of a step.

    Returns None if no version exists in the DB.
    """
    db = _get_db(base)
    meta = db.get_latest_version(base.name, step)
    if not meta:
        return None
    try:
        return load_version(base, step, meta["id"])
    except FileNotFoundError:
        logger.warning("Version file missing for {}/{}", step, meta["id"])
        return None


def list_versions(base: Path, step: str) -> list[dict]:
    """List all versions for a step (newest first).

    Returns list of metadata dicts from the DB.
    """
    db = _get_db(base)
    return db.list_versions(base.name, step)


def version_count(base: Path, step: str) -> int:
    """Return the number of versions for a step."""
    db = _get_db(base)
    return db.version_count(base.name, step)


def has_version(base: Path, step: str) -> bool:
    """Return True if at least one version exists for the given step."""
    return version_count(base, step) > 0


def has_matching_version(base: Path, step: str, params: dict) -> bool:
    """Check if any version exists that was produced with matching params.

    Used by batch pipeline to skip steps already run with the same config.
    Compares the subset of keys present in *params* against each version's
    stored params + model.

    Args:
        base:   AudioPaths.base path.
        step:   Pipeline step name.
        params: Dict of params to match.  Special key ``"model"`` is compared
                against the version's ``model`` field; all other keys are
                compared against the version's ``params`` dict.
    """
    if not params:
        return has_version(base, step)

    try:
        db = _get_db(base)
        versions = db.list_versions(base.name, step)
    except Exception:
        return False

    for v in versions:
        match = True
        for key, val in params.items():
            if key == "model":
                if v.get("model") != val:
                    match = False
                    break
            else:
                if v.get("params", {}).get(key) != val:
                    match = False
                    break
        if match:
            return True
    return False


def delete_version(base: Path, step: str, version_id: str) -> bool:
    """Delete a single version (file + DB row).

    Returns ``True`` if the version was found and deleted.
    """
    seg_path = _version_path(base, step, version_id)
    found = seg_path.exists()
    if found:
        seg_path.unlink()

    try:
        db = _get_db(base)
        count = db.delete_versions(base.name, step, [version_id])
        found = found or count > 0
    except Exception:
        logger.opt(exception=True).warning(
            "Failed to delete version {} from DB", version_id
        )

    if found:
        logger.info("Deleted version {} for step '{}'", version_id, step)
    return found


def prune_versions(base: Path, step: str, keep: int) -> int:
    """Remove old versions, keeping the newest *keep* entries.

    Returns number of versions removed.
    """
    try:
        db = _get_db(base)
        versions = db.list_versions(base.name, step)
    except Exception:
        versions = []

    if len(versions) <= keep:
        return 0

    to_remove = versions[keep:]
    ids = [v["id"] for v in to_remove]

    # Delete files
    sdir = _step_dir(base, step)
    for vid in ids:
        seg_path = sdir / f"{vid}.json"
        if seg_path.exists():
            seg_path.unlink()

    # Delete from DB
    try:
        db = _get_db(base)
        db.delete_versions(base.name, step, ids)
    except Exception:
        logger.opt(exception=True).warning("Failed to prune versions from DB")

    logger.info("Pruned {} versions for step '{}', kept {}", len(ids), step, keep)
    return len(ids)
