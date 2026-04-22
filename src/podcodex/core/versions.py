"""
podcodex.core.versions -- Generation versioning for pipeline outputs.

Every pipeline save (transcribe, correct, translate, manual edit) creates
a new version.  Data is stored as JSON or parquet files in per-step
subdirectories; metadata (index) lives in the ``versions`` table of
the show-level ``pipeline.db``.  The DB is the source of truth for
lookups — there is no filesystem fallback.

Storage layout per episode::

    episode/
      transcript/
        20260401T103000Z_raw.json         # final transcript
        segments/
          20260401T102000Z_raw.parquet    # WhisperX raw output
        diarization/
          20260401T102500Z_raw.parquet    # pyannote speaker timeline
        diarized_segments/
          20260401T102800Z_raw.parquet    # segments with speakers assigned
      corrected/
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

# Steps that store data as parquet files (transcription intermediates).
# These are nested under transcript/ on disk.
PARQUET_STEPS = frozenset({"segments", "diarization", "diarized_segments"})


# ------------------------------------------------------------------
# Data types
# ------------------------------------------------------------------


@dataclass
class VersionMeta:
    """Provenance metadata for one version."""

    step: str  # e.g. "transcript", "corrected", "english"
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


def is_edited(meta: dict | None) -> bool:
    """Return True when a version should be labelled "edited" in the UI.

    Covers both user hand-edits (``manual_edit``) and processed-but-not-raw
    outputs such as clean exports or applied manual-LLM passes
    (``type == "validated"``). Mirrors the frontend ``isEdited`` helper so
    the check cannot drift between surfaces.
    """
    if not meta:
        return False
    return meta.get("type") == "validated" or bool(meta.get("manual_edit"))


def versions_dir(base: Path) -> Path:
    """Return the versions directory for an episode (the episode output dir)."""
    return base.parent


def _step_dir(base: Path, step: str) -> Path:
    """Return the directory holding version files for a step.

    Parquet steps (segments, diarization, diarized_segments) are nested
    under ``transcript/`` since they are sub-steps of transcription.
    """
    root = versions_dir(base)
    if step in PARQUET_STEPS:
        return root / "transcript" / step
    return root / step


def _version_path(base: Path, step: str, version_id: str) -> Path:
    """Return the path to a version file (.json or .parquet)."""
    ext = ".parquet" if step in PARQUET_STEPS else ".json"
    return _step_dir(base, step) / f"{version_id}{ext}"


def _get_db(base: Path):
    """Get the PipelineDB for the show containing this episode."""
    from podcodex.core.pipeline_db import get_pipeline_db

    show_dir = base.parent.parent
    return get_pipeline_db(show_dir)


def backfill_versions(show_dir: Path) -> int:
    """Create version entries for legacy transcript files missing from the DB.

    Scans episode directories for ``*.transcript.json`` or
    ``*.transcript.raw.json`` files that have no corresponding version in
    the DB.  Reuses the existing provenance from the episodes table so
    labels (e.g. "YouTube subtitles") are preserved.

    Returns the number of versions created.
    """
    from podcodex.core.pipeline_db import get_pipeline_db

    db = get_pipeline_db(show_dir)
    count = 0

    # Build provenance lookup from episodes table
    ep_provenance: dict[str, dict] = {}
    for row in db.all_episodes():
        prov = row.get("provenance") or {}
        if isinstance(prov, str):
            try:
                prov = json.loads(prov)
            except Exception:
                prov = {}
        ep_provenance[row["stem"]] = prov

    for ep_dir in sorted(show_dir.iterdir()):
        if not ep_dir.is_dir() or ep_dir.name.startswith("."):
            continue
        stem = ep_dir.name
        # Skip if transcript version already exists
        if db.list_versions(stem, "transcript"):
            continue

        # Look for legacy transcript files
        candidates = [
            ep_dir / f"{stem}.transcript.json",
            ep_dir / f"{stem}.transcript.raw.json",
        ]
        seg_file = next((f for f in candidates if f.exists()), None)
        if not seg_file:
            continue

        try:
            segments = json.loads(seg_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Reuse existing provenance from the episodes table
        existing_prov = ep_provenance.get(stem, {}).get("transcript", {})
        vtype = existing_prov.get(
            "type", "validated" if seg_file.name.endswith(".transcript.json") else "raw"
        )
        base = ep_dir / stem
        save_version(
            base=base,
            step="transcript",
            segments=segments,
            provenance=existing_prov
            or {
                "step": "transcript",
                "type": vtype,
            },
        )
        count += 1
        logger.debug("Backfilled transcript version for {}", stem)

    return count


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
    3. Write segments JSON to {step}/{id}.json
    4. INSERT into versions table in pipeline.db
    5. Return version_id

    Args:
        base:       The AudioPaths.base path (episode stem path).
        step:       Pipeline step name ("transcript", "corrected", "english", ...).
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

    sdir = _step_dir(base, step)
    sdir.mkdir(parents=True, exist_ok=True)
    if step in PARQUET_STEPS:
        from podcodex.core._utils import write_parquet

        write_parquet(sdir / f"{version_id}.parquet", segments)
    else:
        from podcodex.core._utils import write_json

        write_json(sdir / f"{version_id}.json", segments)

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

    Treats a missing file or unreadable payload (truncated, zero-filled
    by an interrupted sync, parquet backend error, etc.) as "not found"
    so callers fall back to older versions. The DB row is preserved —
    unavailability may be transient.

    Raises:
        FileNotFoundError: file missing on disk, or payload unreadable.
    """
    seg_path = _version_path(base, step, version_id)
    if not seg_path.exists():
        raise FileNotFoundError(
            f"Version {version_id} missing on disk for step '{step}'"
        )
    try:
        if step in PARQUET_STEPS:
            from podcodex.core._utils import read_parquet

            return read_parquet(seg_path)
        return json.loads(seg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise FileNotFoundError(
            f"Version {version_id} unreadable for step '{step}': {e}"
        ) from e


def load_latest(base: Path, step: str) -> list[dict] | None:
    """Load segments from the most recent usable version of a step.

    Walks versions newest-first and returns the first one that loads
    cleanly. If the newest file is missing or corrupt (crash, cloud-sync
    stub, user deleted it) the call falls through to the next-newest
    instead of failing.

    Returns None if no version exists or all versions are unreadable.
    """
    db = _get_db(base)
    versions = db.list_versions(base.name, step)
    if not versions:
        return None
    for meta in versions:
        try:
            return load_version(base, step, meta["id"])
        except FileNotFoundError as e:
            logger.warning("Skipping version {}/{}: {}", step, meta["id"], e)
    return None


def get_latest_provenance(base: Path, step: str) -> dict | None:
    """Return the provenance dict of the most recent version, or None."""
    db = _get_db(base)
    meta = db.get_latest_version(base.name, step)
    if not meta:
        return None
    return {
        "model": meta.get("model"),
        "type": meta.get("type"),
        "params": meta.get("params", {}),
        "manual_edit": meta.get("manual_edit", False),
    }


def list_versions(base: Path, step: str) -> list[dict]:
    """List all versions for a step (newest first).

    Returns list of metadata dicts from the DB.
    """
    db = _get_db(base)
    return db.list_versions(base.name, step)


def _backfill_episode(base: Path, db) -> None:
    """Lazily backfill version entries for a single episode if legacy files exist.

    Called when list_all_versions returns empty for an episode that may have been
    transcribed before the versioning DB was introduced.
    """
    stem = base.name
    ep_dir = base.parent

    steps_to_check = [
        (
            "transcript",
            [
                ep_dir / f"{stem}.transcript.json",
                ep_dir / f"{stem}.transcript.raw.json",
            ],
        ),
        (
            "corrected",
            [
                ep_dir / f"{stem}.corrected.json",
            ],
        ),
    ]

    for step, candidates in steps_to_check:
        if db.list_versions(stem, step):
            continue
        seg_file = next((f for f in candidates if f.exists()), None)
        if not seg_file:
            continue
        try:
            segments = json.loads(seg_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        vtype = (
            "validated"
            if step == "transcript" and seg_file.name.endswith(".transcript.json")
            else "raw"
        )
        save_version(
            base=base,
            step=step,
            segments=segments,
            provenance={"step": step, "type": vtype},
        )
        logger.debug("Lazy-backfilled {} version for {}", step, stem)


def list_all_versions(base: Path) -> list[dict]:
    """List all versions across all steps for an episode (newest first).

    If the DB is empty for this episode but legacy transcript files exist on
    disk, performs a one-time lazy backfill before returning.
    """
    db = _get_db(base)
    versions = db.list_all_versions(base.name)
    if not versions:
        _backfill_episode(base, db)
        versions = db.list_all_versions(base.name)
    return versions


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
        _refresh_status_after_delete(base, step)
    return found


def _refresh_status_after_delete(base: Path, step: str) -> None:
    """Clear pipeline_db status flags when no versions remain for a step."""
    try:
        db = _get_db(base)
        stem = base.name
        if db.list_versions(stem, step):
            return

        if step == "transcript":
            for legacy in (
                base.parent / f"{stem}.transcript.json",
                base.parent / f"{stem}.transcript.raw.json",
            ):
                if legacy.exists():
                    legacy.unlink()
                    logger.info("Removed legacy transcript file {}", legacy.name)
            db.mark(stem, transcribed=False)
        elif step == "corrected":
            db.mark(stem, corrected=False)
        else:
            row = db.get_episode(stem)
            if row:
                translations = list(row.get("translations") or [])
                if step in translations:
                    translations.remove(step)
                    db.mark(stem, translations=translations)
    except Exception:
        logger.opt(exception=True).warning(
            "Failed to refresh status after delete (step={})", step
        )


def _latest_content_hash(base: Path, step: str) -> str | None:
    """Return the content_hash of the newest version for a step, or None."""
    meta = _get_db(base).get_latest_version(base.name, step)
    return meta["content_hash"] if meta else None


def save_speaker_map_version(base: Path, mapping: dict[str, str]) -> str:
    """Save a speaker map as a versioned ``speaker_map`` entry.

    The map is encoded as a sorted list of ``{"id", "name"}`` dicts to fit
    the ``save_version`` segment-list schema. Lineage to the diarization
    that produced the SPEAKER_XX IDs is tracked via ``input_hash`` (the
    latest diarization version's ``content_hash``, or ``None`` if no
    diarization version exists yet).

    Only one speaker_map version is retained per episode — prior versions
    are pruned after the new one is safely saved.
    """
    entries = [{"id": k, "name": v} for k, v in sorted(mapping.items())]
    vid = save_version(
        base=base,
        step="speaker_map",
        segments=entries,
        provenance={
            "step": "speaker_map",
            "type": "validated",
            "manual_edit": True,
            "input_hash": _latest_content_hash(base, "diarization"),
        },
    )
    prune_versions(base, "speaker_map", keep=1)
    return vid


def load_latest_speaker_map(base: Path) -> dict[str, str]:
    """Load the latest speaker map if it still matches the current diarization.

    Returns an empty dict if no speaker_map version exists, or if the
    stored ``input_hash`` does not match the latest diarization's
    ``content_hash`` (indicating the map is stale after re-diarization).
    """
    latest = _get_db(base).get_latest_version(base.name, "speaker_map")
    if not latest:
        return {}
    if latest.get("input_hash") != _latest_content_hash(base, "diarization"):
        return {}

    try:
        entries = load_version(base, "speaker_map", latest["id"])
    except FileNotFoundError:
        return {}
    return {e["id"]: e["name"] for e in entries}


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
    for vid in ids:
        seg_path = _version_path(base, step, vid)
        if seg_path.exists():
            seg_path.unlink()

    # Delete from DB
    try:
        db.delete_versions(base.name, step, ids)
    except Exception:
        logger.opt(exception=True).warning("Failed to prune versions from DB")

    logger.info("Pruned {} versions for step '{}', kept {}", len(ids), step, keep)
    return len(ids)
