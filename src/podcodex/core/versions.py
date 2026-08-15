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

# Steps that store data as audio. Only synth currently; kept as a frozenset
# so future audio steps (e.g. per-segment TTS archive) plug in without
# touching the dispatch in _step_ext.
WAV_STEPS = frozenset({"synthesize"})

# Map of stem-level pipeline_db boolean flag per versioned step. Used by
# _refresh_status_after_delete to demote the flag when the last version
# is removed, and by status reconcile in shows.py.
STEP_FLAG = {
    "transcript": "transcribed",
    "corrected": "corrected",
    "synthesize": "synthesized",
}

# Steps whose content_hash can serve as a speaker_map ``input_hash`` (the
# IDs a speaker map references come from one of these). Order is the
# bucket-hash preference: diarized_segments wins when both exist.
SPEAKER_LABEL_SOURCE_STEPS = ("diarized_segments", "segments")

# Canonical pipeline step names. Anything stored under ``versions.step`` that
# is NOT in this set is treated as a translation language code. Single source
# of truth for list_translations() and any read-side scrub: adding a new
# pipeline step means appending one entry here.
PIPELINE_STEPS = frozenset(
    {
        "transcript",
        "corrected",
        "indexed",
        "speaker_map",
        "segments",
        "diarization",
        "diarized_segments",
        "synthesize",
    }
)


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


def _step_ext(step: str) -> str:
    """Return the on-disk file extension for a step's version files."""
    if step in WAV_STEPS:
        return ".wav"
    if step in PARQUET_STEPS:
        return ".parquet"
    return ".json"


def step_ext(step: str) -> str:
    """Public accessor for a step's version-file extension.

    Callers outside this module need it to recognise version files on disk
    (e.g. the status reconcile in the shows route); the layout itself stays
    owned here alongside ``version_path``.
    """
    return _step_ext(step)


def version_path(base: Path, step: str, version_id: str) -> Path:
    """Return the canonical on-disk path for a step version file.

    Public entry point for callers that need the destination of a future
    save (assemble_episode) or the resolved location of an existing
    version (existence-checking helpers).
    """
    return _step_dir(base, step) / f"{version_id}{_step_ext(step)}"


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


def _parse_version_id(version_id: str, fallback: Path) -> tuple[str, str]:
    """Recover ``(type, iso_timestamp)`` from a version filename.

    Ids are ``<%Y%m%dT%H%M%S><micros>Z_<type>`` (see `new_version_id`). Both
    halves are best-effort: an id that predates the format keeps type "raw"
    and borrows the file's mtime, which only affects ordering.
    """
    head, _, vtype = version_id.rpartition("_")
    if not head:
        head, vtype = version_id, "raw"
    try:
        stamp = datetime.strptime(head.rstrip("Z"), "%Y%m%dT%H%M%S%f").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        try:
            stamp = datetime.fromtimestamp(fallback.stat().st_mtime, tz=timezone.utc)
        except OSError:
            stamp = datetime.now(timezone.utc)
    return (vtype or "raw"), stamp.isoformat()


def backfill_versions_from_disk(show_folder: Path) -> int:
    """Register version files on disk that pipeline.db has no row for.

    The DB is the version index: every read path resolves an id through it,
    so files whose rows are gone cannot be opened even though the content is
    right there. That happens whenever the DB is rebuilt from a filesystem
    scan (`POST /resync`, a first open of a pre-DB library, a DB lost to a
    sync conflict), because `populate_from_scan` restores the per-episode
    flags but not the version index.

    Provenance cannot be recovered, so rows come back with no model and no
    params. The ``type`` suffix in the filename is preserved, which is what
    decides whether a version reads as edited.

    Returns the number of rows inserted.
    """
    from podcodex.core._utils import read_parquet
    from podcodex.core.pipeline_db import get_pipeline_db

    show_folder = Path(show_folder)
    db = get_pipeline_db(show_folder)
    inserted = 0
    known: dict[str, dict[str, set[str]]] = {}

    def _ids(step: str) -> dict[str, set[str]]:
        if step not in known:
            known[step] = db.version_ids_by_stem(step)
        return known[step]

    def _register(stem: str, step: str, path: Path) -> None:
        nonlocal inserted
        version_id = path.stem
        if version_id in _ids(step).get(stem, set()):
            return
        if step in WAV_STEPS:
            try:
                content_hash = f"size:{path.stat().st_size}"
            except OSError:
                return
            segment_count = 0
        else:
            try:
                segments = (
                    read_parquet(path)
                    if step in PARQUET_STEPS
                    else json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception:
                logger.warning("Skipping unreadable version file {}", path)
                return
            if not isinstance(segments, list):
                return
            content_hash = compute_hash(segments)
            segment_count = len(segments)
        vtype, timestamp = _parse_version_id(version_id, path)
        db.insert_version(
            stem,
            step,
            asdict(
                VersionMeta(
                    step=step,
                    type=vtype,
                    id=version_id,
                    timestamp=timestamp,
                    content_hash=content_hash,
                    segment_count=segment_count,
                )
            ),
        )
        inserted += 1

    for ep_dir in sorted(p for p in show_folder.iterdir() if p.is_dir()):
        if ep_dir.name.startswith("."):
            continue
        stem = ep_dir.name
        for step_dir in sorted(p for p in ep_dir.iterdir() if p.is_dir()):
            step = step_dir.name
            for path in sorted(step_dir.glob(f"*{_step_ext(step)}")):
                _register(stem, step, path)
            # Parquet sub-steps live one level under transcript/.
            if step == "transcript":
                for sub in sorted(p for p in step_dir.iterdir() if p.is_dir()):
                    if sub.name not in PARQUET_STEPS:
                        continue
                    for path in sorted(sub.glob(f"*{_step_ext(sub.name)}")):
                        _register(stem, sub.name, path)

    if inserted:
        logger.info(
            "Rebuilt {} version rows from disk for {}", inserted, show_folder.name
        )
    return inserted


def new_version_id(vtype: str = "raw") -> tuple[datetime, str]:
    """Return (now, version_id) for a fresh version using the canonical format."""
    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"
    return now, f"{ts_str}_{vtype}"


def save_synthesize_version(
    base: Path,
    audio_file: Path,
    *,
    version_id: str,
    now: datetime,
    strategy: str,
    silence_duration: float,
    source_version_id: str | None,
    language: str,
    model_size: str | None,
    segment_count: int,
    duration_s: float,
) -> str:
    """Register an assembled .wav as a synthesize-step version.

    The audio bytes already live on disk at ``audio_file`` (the route writes
    them via assemble_episode before calling us). Caller pre-allocates
    ``version_id`` / ``now`` so the filename's id and the DB row's id stay
    in sync (the file path encodes the id; computing a second timestamp here
    would silently drift them apart).

    Content hash is `size:<bytes>` rather than a sha256 of the audio — synth
    rows are addressed by version_id and never deduped against other rows,
    so a full-file hash would just burn I/O on every Assemble for no signal.
    """
    stat = audio_file.stat()
    file_size_bytes = stat.st_size
    content_hash = f"size:{file_size_bytes}"
    meta = VersionMeta(
        step="synthesize",
        type="raw",
        model=model_size,
        params={
            "strategy": strategy,
            "silence_duration": silence_duration,
            "source_version_id": source_version_id,
            "language": language,
            "duration_s": round(duration_s, 2),
            "file_size_bytes": file_size_bytes,
        },
        manual_edit=False,
        input_hash=None,
        id=version_id,
        timestamp=now.isoformat(),
        content_hash=content_hash,
        segment_count=segment_count,
    )
    _get_db(base).insert_version(base.name, "synthesize", asdict(meta))
    logger.debug("Saved synthesize version {} ({})", version_id, audio_file.name)
    return version_id


def synthesize_version_path(base: Path, version_id: str) -> Path | None:
    """Resolve a synthesize version's on-disk .wav path.

    Returns ``None`` if no file exists at the canonical version location.
    """
    path = version_path(base, "synthesize", version_id)
    return path if path.is_file() else None


def backfill_version_sizes(base: Path, versions: list[dict]) -> None:
    """Mutate ``versions`` in place to add ``params.file_size_bytes``.

    Stats each backing file once and persists the result back into the DB so
    subsequent reads skip the stat call. Silently leaves params untouched
    when the file is missing or stat fails.
    """
    db = None
    for v in versions:
        params = v.get("params") or {}
        if not isinstance(params, dict):
            continue
        if params.get("file_size_bytes"):
            v["params"] = params
            continue
        step = v.get("step") or ""
        path = version_path(base, step, v["id"])
        if not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        params["file_size_bytes"] = size
        v["params"] = params
        if db is None:
            db = _get_db(base)
        # Persist so future list calls don't re-stat; insert_version uses
        # INSERT OR REPLACE keyed on id, so this just updates the params blob.
        meta = {
            "id": v["id"],
            "timestamp": v["timestamp"],
            "type": v.get("type", "raw"),
            "model": v.get("model"),
            "params": params,
            "manual_edit": v.get("manual_edit", False),
            "content_hash": v.get("content_hash"),
            "segment_count": v.get("segment_count", 0),
            "input_hash": v.get("input_hash"),
        }
        try:
            db.insert_version(base.name, step, meta)
        except Exception:
            logger.opt(exception=True).debug(
                "file_size_bytes backfill DB write failed for {}", v["id"]
            )


def load_version(base: Path, step: str, version_id: str) -> list[dict]:
    """Load segments for a specific version.

    Treats a missing file or unreadable payload (truncated, zero-filled
    by an interrupted sync, parquet backend error, etc.) as "not found"
    so callers fall back to older versions. The DB row is preserved —
    unavailability may be transient.

    Raises:
        FileNotFoundError: file missing on disk, or payload unreadable.
    """
    seg_path = version_path(base, step, version_id)
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


def resolve_canonical_ref(base: Path) -> tuple[str, str] | None:
    """The canonical seglist's ``(step, version_id)``: the single definition.

    The user's verified pick (``resolve_verified_source``) always wins; failing
    that, the newest ``corrected`` (honoring the "edited beats freshness"
    ordering), then the newest ``transcript``. DB-only (it does not read the
    seglist file), so batched callers (e.g. the speaker roster) can resolve
    every episode's ref single-threaded and then parallelize the file loads.
    Returns None when the episode has no version of either step.
    """
    verified = resolve_verified_source(base)
    if verified:
        step, vid, _ = verified
        return step, vid
    for step in ("corrected", "transcript"):
        ordered = _default_ordered_versions(base, step)
        if ordered:
            return step, ordered[0]["id"]
    return None


def resolve_canonical_refs(
    show_dir: Path, stems: list[str]
) -> dict[str, tuple[str, str] | None]:
    """Bulk :func:`resolve_canonical_ref` for many stems, O(1) DB queries.

    Same ladder per stem (verified pointer, then edited-first ``corrected``,
    then newest ``transcript``) but resolved from two bulk queries instead of
    2-3 per stem, so per-show consumers (speaker roster) scale. The verified
    pointer's on-disk check mirrors ``resolve_verified_source`` and only costs
    a stat for episodes that actually have a pointer.
    """
    from podcodex.core.pipeline_db import get_pipeline_db

    db = get_pipeline_db(show_dir)
    verified = db.verified_pointers()
    by_step = db.versions_for_steps(["corrected", "transcript"])
    out: dict[str, tuple[str, str] | None] = {}
    for stem in stems:
        base = show_dir / stem / stem
        ref: tuple[str, str] | None = None
        ptr = verified.get(stem)
        if (
            ptr
            and ptr["step"] in VERIFIABLE_STEPS
            and version_path(base, ptr["step"], ptr["version_id"]).exists()
        ):
            ref = (ptr["step"], ptr["version_id"])
        if ref is None:
            for step in ("corrected", "transcript"):
                versions = by_step.get((stem, step)) or []
                if step not in _STRICT_NEWEST_STEPS:
                    versions = sort_versions_for_default(versions)
                if versions:
                    ref = (step, versions[0]["id"])
                    break
        out[stem] = ref
    return out


def load_canonical_segments(base: Path) -> list[dict] | None:
    """Load an episode's canonical seglist (see :func:`resolve_canonical_ref`).

    Returns None when the episode has no readable transcript version. ``base``
    is the ``{show}/{stem}/{stem}`` version root used everywhere else.
    """
    ref = resolve_canonical_ref(base)
    if not ref:
        return None
    step, vid = ref
    try:
        return load_version(base, step, vid)
    except FileNotFoundError:
        return None


def load_version_by_id(base: Path, version_id: str) -> tuple[list[dict], str] | None:
    """Resolve a version_id to ``(segments, step)``, or None if unknown.

    Centralises the lookup pattern used by index + LLM step entry points
    (single-episode and batch).
    """
    meta = _get_db(base).get_version(version_id)
    if not meta:
        return None
    try:
        return load_version(base, meta["step"], version_id), meta["step"]
    except FileNotFoundError:
        return None


def sort_versions_for_default(versions: list[dict]) -> list[dict]:
    """Sort version list so the default pick (index 0) is edited-first.

    Hand-edited / validated versions outrank model recency for any "what's
    the current best" decision (pipeline source defaults, status pills,
    dropdown defaults). Within each tier (edited / non-edited) the input
    order is preserved, so a newest-first input stays newest-first.
    """
    return sorted(versions, key=lambda v: not is_edited(v))


# The transcript step's default pick is strictly newest: a fresh transcribe
# supersedes an older hand-edited transcript (segment structure can differ
# entirely, e.g. a partial json import). Other steps stay edited-first.
_STRICT_NEWEST_STEPS = {"transcript"}


def _default_ordered_versions(base: Path, step: str) -> list[dict]:
    """Versions ordered so index 0 is the default pick for *step*."""
    versions = _get_db(base).list_versions(base.name, step)
    if step in _STRICT_NEWEST_STEPS:
        return versions  # list_versions is already newest-first
    return sort_versions_for_default(versions)


def load_latest(base: Path, step: str) -> list[dict] | None:
    """Load segments from the best available version of a step.

    Most steps prefer hand-edited / validated versions over more recent
    model output (the "edited beats freshness" rule); the ``transcript``
    step takes the strictly-newest version instead. Walks the resulting
    order and returns the first version that loads cleanly — a missing or
    corrupt file falls through to the next candidate.

    Returns None if no version exists or all versions are unreadable.
    """
    versions = _default_ordered_versions(base, step)
    if not versions:
        return None
    for meta in versions:
        try:
            return load_version(base, step, meta["id"])
        except FileNotFoundError as e:
            logger.warning("Skipping version {}/{}: {}", step, meta["id"], e)
    return None


def get_latest_provenance(base: Path, step: str) -> dict | None:
    """Return the provenance dict of the default-pick version, or None.

    Uses the same ordering as ``load_latest`` so status surfaces and
    pipeline defaults agree on which version is "current".
    """
    versions = _default_ordered_versions(base, step)
    if not versions:
        return None
    meta = versions[0]
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


def list_all_versions(base: Path) -> list[dict]:
    """List all versions across all steps for an episode (newest first)."""
    return _get_db(base).list_all_versions(base.name)


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


def delete_version_by_id(base: Path, version_id: str) -> bool:
    """Delete a version when only its id is known. Step is resolved from the DB.

    Mirrors ``load_version_by_id`` for callers (e.g. step-agnostic delete
    routes) that don't need to plumb the step through.
    """
    meta = _get_db(base).get_version(version_id)
    if not meta:
        return False
    return delete_version(base, meta["step"], version_id)


def delete_version(base: Path, step: str, version_id: str) -> bool:
    """Delete a single version (file + DB row).

    Returns ``True`` if the version was found and deleted.

    Cascades: deleting a ``diarized_segments`` or ``segments`` version also
    drops any ``speaker_map`` versions whose ``input_hash`` matches the
    deleted version's ``content_hash``, since the SPEAKER_XX (or imported
    label) IDs those maps reference no longer exist.
    """
    found = False
    try:
        version_path(base, step, version_id).unlink()
        found = True
    except FileNotFoundError:
        pass

    try:
        db = _get_db(base)
        deleted_meta = (
            db.get_version(version_id) if step in SPEAKER_LABEL_SOURCE_STEPS else None
        )
        count = db.delete_versions(base.name, step, [version_id])
        found = found or count > 0
    except Exception:
        deleted_meta = None
        logger.opt(exception=True).warning(
            "Failed to delete version {} from DB", version_id
        )

    if found:
        logger.info("Deleted version {} for step '{}'", version_id, step)
        _refresh_status_after_delete(base, step)
        if deleted_meta:
            deleted_hash = deleted_meta.get("content_hash")
            if deleted_hash:
                _delete_speaker_maps_where(base, input_hash=deleted_hash)
    return found


def _delete_speaker_maps_where(
    base: Path,
    *,
    input_hash: str | None,
    exclude_id: str | None = None,
) -> None:
    """Delete every ``speaker_map`` version with the given ``input_hash``.

    Single-source pruner used both for bucket-scoped saves (drop siblings
    in the same bucket) and for cascade deletes (drop orphans when their
    label-source version is removed). Routes through ``delete_version`` so
    file + DB stay in sync and the standard status refresh fires.
    """
    try:
        db = _get_db(base)
        targets = [
            v["id"]
            for v in db.list_versions(base.name, "speaker_map")
            if v.get("input_hash") == input_hash and v["id"] != exclude_id
        ]
    except Exception:
        logger.opt(exception=True).warning(
            "Failed to list speaker_map versions for prune (input_hash={})",
            input_hash,
        )
        return
    for vid in targets:
        delete_version(base, "speaker_map", vid)


def _refresh_status_after_delete(base: Path, step: str) -> None:
    """Clear pipeline_db status flags when no versions remain for a step.

    The "no versions left" test reads the DB, not the step directory: a
    normal delete unlinks the file before dropping the row, so both agree.
    Note the status reconcile in the shows route is deliberately broader (a
    step counts as done when it has a row *or* a file on disk), because it
    also has to preserve flags a filesystem-derived bootstrap wrote before
    any version rows existed.

    Deliberately does not pre-check with ``list_versions``: pipeline steps run
    in spawned subprocesses that write to this same DB, so the check and the
    demotion have to be one transaction (``demote_step_if_no_versions``) or a
    version landing in between strands a False flag next to live content.
    """
    try:
        db = _get_db(base)
        stem = base.name
        # The verified pointer can reference any version, including one that
        # was just deleted. Clear the pointer when the target is gone. Safe
        # to do independently: a version arriving concurrently cannot make a
        # deleted pointer target valid again.
        ptr = db.get_verified(stem)
        if ptr and ptr["step"] == step:
            remaining_ids = {v["id"] for v in db.list_versions(stem, step)}
            if ptr["version_id"] not in remaining_ids:
                db.clear_verified(stem)

        if not db.demote_step_if_no_versions(stem, step, STEP_FLAG.get(step)):
            return

        # The step really is empty — drop any recorded LLM batch failures
        # too, which referenced the now-deleted versions.
        from podcodex.core.llm_failures import clear_step

        clear_step(base, step)
    except Exception:
        logger.opt(exception=True).warning(
            "Failed to refresh status after delete (step={})", step
        )


# Steps eligible for verification. Verified pointer must reference one of
# these; downstream consumers (translate / index / synthesize) read from
# whichever one the user marked.
VERIFIABLE_STEPS = frozenset({"transcript", "corrected"})


def resolve_verified_source(base: Path) -> tuple[str, str, Path] | None:
    """Return the verified source ``(step, version_id, file_path)`` or None.

    Single facility used by panel source pickers, RAG indexer, translate,
    synthesize, and bot retrieval to honor the user's canonical pick.
    Returns None when no pointer is set OR the referenced version no longer
    exists (stale pointer; reconcile pass clears it asynchronously).
    """
    try:
        db = _get_db(base)
        ptr = db.get_verified(base.name)
    except Exception:
        return None
    if not ptr:
        return None
    step = ptr["step"]
    vid = ptr["version_id"]
    if step not in VERIFIABLE_STEPS:
        return None
    path = version_path(base, step, vid)
    if not path.exists():
        return None
    return step, vid, path


def _latest_content_hash(base: Path, step: str) -> str | None:
    """Return the content_hash of the newest version for a step, or None."""
    meta = _get_db(base).get_latest_version(base.name, step)
    return meta["content_hash"] if meta else None


def diarized_segments_input_hash(base: Path) -> str | None:
    """Combined lineage hash for the diarized_segments step.

    ``diarized_segments`` is derived from both ``segments`` (WhisperX) and
    ``diarization`` (pyannote). Returns the sha256 of the two source
    content_hashes so the value shape matches other ``input_hash`` fields
    (``sha256:...``). Returns ``None`` if either source is missing.
    """
    seg_h = _latest_content_hash(base, "segments")
    diar_h = _latest_content_hash(base, "diarization")
    if not seg_h or not diar_h:
        return None
    return compute_hash([{"seg": seg_h, "diar": diar_h}])


def diarized_segments_is_fresh(base: Path) -> bool:
    """True when the latest diarized_segments was built from the latest
    segments + diarization.
    """
    expected = diarized_segments_input_hash(base)
    if not expected:
        return False
    latest = _get_db(base).get_latest_version(base.name, "diarized_segments")
    return bool(latest and latest.get("input_hash") == expected)


def _speaker_map_bucket_hash(base: Path) -> str | None:
    """Return the content_hash that defines the current speaker label set.

    Speaker IDs come from ``diarized_segments`` for the diarized pipeline,
    and from ``segments`` for subtitle-imported transcripts (YouTube ``<v>``
    tags). Walks ``SPEAKER_LABEL_SOURCE_STEPS`` in preference order and
    returns the first available content_hash, or ``None`` if neither step
    has a version.
    """
    for step in SPEAKER_LABEL_SOURCE_STEPS:
        h = _latest_content_hash(base, step)
        if h is not None:
            return h
    return None


def save_speaker_map_version(base: Path, mapping: dict[str, str]) -> str:
    """Save a speaker map as a versioned ``speaker_map`` entry.

    The map is encoded as a sorted list of ``{"id", "name"}`` dicts to fit
    the ``save_version`` segment-list schema. ``input_hash`` records the
    source of the speaker labels (``diarized_segments`` content_hash for
    the diarized flow, ``segments`` content_hash for subtitle-imported
    transcripts) so each map stays bound to the diarization/import that
    produced its IDs.

    Bucket semantics: one map per ``input_hash``. Saving replaces any
    existing map in the same bucket but leaves maps for other source
    hashes intact, so re-diarize / re-import does not destroy old maps.
    """
    bucket_hash = _speaker_map_bucket_hash(base)
    entries = [{"id": k, "name": v} for k, v in sorted(mapping.items())]
    vid = save_version(
        base=base,
        step="speaker_map",
        segments=entries,
        provenance={
            "step": "speaker_map",
            "type": "validated",
            "manual_edit": True,
            "input_hash": bucket_hash,
        },
    )
    _delete_speaker_maps_where(base, input_hash=bucket_hash, exclude_id=vid)
    return vid


def load_latest_speaker_map(base: Path) -> dict[str, str]:
    """Load the speaker map bound to the current speaker-label source.

    Walks speaker_map versions newest-first and returns the first one
    whose ``input_hash`` matches the current bucket hash (see
    ``_speaker_map_bucket_hash``). Returns an empty dict when no matching
    map exists, so re-diarization or re-import never silently misapplies
    a stale mapping.
    """
    bucket_hash = _speaker_map_bucket_hash(base)
    if bucket_hash is None:
        return {}
    try:
        versions = _get_db(base).list_versions(base.name, "speaker_map")
    except Exception:
        return {}
    for v in versions:
        if v.get("input_hash") != bucket_hash:
            continue
        try:
            entries = load_version(base, "speaker_map", v["id"])
        except FileNotFoundError:
            continue
        return {e["id"]: e["name"] for e in entries}
    return {}
