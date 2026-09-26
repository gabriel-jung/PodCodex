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
        20260401T103000123456Z_raw.json   # final transcript
        segments/
          20260401T102000123456Z_raw.parquet  # WhisperX raw output
        diarization/
          20260401T102500123456Z_raw.parquet  # pyannote speaker timeline
        diarized_segments/
          20260401T102800123456Z_raw.parquet  # segments with speakers assigned
      corrected/
        ...
      english/
        ...

There are no "active" files -- the most recent version by timestamp is
the default.  Users can pick any version from the History dropdown.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from podcodex.core.source import SourceRef, SourceVersion

# is_edited lives with the DB it describes; re-exported here, where every
# version reader has always found it.
from podcodex.core.pipeline_db import get_pipeline_db, is_edited  # noqa: F401

# Steps that store data as parquet files (transcription intermediates).
# These are nested under transcript/ on disk.
PARQUET_STEPS = frozenset({"segments", "diarization", "diarized_segments"})

# Steps that store data as audio. Only synth currently; kept as a frozenset
# so future audio steps (e.g. per-segment TTS archive) plug in without
# touching the dispatch in step_ext.
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


def step_ext(step: str) -> str:
    """Return the on-disk file extension for a step's version files.

    Public: callers outside this module need it to recognise version files
    on disk (e.g. the status reconcile in the shows route); the layout
    itself stays owned here alongside ``version_path``.
    """
    if step in WAV_STEPS:
        return ".wav"
    if step in PARQUET_STEPS:
        return ".parquet"
    return ".json"


def version_path(base: Path, step: str, version_id: str) -> Path:
    """Return the canonical on-disk path for a step version file.

    Public entry point for callers that need the destination of a future
    save (assemble_episode) or the resolved location of an existing
    version (existence-checking helpers).

    ``step`` and ``version_id`` reach here straight from request paths and
    query strings, so both are checked as single path components: a ``lang``
    of ``../../../../.config/podcodex`` otherwise resolved to any JSON file
    on disk, which the version routes then read or unlinked.
    """
    from podcodex.core._utils import bad_path_component

    if bad_path_component(step) or bad_path_component(version_id):
        raise ValueError(f"Invalid version path: step={step!r}, id={version_id!r}")
    return _step_dir(base, step) / f"{version_id}{step_ext(step)}"


def _file_exists(base: Path, step: str, version_id: str) -> bool:
    """Whether a version's file is on disk (False for an invalid step or id)."""
    try:
        return version_path(base, step, version_id).exists()
    except (ValueError, OSError):
        return False


def _get_db(base: Path):
    """Get the PipelineDB for the show containing this episode."""
    from podcodex.core._utils import show_dir_of

    return get_pipeline_db(show_dir_of(base))


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
    3. Write the segments to {step}/{id}.json (.parquet for PARQUET_STEPS)
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

    vtype = provenance.get("type", "raw")
    now, version_id = new_version_id(vtype)

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

    if step in WAV_STEPS:
        raise ValueError(f"{step!r} stores audio; use save_synthesize_version")
    # version_path validates *step* as a single path component: a translation
    # language reaches here from request bodies, and "../../x" used to write
    # outside the episode while reads and deletes refused the same value.
    path = version_path(base, step, version_id)
    if step in PARQUET_STEPS:
        from podcodex.core._utils import write_parquet

        write_parquet(path, segments)
    else:
        from podcodex.core._utils import write_json_atomic

        write_json_atomic(path, segments)

    # File first, then row: a row must never point at nothing. If the row
    # cannot be written (another process holding the DB past the busy
    # timeout, a full disk), take the file back out, or the status reconcile
    # reads the step as done while no reader can open the version.
    try:
        _get_db(base).insert_version(base.name, step, asdict(meta))
    except Exception:
        path.unlink(missing_ok=True)
        raise

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


# How long a version row may point at a missing file before backfill drops
# it. A file can be late rather than gone: a synced pipeline.db arriving
# before the version files it indexes. Pruning on the first miss deleted the
# row's provenance and verified pointer for good.
MISSING_ROW_GRACE_S = 24 * 3600


def backfill_versions_from_disk(show_folder: Path) -> int:
    """Register version files on disk that pipeline.db has no row for.

    The DB is the version index: every read path resolves an id through it,
    so files whose rows are gone cannot be opened even though the content is
    right there. That happens whenever the DB is rebuilt from a filesystem
    scan (a first open of a pre-DB library, a DB lost to a sync conflict),
    because `_populate_from_scan` restores the per-episode
    flags but not the version index.

    Provenance cannot be recovered, so rows come back with no model and no
    params. The ``type`` suffix in the filename is preserved, which is what
    decides whether a version reads as edited. Existing rows always win: a
    save landing between the scan and the insert keeps its provenance.

    The pass also runs the other way: rows whose file is gone are dropped,
    once the file has been missing for ``MISSING_ROW_GRACE_S``. The first
    miss only stamps ``missing_since``; a file that comes back clears it.
    Read paths already skip rows without a file, so a row waiting out its
    grace hides nothing. Removing the last row of a step goes through
    ``_refresh_status_after_delete``, the same hook a normal delete uses, so
    the step's flag demotes instead of pointing at nothing.

    Returns the number of rows inserted.
    """
    import time

    from podcodex.core._utils import episode_base

    show_folder = Path(show_folder)
    db = get_pipeline_db(show_folder)
    inserted = 0
    known: dict[str, dict[str, set[str]]] = {}

    def _ids(step: str) -> dict[str, set[str]]:
        if step not in known:
            known[step] = db.version_ids_by_stem(step)
        return known[step]

    def _row(
        stem: str, step: str, path: Path, input_hash: str | None = None
    ) -> tuple[str, str, dict] | None:
        version_id = path.stem
        if version_id in _ids(step).get(stem, set()):
            return None
        if step in WAV_STEPS or step in PARQUET_STEPS:
            # Audio has no segments to hash, and parquet round-trips numpy
            # arrays that compute_hash cannot serialise. Both fall back to the
            # stat hash `save_synthesize_version` already uses. The original
            # sha256 is unrecoverable either way, so speaker_map `input_hash`
            # lineage does not survive a rebuild; only the versions do.
            try:
                content_hash = f"size:{path.stat().st_size}"
            except OSError:
                return None
            segment_count = 0
        else:
            try:
                segments = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Skipping unreadable version file {}", path)
                return None
            if not isinstance(segments, list):
                return None
            content_hash = compute_hash(segments)
            segment_count = len(segments)
        vtype, timestamp = _parse_version_id(version_id, path)
        meta = VersionMeta(
            step=step,
            type=vtype,
            id=version_id,
            timestamp=timestamp,
            content_hash=content_hash,
            segment_count=segment_count,
            input_hash=input_hash,
        )
        return stem, step, asdict(meta)

    def _flush(rows: list[tuple[str, str, dict] | None]) -> None:
        nonlocal inserted
        inserted += db.insert_versions_if_absent([r for r in rows if r])

    walked: list[str] = []
    for ep_dir in sorted(p for p in show_folder.iterdir() if p.is_dir()):
        if ep_dir.name.startswith("."):
            continue
        stem = ep_dir.name
        walked.append(stem)
        step_dirs = sorted(p for p in ep_dir.iterdir() if p.is_dir())
        # Label sources first, committed before any map is registered: a
        # speaker_map's input_hash points at whichever of them is current.
        label_rows = []
        transcript_dir = ep_dir / "transcript"
        if transcript_dir.is_dir():
            for sub in sorted(p for p in transcript_dir.iterdir() if p.is_dir()):
                if sub.name in PARQUET_STEPS:
                    for path in sorted(sub.glob(f"*{step_ext(sub.name)}")):
                        label_rows.append(_row(stem, sub.name, path))
        _flush(label_rows)
        _flush(
            [
                _row(stem, step_dir.name, path)
                for step_dir in step_dirs
                if step_dir.name != "speaker_map"
                for path in sorted(step_dir.glob(f"*{step_ext(step_dir.name)}"))
            ]
        )
        # Speaker maps last, re-bound to the rebuilt label source. Their
        # original input_hash was the source's sha256, which a rebuild cannot
        # reproduce (parquet gets a stat hash), so binding them to the current
        # bucket is what keeps the mapping loadable instead of silently
        # dropping every hand-assigned name.
        map_dir = ep_dir / "speaker_map"
        if map_dir.is_dir():
            bucket = _speaker_map_bucket_hash(episode_base(show_folder, stem))
            _flush(
                [
                    _row(stem, "speaker_map", path, input_hash=bucket)
                    for path in sorted(map_dir.glob(f"*{step_ext('speaker_map')}"))
                ]
            )

    now = time.time()
    pruned = 0
    back: list[tuple[str, str, str]] = []
    first_miss: list[tuple[str, str, str]] = []
    all_rows = db.list_all_versions_by_stem()
    for stem in walked:
        base = episode_base(show_folder, stem)
        expired: dict[str, list[str]] = {}
        for meta in all_rows.get(stem, []):
            step, version_id = meta["step"], meta["id"]
            exists = _file_exists(base, step, version_id)
            since = meta.get("missing_since")
            if exists:
                if since is not None:
                    back.append((stem, step, version_id))
            elif since is None:
                first_miss.append((stem, step, version_id))
            elif now - since >= MISSING_ROW_GRACE_S:
                expired.setdefault(step, []).append(version_id)
        for step, ids in expired.items():
            pruned += db.delete_versions(stem, step, ids)
            _refresh_status_after_delete(base, step)
    db.set_missing_since(back, None)
    db.set_missing_since(first_miss, now)

    if inserted:
        logger.info(
            "Rebuilt {} version rows from disk for {}", inserted, show_folder.name
        )
    if first_miss:
        logger.info(
            "{} version rows point at missing files in {}; dropped if still "
            "missing after {}h",
            len(first_miss),
            show_folder.name,
            MISSING_ROW_GRACE_S // 3600,
        )
    if pruned:
        logger.info(
            "Dropped {} version rows with no file on disk for {}",
            pruned,
            show_folder.name,
        )
    return inserted


def seed_show_db(show_folder: Path, episodes: list) -> None:
    """Bootstrap a show's pipeline.db from a folder scan.

    The version index is rebuilt *before* the episode rows: every read path
    resolves an id through that index, so without it episodes read "done"
    while their transcripts cannot be opened. Doing it first also makes the
    repair resumable, since callers gate on ``episode_count() == 0`` (or on
    stems with no row) and a process killed midway retries the whole thing.
    The single entry for that order (``PipelineDB._populate_from_scan`` is
    private for this reason).
    """
    backfill_versions_from_disk(show_folder)
    if episodes:
        get_pipeline_db(show_folder)._populate_from_scan(episodes)


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
    source: SourceRef | SourceVersion,
    source_chain: list[str] | None = None,
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

    *source* is the version the segments were read from (the pin, the
    translation, or the canonical source), recorded so the audio says which
    text it speaks.
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
            "source_step": source.step,
            "source_version_id": source.version_id,
            **({"source_chain": source_chain} if source_chain else {}),
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


def _live_pointer(base: Path, pointer: dict | None) -> tuple[str, str] | None:
    """The verified pointer's ``(step, version_id)`` if it is still usable."""
    if (
        pointer
        and pointer["step"] in VERIFIABLE_STEPS
        and _file_exists(base, pointer["step"], pointer["version_id"])
    ):
        return pointer["step"], pointer["version_id"]
    return None


def _canonical_ref_from(
    base: Path,
    pointer: dict | None,
    versions_for: Callable[[str], list[dict]],
) -> tuple[str, str] | None:
    """The canonical ladder for one episode.

    *pointer* is the episode's verified pointer (``{step, version_id}``) and
    *versions_for(step)* its newest-first ``corrected`` / ``transcript``
    rows, asked for only when the rung above did not settle it. Shared by
    the single and bulk resolvers so the rules cannot drift.
    """
    ref = _live_pointer(base, pointer)
    if ref:
        return ref
    for step in ("corrected", "transcript"):
        for meta in _live_in_default_order(base, step, versions_for(step)):
            return step, meta["id"]
    return None


def resolve_canonical_ref(base: Path) -> tuple[str, str] | None:
    """The canonical seglist's ``(step, version_id)``: the single definition.

    The user's verified pick (``resolve_verified_source``) always wins; failing
    that, the newest ``corrected`` (honoring the "edited beats freshness"
    ordering), then the newest ``transcript``. Candidates whose file is gone
    are skipped, the way ``load_latest`` walks past them: rows outlive their
    files (a sync still copying, a manual cleanup, a crash between the unlink
    and the row delete) and backfill only drops them after a grace period, so
    one stale row at the head of the order would otherwise hide every older
    readable version and the episode would read as having no transcript at
    all. Only stats the candidates it considers, so batched callers (e.g. the
    speaker roster) can still resolve every episode's ref single-threaded and
    then parallelize the file loads. Returns None when the episode has no
    readable version of either step.
    """
    try:
        db = _get_db(base)
        pointer = db.get_verified(base.name)
    except Exception:
        return None
    return _canonical_ref_from(
        base, pointer, lambda step: db.list_versions(base.name, step)
    )


def resolve_canonical_refs(
    show_dir: Path, stems: list[str]
) -> dict[str, tuple[str, str] | None]:
    """Bulk :func:`resolve_canonical_ref` for many stems, O(1) DB queries.

    Same ladder per stem, resolved from two bulk queries instead of 2-3 per
    stem, so per-show consumers (speaker roster) scale.
    """
    from podcodex.core._utils import episode_base

    db = get_pipeline_db(show_dir)
    verified = db.verified_pointers()
    by_step = db.versions_for_steps(["corrected", "transcript"])
    return {
        stem: _canonical_ref_from(
            episode_base(show_dir, stem),
            verified.get(stem),
            lambda step, stem=stem: by_step.get((stem, step)) or [],
        )
        for stem in stems
    }


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
    meta = _get_db(base).get_version(version_id, stem=base.name)
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


def _live_in_default_order(base: Path, step: str, versions: list[dict]):
    """Yield *versions* (newest-first rows) in default-pick order, skipping
    rows whose file is gone. Files are stat'ed lazily, so a caller that only
    wants the head pays for one stat, not one per version.

    Rows whose file is gone are dropped here rather than in each reader.
    A version can lose its file out of band (a sync, a manual delete, a
    half-finished restore), and every consumer of this order has to agree
    on which version is "current": ``get_latest_provenance`` reading a
    dead head row while ``load_latest`` walked past it to the next one is
    how the status surfaces came to describe a different version than the
    one whose segments were actually loaded.
    """
    if step not in _STRICT_NEWEST_STEPS:
        versions = sort_versions_for_default(versions)
    for meta in versions:
        if _file_exists(base, step, meta["id"]):
            yield meta


def load_latest_with_meta(base: Path, step: str) -> tuple[dict, list[dict]] | None:
    """``(version meta, segments)`` of the best available version of a step.

    Most steps prefer hand-edited / validated versions over more recent
    model output (the "edited beats freshness" rule); the ``transcript``
    step takes the strictly-newest version instead. A missing or corrupt
    file falls through to the next candidate, and the meta returned is the
    one whose segments were loaded, so a caller recording what it consumed
    never pairs one version's id with another's content.
    """
    rows = _get_db(base).list_versions(base.name, step)
    for meta in _live_in_default_order(base, step, rows):
        try:
            return meta, load_version(base, step, meta["id"])
        except FileNotFoundError as e:
            logger.warning("Skipping version {}/{}: {}", step, meta["id"], e)
    return None


def load_latest(base: Path, step: str) -> list[dict] | None:
    """Segments of the best available version of a step, or None.

    See :func:`load_latest_with_meta`.
    """
    found = load_latest_with_meta(base, step)
    return found[1] if found else None


def provenance_of(meta: dict) -> dict:
    """The provenance fields of a version row."""
    return {
        "model": meta.get("model"),
        "type": meta.get("type"),
        "params": meta.get("params", {}),
        "manual_edit": meta.get("manual_edit", False),
    }


def get_version_provenance(
    base: Path, version_id: str, step: str | None = None
) -> dict | None:
    """Provenance of one of this episode's versions, or None when unknown."""
    try:
        meta = _get_db(base).get_version(version_id, stem=base.name, step=step)
    except Exception:
        return None
    return provenance_of(meta) if meta else None


def current_version(base: Path, step: str) -> dict | None:
    """The default-pick version row of *step* (what ``load_latest`` loads).

    Same ordering, including the skip of rows whose file is missing, so
    status surfaces and pipeline defaults agree on which version is
    "current". Reads no segments.
    """
    rows = _get_db(base).list_versions(base.name, step)
    return next(_live_in_default_order(base, step, rows), None)


def get_latest_provenance(base: Path, step: str) -> dict | None:
    """Return the provenance dict of the default-pick version, or None."""
    head = current_version(base, step)
    return provenance_of(head) if head else None


def clean_translations(translations: list[str]) -> list[str]:
    """Strip pipeline-step names from a `translations` list.

    Legacy DBs (pre-PIPELINE_STEPS fix) leaked step names like ``segments`` or
    ``diarization`` into the pipeline_db `translations` array. Filter on read so
    stale rows don't surface as fake languages in the UI; new writes are
    already clean. Lives beside PIPELINE_STEPS: the status routes need it at
    boot, and it used to drag the LLM translate module in with it.
    """
    return [t for t in translations if t not in PIPELINE_STEPS]


def translation_steps(base: Path) -> list[str]:
    """Sorted translation languages with at least one version row."""
    try:
        steps = _get_db(base).list_steps(base.name)
    except Exception:
        logger.opt(exception=True).debug("translation_steps: DB error for {}", base)
        return []
    return clean_translations(steps)


def list_versions(base: Path, step: str) -> list[dict]:
    """List all versions for a step (newest first).

    Returns list of metadata dicts from the DB.
    """
    db = _get_db(base)
    return db.list_versions(base.name, step)


def list_all_versions(base: Path) -> list[dict]:
    """List all versions across all steps for an episode (newest first)."""
    return _get_db(base).list_all_versions(base.name)


def list_all_versions_by_stem(show_dir: Path) -> dict[str, list[dict]]:
    """Every version in a show, grouped by episode stem (newest first)."""
    return get_pipeline_db(show_dir).list_all_versions_by_stem()


def version_count(base: Path, step: str) -> int:
    """Return the number of versions for a step."""
    db = _get_db(base)
    return db.version_count(base.name, step)


def has_version(base: Path, step: str) -> bool:
    """Return True if at least one readable version exists for the step.

    Rows whose file is gone do not count: batch runs skip a step on this, and
    skipping to output nobody can open left the step undone for good.
    """
    return find_matching_version(base, step, {}) is not None


def find_matching_version(
    base: Path, step: str, params: dict, *, current_only: bool = False
) -> str | None:
    """Newest version id produced with matching params, or None.

    Used by batch pipeline to skip steps already run with the same config.
    Compares the subset of keys present in *params* against each version's
    stored params + model; only versions whose file exists count.

    With *current_only*, only the step's default pick (what ``load_latest``
    returns) is considered. The transcription chain needs that: every later
    step loads the newest segments / diarization, so skipping because an
    *older* version matches fed the next step another model's output.

    Args:
        base:   AudioPaths.base path.
        step:   Pipeline step name.
        params: Dict of params to match.  Special key ``"model"`` is compared
                against the version's ``model`` field; all other keys are
                compared against the version's ``params`` dict. A frozenset
                value accepts any of its members. Empty matches any version.
    """
    try:
        versions = _get_db(base).list_versions(base.name, step)
    except Exception:
        return None
    if current_only:
        head = next(_live_in_default_order(base, step, versions), None)
        versions = [head] if head else []

    def accepts(val, actual) -> bool:
        return actual in val if isinstance(val, frozenset) else actual == val

    for v in versions:
        match = all(
            accepts(
                val, v.get("model") if key == "model" else v.get("params", {}).get(key)
            )
            for key, val in params.items()
        )
        if not match:
            continue
        if _file_exists(base, step, v["id"]):
            return v["id"]
    return None


def has_matching_version(
    base: Path, step: str, params: dict, *, current_only: bool = False
) -> bool:
    """Whether :func:`find_matching_version` finds one."""
    return (
        find_matching_version(base, step, params, current_only=current_only) is not None
    )


def delete_version_by_id(base: Path, version_id: str) -> bool:
    """Delete a version when only its id is known. Step is resolved from the DB.

    Mirrors ``load_version_by_id`` for callers (e.g. step-agnostic delete
    routes) that don't need to plumb the step through.
    """
    meta = _get_db(base).get_version(version_id, stem=base.name)
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
            db.get_version(version_id, stem=base.name, step=step)
            if step in SPEAKER_LABEL_SOURCE_STEPS
            else None
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
    Note the status reconcile in the shows route goes by files instead (a
    step counts as done when a version file is on disk), because it also has
    to preserve flags a filesystem-derived bootstrap wrote before any version
    rows existed.

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
        remaining_ids = {v["id"] for v in db.list_versions(stem, step)}
        ptr = db.get_verified(stem)
        if ptr and ptr["step"] == step and ptr["version_id"] not in remaining_ids:
            db.clear_verified(stem)

        # Same for the LLM failures section: its batch indices describe one
        # version, and once that version is gone the batch-fix flow has
        # nothing to patch (it would 404 on every attempt).
        from podcodex.core.llm_failures import clear_step, load_failures

        run_version = (load_failures(base).get(step) or {}).get("version_id")
        if run_version and run_version not in remaining_ids:
            clear_step(base, step)

        if not db.demote_step_if_no_versions(stem, step, STEP_FLAG.get(step)):
            return

        # The step really is empty: drop any recorded LLM batch failures
        # too, which referenced the now-deleted versions.
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
    ref = _live_pointer(base, ptr)
    if ref is None:
        return None
    step, vid = ref
    return step, vid, version_path(base, step, vid)


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


def apply_speaker_map(segments: list[dict], mapping: dict[str, str]) -> list[dict]:
    """*segments* with each original label replaced by its current name.

    The one way a stored seglist is shown with renames made after it was
    generated (translations, synthesis sources); unmapped labels pass through.
    """
    if not mapping:
        return segments
    return [
        {**s, "speaker": mapping.get(s.get("speaker", ""), s.get("speaker", ""))}
        for s in segments
    ]


def compose_speaker_map(
    current: dict[str, str], renames: dict[str, str]
) -> dict[str, str]:
    """Apply *renames* (keyed by displayed label) on top of *current*.

    An entry of *current* whose name is renamed follows the rename, and the
    renamed label itself maps too, since seglists saved after an earlier
    rename carry that label.
    """
    composed = dict(current)
    for source, target in renames.items():
        for orig, name in current.items():
            if name == source:
                composed[orig] = target
        # The displayed label itself also maps: text saved after the earlier
        # rename (an edited transcript, a correction or translation of it)
        # carries the intermediate name, not the original ID.
        composed[source] = target
    return composed


def save_speaker_map_version(base: Path, mapping: dict[str, str]) -> str:
    """Save a speaker map as a versioned ``speaker_map`` entry.

    The map is encoded as a sorted list of ``{"id", "name"}`` dicts to fit
    the ``save_version`` segment-list schema. ``input_hash`` records the
    source of the speaker labels (``diarized_segments`` content_hash for
    the diarized flow, ``segments`` content_hash for subtitle-imported
    transcripts) so each map stays bound to the diarization/import that
    produced its IDs.

    *mapping* holds renames keyed by the label the user currently sees,
    which after an earlier rename is a name, not an original ID. They are
    composed onto the bucket's current map, so every original label keeps
    pointing at its latest name: renaming SPEAKER_00 to Alice, then Alice to
    Alicia, stores ``SPEAKER_00 -> Alicia``, and renaming Bob later leaves
    that entry alone. Consumers only ever map original labels (translations
    and re-exports still carry them) through the single latest map.

    Bucket semantics: one map per ``input_hash``. Saving replaces any
    existing map in the same bucket but leaves maps for other source
    hashes intact, so re-diarize / re-import does not destroy old maps.
    """
    bucket_hash = _speaker_map_bucket_hash(base)
    composed = compose_speaker_map(load_latest_speaker_map(base), mapping)
    entries = [{"id": k, "name": v} for k, v in sorted(composed.items())]
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
