"""Episode pipeline status: the flag reconcile and the per-step status rules.

What ``/unified``, ``/status`` and the shows list report for each episode,
and the pass that keeps ``pipeline.db`` in line with the version files on
disk (flags promote and demote, translations lists, indexed flags against
LanceDB, stale verified pointers). It used to be private to the shows route,
so nothing else could ask the same question and the shows list read counts
the reconcile had not touched unless the show page had been opened.

Files decide: a flag says a step is done when a version file for it is on
disk (see ``versions.MISSING_ROW_GRACE_S`` for why rows do not). A listing
that could not be read is "unknown", never "empty".
"""

from __future__ import annotations

import os
import re
from collections.abc import Container
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

from podcodex.core._utils import MTIME_SETTLE_SECONDS, normalize_lang
from podcodex.core.constants import AUDIO_EXTENSIONS
from podcodex.core.llm_failures import FAILURES_FILENAME, rejected_steps
from podcodex.core.pipeline_db import get_pipeline_db
from podcodex.core.source import show_audio_files
from podcodex.core.versions import (
    PIPELINE_STEPS,
    STEP_FLAG,
    clean_translations,
    is_edited,
    seed_show_db,
    step_ext,
)

# ingest imports core.source at module level, never this module, so importing
# it here is not a cycle.
from podcodex.ingest.folder import (
    dir_holds_episode,
    invalidate_scan_cache,
    lance_indexed_stems,
    scan_folder,
)
from podcodex.ingest.show import load_show_meta

if TYPE_CHECKING:
    from podcodex.ingest.show import ShowMeta as _ShowMeta

# What ``_batch_transcribe_from_subs`` can actually consume: the cached
# ``{stem}.subtitles.{lang}.vtt`` that ``youtube.py`` and ``batch.py`` write.
_BATCH_SUBS_RE = re.compile(r"\.subtitles\.[^.]+\.vtt$", re.IGNORECASE)


class StatusContext(NamedTuple):
    """Per-request state shared by every episode's status build."""

    status_map: dict[str, dict]
    seg_counts: dict[str, int]
    stems_with_speaker_map: Container[str]
    local_audio: dict[str, Path]
    episode_files: dict[str, list[str]]
    episode_dirs: set[str]
    # Stems whose llm_failures.json is worth reading: the file shows in the
    # cached listing, or the walk was incomplete and the listing can't be
    # trusted (read directly rather than wrongly hiding failures).
    llm_failure_stems: Container[str]
    effective: dict


class Reconciled(NamedTuple):
    """What ``reconcile_show_status`` leaves behind: the corrected rows and
    the listings it read to correct them."""

    status_map: dict[str, dict]
    local_audio: dict[str, Path]
    episode_files: dict[str, list[str]]
    episode_dirs: set[str]
    llm_failure_stems: set[str]


def load_status_context(path: Path, *, check_index: bool = True) -> StatusContext:
    """Gather everything the status half of an episode payload needs.

    Shared by ``/unified`` and ``/status`` so the two can never disagree about
    a flag. Runs ``reconcile_show_status`` first, which must happen on the
    polled endpoint too or a step finishing mid-batch would not surface until
    the next heavy fetch.
    """
    from podcodex.core.app_config import PipelineAppDefaults, load_config

    rec = reconcile_show_status(path, check_index=check_index)
    app_defaults = (
        load_config().pipeline_defaults or PipelineAppDefaults()
    ).status_defaults()
    db = get_pipeline_db(path)
    return StatusContext(
        status_map=rec.status_map,
        seg_counts=db.latest_segment_counts("transcript"),
        stems_with_speaker_map=db.stems_with_step("speaker_map"),
        local_audio=rec.local_audio,
        episode_files=rec.episode_files,
        episode_dirs=rec.episode_dirs,
        llm_failure_stems=rec.llm_failure_stems,
        effective=_resolve_defaults(app_defaults, load_show_meta(path)),
    )


def reconcile_show_status(path: Path, *, check_index: bool = True) -> Reconciled:
    """Bring ``pipeline.db`` in line with the files (and index) of one show.

    Bootstraps an empty DB from a scan, adds rows for episodes that appeared
    since, and reconciles the step flags, translations lists, indexed flags
    and verified pointers. ``check_index=False`` (the shows list, one call per
    show) skips the LanceDB read and keeps the stored indexed flags, the same
    as an unreadable index does.
    """
    # None when the index could not be read (or was not asked): "unknown"
    # must not demote the stored flags the way "nothing indexed" would.
    lance_indexed = lance_indexed_stems(path) if check_index else None
    indexed_seed = lance_indexed if lance_indexed is not None else set()

    db = get_pipeline_db(path)
    if db.episode_count() == 0:
        episodes = scan_folder(path, indexed_stems=indexed_seed)
        if episodes:
            seed_show_db(path, episodes)

    status_map: dict[str, dict] = {row["stem"]: row for row in db.all_episodes()}

    local_audio = {stem: files[0] for stem, files in show_audio_files(path).items()}
    episode_files, episode_dirs, scan_incomplete = _scan_episode_files(
        path, local_audio
    )
    # An unreadable show folder makes every stem's listing unknown; treating
    # it as empty would demote every flag and wipe every translations list.
    incomplete = set(status_map) if scan_incomplete is None else scan_incomplete

    # Heal rows for episodes that appeared on disk after the initial populate
    # (standalone-file import, bundle import, files copied in by hand). The DB
    # only bootstraps from a scan while it is empty, so without this pass a
    # later arrival never gets a row and /unified never lists it. Root audio
    # always qualifies; a bare directory only counts as an episode under the
    # scanner's own `dir_holds_episode` rule, so stray dirs can't trigger a
    # rescan on every poll (they cost one scandir per poll and nothing more).
    missing = set(local_audio) - status_map.keys()
    for name in episode_dirs - status_map.keys() - missing:
        try:
            names = set(os.listdir(path / name))
        except OSError:
            continue
        if dir_holds_episode(names):
            missing.add(name)
    if missing:
        # The stale-scan guard: these stems were found by uncached scandirs,
        # so a cached scan_folder result that misses them must be dropped.
        invalidate_scan_cache(path)
        new_eps = [
            ep
            for ep in scan_folder(path, indexed_stems=indexed_seed)
            if ep.stem not in status_map
        ]
        # Same order as the bootstrap: version index first (idempotent,
        # registers only files without rows), then the rows.
        seed_show_db(path, new_eps)
        if new_eps:
            status_map = {row["stem"]: row for row in db.all_episodes()}

    if lance_indexed is not None:
        indexed_updates: dict[str, bool] = {}
        for stem, row in status_map.items():
            truth = stem in lance_indexed
            if bool(row.get("indexed", False)) != truth:
                indexed_updates[stem] = truth
                row["indexed"] = truth
        if indexed_updates:
            db.mark_indexed_bulk(indexed_updates)
    # Stems worth reading llm_failures.json for: the file showed up in the
    # listing, or the walk was incomplete so the listing can't be trusted.
    llm_failure_stems = incomplete | {
        stem
        for stem, files in episode_files.items()
        if f"{stem}/{FAILURES_FILENAME}" in files
    }

    # Reconcile the per-step flags: an episode is transcribed / corrected /
    # synthesized when a version file for that step is on disk. Both
    # directions matter. Without the promote, the overview StageCard stays
    # "not started" for any episode whose first sync predates our first
    # assemble; without the demote, a flag survives content deleted out of
    # band.
    #
    # Files, not rows: a DB bootstrapped from a scan has files and no rows
    # yet, and a row whose file is gone (kept by backfill for its grace
    # period, see versions.MISSING_ROW_GRACE_S) is a version no reader can
    # open. Every read path already goes by the file, so the flag does too.
    # Read from the already-cached file list, so this costs no extra syscalls.
    # A stem whose walk failed has an untrustworthy file list: "no files" there
    # means "could not look", so leave its status alone until a clean scan.
    flag_updates: dict[str, dict[str, object]] = {}
    for step, flag in STEP_FLAG.items():
        ext = step_ext(step)
        for stem, row in status_map.items():
            if stem in incomplete:
                continue
            desired = _has_step_files(episode_files.get(stem, []), stem, step, ext)
            if row.get(flag, False) != desired:
                row[flag] = desired
                flag_updates.setdefault(stem, {})[flag] = desired

    # Same treatment for the translations list, which is the per-language
    # equivalent of those flags. A rebuilt DB restores the language versions
    # but not this list, so a translated episode would report "not started"
    # with its translation sitting right there. Rebuilding it here also drops
    # the pipeline-step names legacy rows leaked into it.
    for stem, row in status_map.items():
        if stem in incomplete:
            continue
        desired_langs = _episode_languages(episode_files.get(stem, []), stem)
        if sorted(clean_translations(row.get("translations") or [])) != desired_langs:
            row["translations"] = desired_langs
            flag_updates.setdefault(stem, {})["translations"] = desired_langs
    db.mark_bulk(flag_updates)

    # Reconcile verified pointers: a pointer whose target version no longer
    # exists (out-of-band file deletion, manual DB edit) is stale and must
    # be cleared so the UI never highlights a missing version.
    verified_pointers = db.verified_pointers()
    if verified_pointers:
        ids_by_step: dict[str, dict[str, set[str]]] = {}
        for step_name in {p["step"] for p in verified_pointers.values()}:
            ids_by_step[step_name] = db.version_ids_by_stem(step_name)
        for stem, ptr in list(verified_pointers.items()):
            step_ids = ids_by_step.get(ptr["step"], {}).get(stem, set())
            if ptr["version_id"] not in step_ids:
                db.clear_verified(stem)
                verified_pointers.pop(stem, None)
                row = status_map.get(stem)
                if row:
                    row["verified"] = None

    return Reconciled(
        status_map=status_map,
        local_audio=local_audio,
        episode_files=episode_files,
        episode_dirs=episode_dirs,
        llm_failure_stems=llm_failure_stems,
    )


def _episode_languages(ep_files: list[str], stem: str) -> list[str]:
    """Translation languages this episode has versions for, sorted.

    A step directory that is not a known pipeline step is a language code
    (`PIPELINE_STEPS` is the single source of truth for that distinction).
    Read from files rather than the DB because a version file is always
    written before its row, so the files are the superset, and because this
    runs per episode where a query would not.
    """
    langs = set()
    prefix = f"{stem}/"
    for f in ep_files:
        if not f.startswith(prefix) or not f.endswith(".json"):
            continue
        rest = f[len(prefix) :]
        head, sep, tail = rest.partition("/")
        if sep and "/" not in tail and head not in PIPELINE_STEPS:
            langs.add(head)
    return sorted(langs)


def _has_step_files(ep_files: list[str], stem: str, step: str, ext: str) -> bool:
    """True when the episode's file list holds a version file for *step*.

    `ep_files` entries are paths relative to the show folder, so a version
    file reads as ``<stem>/<step>/<id><ext>``. Matching only that one level
    keeps this in step with `ingest/folder._step_has_versions`, which globs
    ``<step>/*<ext>`` and feeds the very bootstrap this defends; a nested
    sub-step that ever emits the same extension would otherwise make the two
    disagree about the same episode.
    """
    prefix = f"{stem}/{step}/"
    return any(
        f.startswith(prefix) and f.endswith(ext) and "/" not in f[len(prefix) :]
        for f in ep_files
    )


def build_status_out(
    *,
    stem: str | None,
    audio_path: Path | None,
    output_dir: Path | None,
    st: dict,
    ep_files: list[str],
    ctx: StatusContext,
) -> dict:
    """Build the `EpisodeStatusOut` half of an episode payload."""
    prov = _normalize_provenance(st.get("provenance", {}))
    # Speaker labels resolved by user counts as editing the displayed transcript,
    # even though raw segment text is unchanged.
    if stem and stem in ctx.stems_with_speaker_map:
        tprov = prov.get("transcript")
        prov["transcript"] = {
            **(tprov if isinstance(tprov, dict) else {}),
            "manual_edit": True,
        }
    cleaned_translations = clean_translations(st.get("translations", []))
    # The scan already listed the show folder's subdirectories; a per-episode
    # is_dir() here would re-stat all of them on every poll.
    out_dir_exists = bool(output_dir) and output_dir.name in ctx.episode_dirs
    # Two deliberately different questions, do not collapse them:
    # `subtitle_files` is what the episode panel can hand to the manual
    # reimport (which parses .vtt and .srt alike), while `has_subtitles`
    # gates the *batch* subtitle source — and `_batch_transcribe_from_subs`
    # only ever reads a cached `{stem}.subtitles.{lang}.vtt`, so promising
    # .srt there would select episodes the batch cannot process.
    subtitle_files = [f for f in ep_files if f.lower().endswith((".vtt", ".srt"))]
    # The batch's glob carries a language code, so a hand-uploaded
    # `{stem}.subtitles.vtt` (no code) satisfied the flag without satisfying
    # the run: the episode was selected as a subtitle source and then found
    # nothing to read.
    batch_ready_subs = any(_BATCH_SUBS_RE.search(f) for f in ep_files)
    return {
        "stem": stem,
        "audio_path": str(audio_path) if audio_path else None,
        "output_dir": str(output_dir) if out_dir_exists else None,
        "downloaded": audio_path is not None,
        "transcribed": st.get("transcribed", False),
        "corrected": st.get("corrected", False),
        "indexed": st.get("indexed", False),
        "synthesized": st.get("synthesized", False),
        "has_subtitles": batch_ready_subs,
        "translations": cleaned_translations,
        "segment_count": ctx.seg_counts.get(stem) if stem else None,
        "subtitle_files": subtitle_files,
        "provenance": prov,
        "verified": st.get("verified"),
        # Candidate set computed once per request from the cached listing:
        # the failures file rarely exists, and rejected_steps stats + reads
        # it per episode.
        "llm_failed_steps": (
            rejected_steps(output_dir)
            if out_dir_exists and stem in ctx.llm_failure_stems
            else []
        ),
        **_step_statuses(st, prov, ctx.effective, cleaned_translations),
    }


_PARAM_RENAMES = {"mode": "llm_mode"}


def _normalize_provenance(prov: dict) -> dict:
    """Rename legacy param keys (mode→llm_mode)."""
    out = {}
    for step_key, meta in prov.items():
        if not isinstance(meta, dict):
            out[step_key] = meta
            continue
        params = meta.get("params")
        if isinstance(params, dict):
            params = {_PARAM_RENAMES.get(k, k): v for k, v in params.items()}
            meta = {**meta, "params": params}
        out[step_key] = meta
    return out


def _resolve_defaults(app_defaults: dict, show_meta: _ShowMeta | None) -> dict:
    """Merge app-level defaults with show-level overrides.

    Show-level values override app defaults when explicitly set. Strings
    use `""` as the unset sentinel; `diarize` uses `None`.
    """
    effective = dict(app_defaults)
    # Merge per-mode model dicts: app first, show overrides per-mode entries.
    app_models = dict(effective.get("llm_models_by_mode") or {})
    effective.pop("llm_models_by_mode", None)
    show_models: dict[str, str] = {}
    if show_meta and show_meta.pipeline:
        p = show_meta.pipeline
        if p.model_size:
            effective["model_size"] = p.model_size
        if p.llm_mode:
            effective["llm_mode"] = p.llm_mode
        if p.llm_provider_profile:
            effective["llm_provider_profile"] = p.llm_provider_profile
        if p.llm_key_name:
            effective["llm_key_name"] = p.llm_key_name
        if p.target_lang:
            effective["target_lang"] = p.target_lang
        if p.diarize is not None:
            effective["diarize"] = p.diarize
        if p.llm_batch_minutes is not None and p.llm_batch_minutes > 0:
            effective["llm_batch_minutes"] = p.llm_batch_minutes
        show_models = {k: v for k, v in (p.llm_models_by_mode or {}).items() if v}
    merged_models = {**app_models, **show_models}
    mode = effective.get("llm_mode", "")
    resolved_model = merged_models.get(mode, "") if mode else ""
    if resolved_model:
        effective["llm_model"] = resolved_model
    return effective


def _transcribe_outdated(prov: dict, effective: dict) -> bool:
    """Check if a transcribe step's provenance is outdated relative to effective defaults."""
    params = prov.get("params", {})
    source = params.get("source", "whisper")
    # Imported/uploaded transcripts are not outdated — they weren't auto-generated
    if source not in ("whisper",):
        return False
    if not effective:
        return False
    if effective.get("model_size") and prov.get("model") != effective["model_size"]:
        return True
    if "diarize" in effective and params.get("diarize") != effective["diarize"]:
        return True
    return False


def _llm_outdated(prov: dict, effective: dict) -> bool:
    """Check if an LLM step's provenance is outdated relative to effective defaults."""
    params = prov.get("params", {})
    if effective.get("llm_mode") and params.get("llm_mode") != effective["llm_mode"]:
        return True
    if (
        effective.get("llm_provider_profile")
        and params.get("llm_provider_profile") != effective["llm_provider_profile"]
    ):
        return True
    if effective.get("llm_model") and prov.get("model") != effective["llm_model"]:
        return True
    if (
        effective.get("source_lang")
        and params.get("source_lang") != effective["source_lang"]
    ):
        return True
    return False


def _step_statuses(
    st: dict, provenance: dict, effective: dict, translations: list[str]
) -> dict:
    """Compute per-step status: 'none' | 'outdated' | 'done'.

    Compares the episode's provenance against the effective defaults.
    User-validated versions short-circuit to 'done': re-running would
    discard the edits, so 'outdated' is misleading.

    `translations` is the pre-cleaned languages list (see clean_translations);
    callers pass it through so the scrub runs once per episode, not twice.
    """

    verified = st.get("verified") or {}
    verified_step = verified.get("step") if isinstance(verified, dict) else None

    def _check_transcribe() -> str:
        if not st.get("transcribed", False):
            return "none"
        # Verified pointer is the user's explicit "I'm done with this step"
        # signal; it outranks model drift just like edited content does.
        if verified_step == "transcript":
            return "done"
        prov = provenance.get("transcript")
        if not prov:
            return "done"  # no provenance → legacy, assume done
        if is_edited(prov):
            return "done"
        return "outdated" if _transcribe_outdated(prov, effective) else "done"

    def _check_correct() -> str:
        if not st.get("corrected", False):
            return "none"
        if verified_step == "corrected":
            return "done"
        prov = provenance.get("corrected")
        if not prov or not effective:
            return "done"
        if is_edited(prov):
            return "done"
        return "outdated" if _llm_outdated(prov, effective) else "done"

    def _check_translate() -> str:
        if not translations:
            return "none"
        # Translations are stored under normalize_lang, which also turns
        # spaces into underscores; a bare lower() never matches a multi-word
        # target such as "Brazilian Portuguese".
        target = normalize_lang(effective.get("target_lang", ""))
        if target and target not in translations:
            return "none"
        lang_key = target or (translations[0] if translations else "")
        prov = provenance.get(lang_key)
        if not prov or not effective:
            return "done"
        if is_edited(prov):
            return "done"
        return "outdated" if _llm_outdated(prov, effective) else "done"

    return {
        "transcribe_status": _check_transcribe(),
        "correct_status": _check_correct(),
        "translate_status": _check_translate(),
    }


_INTERESTING_EXTS = AUDIO_EXTENSIONS | {
    ".vtt",
    ".srt",  # subtitles
    ".json",
    ".parquet",  # transcripts / pipeline outputs
}
_SKIP_PREFIXES = (".", "__")
_SKIP_NAMES = {"manifest.json"}


def _walk_episode_dir(
    root: Path, rel_prefix: str
) -> tuple[list[str], list[tuple[str, int]] | None]:
    """Recursively collect interesting files under an episode dir.

    ``rel_prefix`` is the path (relative to the show folder) to prepend to
    each file name, so we skip allocating a Path per entry just to call
    ``relative_to``.

    Returns the file list plus every directory visited paired with its mtime,
    which is what `_scan_episode_files` caches on. The directory list is
    ``None`` when any part of the walk hit an ``OSError``: the file list is
    then incomplete, and caching it would pin a truncated result against
    mtimes that will not change. Since this list also drives the status-flag
    reconcile, a transient EACCES could otherwise demote a step and keep it
    demoted.
    """
    import os

    collected: list[str] = []
    try:
        stamp = os.stat(root).st_mtime_ns
    except OSError:
        return [], None
    visited: list[tuple[str, int]] | None = [(str(root), stamp)]
    try:
        with os.scandir(root) as it:
            for f in it:
                name = f.name
                if name.startswith(_SKIP_PREFIXES):
                    continue
                if f.is_dir(follow_symlinks=False):
                    sub_files, sub_dirs = _walk_episode_dir(
                        Path(f.path), f"{rel_prefix}/{name}"
                    )
                    collected.extend(sub_files)
                    if sub_dirs is None:
                        visited = None
                    elif visited is not None:
                        visited.extend(sub_dirs)
                    continue
                if not f.is_file(follow_symlinks=False) or name in _SKIP_NAMES:
                    continue
                dot = name.rfind(".")
                if dot <= 0 or name[dot:].lower() not in _INTERESTING_EXTS:
                    continue
                collected.append(f"{rel_prefix}/{name}")
    except OSError:
        return collected, None
    return collected, visited


# show folder → stem → (visited dirs with their mtimes, file list). Walking a
# show's episode dirs is the most expensive part of building the episode list
# (~20ms for 269 episodes) and it runs on every request, including the 5s
# status poll. Re-stat'ing the recorded directories instead costs ~0.7ms.
_EPISODE_FILES_CACHE: dict[str, dict[str, tuple[list[tuple[str, int]], list[str]]]] = {}

# A directory whose mtime is younger than this is treated as a cache miss.
# Same reasoning (and same value) as the mtime caches in ingest/rss.py; see
# core/_utils.MTIME_SETTLE_SECONDS.
_MTIME_SETTLE_NS = int(MTIME_SETTLE_SECONDS * 1_000_000_000)


def _dirs_unchanged(visited: list[tuple[str, int]]) -> bool:
    """True when every recorded directory still has its recorded mtime."""
    import os

    for path, stamp in visited:
        try:
            if os.stat(path).st_mtime_ns != stamp:
                return False
        except OSError:
            return False
    return True


def _settled(visited: list[tuple[str, int]], now_ns: int) -> bool:
    """True when every recorded mtime was already old when we recorded it.

    The settle window has to gate *storing* an entry, not trusting one: a
    directory written twice inside one coarse timestamp bucket (FAT32 rounds
    to 2s) keeps the same mtime, so an entry recorded between the two writes
    matches forever and hides the second. Refusing to cache until the mtime
    has stopped moving means anything we do cache cannot have a same-bucket
    write after it.
    """
    return all(now_ns - stamp >= _MTIME_SETTLE_NS for _, stamp in visited)


def _scan_episode_files(
    show_folder: Path, local_audio: dict[str, Path]
) -> tuple[dict[str, list[str]], set[str], set[str] | None]:
    """Scan episode subdirectories for user-facing files.

    Returns ``(files, dirs, incomplete)``: a mapping of stem → list of
    filenames relative to show folder; the set of episode directory names
    seen (including empty ones), so callers don't have to re-stat per
    episode to know a directory exists; and the stems whose walk hit an
    OSError: their file list is partial (better than none for display)
    and must not drive status reconciliation. ``incomplete`` is None when
    the show folder itself could not be listed, meaning every stem is.
    Walks version subdirectories (``transcript/``, ``corrected/``,
    ``speaker_map/``, language folders, etc.) so the Pipeline file list
    surfaces version artifacts alongside legacy flat files.

    Per-episode results are cached against the mtimes of every directory the
    walk touched, so an added or removed file anywhere in the tree is caught:
    adding a file bumps its directory's mtime, and adding a directory bumps
    its parent's. Content edits don't bump anything, which is fine because
    only names are reported.
    """
    import os
    import time

    cached = _EPISODE_FILES_CACHE.get(str(show_folder), {})
    fresh: dict[str, tuple[list[tuple[str, int]], list[str]]] = {}
    now_ns = time.time_ns()

    result: dict[str, list[str]] = {}
    dirs: set[str] = set()
    incomplete: set[str] = set()
    try:
        with os.scandir(show_folder) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                stem = entry.name
                if stem.startswith("."):
                    continue
                dirs.add(stem)
                hit = cached.get(stem)
                if hit is not None and _dirs_unchanged(hit[0]):
                    fresh[stem] = hit
                    files = hit[1]
                else:
                    files, visited = _walk_episode_dir(Path(entry.path), stem)
                    files.sort()
                    if visited is None:
                        # The list is truncated, so it must not be cached and
                        # must not drive status either: reconciling against it
                        # would demote a step and wipe the language list from
                        # a transient EACCES.
                        incomplete.add(stem)
                    elif _settled(visited, now_ns):
                        fresh[stem] = (visited, files)
                # Hand out a copy: the root-audio merge below prepends to these
                # lists, which would otherwise grow the cached entry per call.
                if files:
                    result[stem] = list(files)
    except OSError as exc:
        # The show folder itself could not be listed, so every stem's list is
        # unknown, not empty. None tells the caller to skip reconciling, and
        # the cache is kept for the next clean listing.
        logger.warning("Could not list show folder {}: {!r}", show_folder, exc)
        listed = False
    else:
        listed = True
        # Replacing the show's map (rather than updating it) drops entries
        # for episode directories that no longer exist.
        _EPISODE_FILES_CACHE[str(show_folder)] = fresh

    # Prepend root audio (already discovered by show_audio_files).
    for stem, audio_path in local_audio.items():
        result.setdefault(stem, []).insert(0, audio_path.name)

    return result, dirs, incomplete if listed else None
