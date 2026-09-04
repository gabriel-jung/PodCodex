"""Resolving an episode's source segments, and the stems on disk beside it.

Domain logic lifted out of ``api/routes/_helpers`` for the same reason as
``core.provenance``: ``rag.index_job`` and ``podcodex-reindex`` need
``build_index_transcript``, and reaching it through the route package pulled
in fastapi, which a bot+rag install does not have.

``_helpers`` re-exports every name here, so route modules are unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

from podcodex.core.constants import AUDIO_EXTENSIONS


# Single source of truth — keeping this aligned with the scanner's set
# avoids "is_downloaded says yes, scanner says no" mismatches that hid
# yt-dlp output behind missing-ffmpeg failures.
AUDIO_EXTS = AUDIO_EXTENSIONS


def scan_show_stems(show_folder: Path) -> tuple[frozenset[str], frozenset[str]]:
    """``(all stems, audio stems)`` from a single directory listing.

    The first set is every stem an episode could already occupy (output
    directories included), which is what :func:`episode_stem` needs to pick
    a free one. The second is the stems that actually have audio on disk,
    which is what "downloaded" means — :func:`is_downloaded` answers the
    same question by relisting and stat-ing the whole show folder, so a
    loop over a 500-episode feed did 500 listings for it.

    Both frozen: ``episode_stem`` memoizes a suffix index keyed on the set.
    """
    import os

    stems: set[str] = set()
    audio: set[str] = set()
    try:
        with os.scandir(show_folder) as it:
            for entry in it:
                name = entry.name
                if entry.is_dir(follow_symlinks=False):
                    if not name.startswith("."):
                        stems.add(name)
                    continue
                # Symlinks followed, matching `episode_audio_files` (the
                # scanner, and the live `is_downloaded` check the download
                # loop still runs). Not following them reported an episode
                # symlinked in from another disk as not downloaded, so the
                # show page offered a Download button that then skipped it.
                if not entry.is_file():
                    continue
                dot = name.rfind(".")
                if dot > 0 and name[dot:].lower() in AUDIO_EXTENSIONS:
                    stems.add(name[:dot])
                    audio.add(name[:dot])
    except OSError:
        pass
    return frozenset(stems), frozenset(audio)


def list_show_stems(show_folder: Path) -> frozenset[str]:
    """One-shot listing of stems on disk in a show folder.

    Pass into :func:`episode_stem` / :func:`rss_episode_to_out` from any
    loop that processes many episodes — without it, each call inside the
    loop would do its own ``os.scandir``. Callers that also need to know
    which stems are downloaded should take both sets from
    :func:`scan_show_stems` instead of scanning twice.
    """
    return scan_show_stems(show_folder)[0]


def is_downloaded(show_folder: Path, stem: str) -> bool:
    """Check if an audio file with the given stem exists in the show folder."""
    from podcodex.core.delete_episode import episode_audio_file

    return episode_audio_file(show_folder, stem) is not None


def apply_broadcast_pattern(pattern: str, title: str) -> int | None:
    """Apply *pattern* to *title*, returning the first capture group as an int.

    Returns ``None`` when the pattern or title is empty, the pattern has no
    capture group, the pattern does not match, or the captured group is not an
    integer. Raises ``re.error`` when the pattern itself is invalid, even for
    an empty title, so callers can always surface a bad regex (e.g. the live
    preview must not silently accept one on a show with no titled episode).
    """
    if not pattern:
        return None
    compiled = re.compile(pattern)  # raises re.error on a bad pattern
    if not title:
        return None
    m = compiled.search(title)
    if not m or not m.lastindex:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_broadcast_number(show_dir: Path, title: str) -> int | None:
    """Apply the show's ``broadcast_number_pattern`` to *title*, if configured.

    Returns the first captured group as an int, or ``None`` when the show has
    no pattern, the title is empty, or the pattern does not match. Invalid
    patterns are swallowed (indexing must never crash on a bad regex).
    """
    if not title:
        return None
    try:
        from podcodex.ingest.show import load_show_meta

        meta = load_show_meta(show_dir)
    except Exception:
        return None
    pattern = meta.broadcast_number_pattern if meta else ""
    try:
        return apply_broadcast_pattern(pattern, title)
    except re.error:
        return None


def _resolve_source_segments(p, source: str) -> tuple[list[dict], str]:
    """Resolve source segments from the version DB.

    Returns (segments, source_label). ``auto`` is ``versions.resolve_canonical_ref``,
    the single definition of the canonical seglist (verified pointer, then
    edited-first ``corrected``, then newest ``transcript``), so index,
    translate, synthesize and the batch runner pick the same version the
    speaker roster does. Raises ValueError if nothing found.
    """
    from podcodex.core._utils import normalize_lang
    from podcodex.core.versions import (
        load_latest,
        load_version,
        resolve_canonical_ref,
    )

    if source == "auto":
        ref = resolve_canonical_ref(p.base)
        if ref is not None:
            step, vid = ref
            try:
                segs = load_version(p.base, step, vid)
            except FileNotFoundError:
                segs = None
            if segs:
                return segs, step
        # The canonical ref is DB-only, so its file can be missing or
        # truncated (a sync conflict). Walk the remaining versions rather
        # than fail while a readable transcript sits on disk.
        for step in ("corrected", "transcript"):
            segs = load_latest(p.base, step)
            if segs:
                return segs, step
        raise ValueError("No transcript found — transcribe first")

    # Explicit steps read from ``p.base`` directly. Going through the audio
    # path would rebuild AudioPaths from ``p.audio_path``, which for an
    # output_dir-only episode (no audio; the synthetic path *is* the base)
    # lands one level too deep and never finds the transcript.
    if source == "transcript":
        segs = load_latest(p.base, "transcript")
        if segs:
            return segs, "transcript"
        raise ValueError("No transcript found — transcribe first")

    if source == "corrected":
        segs = load_latest(p.base, "corrected")
        if segs:
            return segs, "corrected"
        raise ValueError("No corrected segments found")

    # Language code
    lang_norm = normalize_lang(source)
    segs = load_latest(p.base, lang_norm)
    if segs:
        return segs, lang_norm
    raise ValueError(f"No translation found for '{source}'")


def load_best_source(
    audio_path: str | None = None, output_dir: str | None = None
) -> list[dict]:
    """Load the canonical source segments (see ``_resolve_source_segments``).

    Raises ValueError if no source segments are found.
    """
    from podcodex.core._utils import AudioPaths

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)
    segments, _ = _resolve_source_segments(p, "auto")
    return segments


def build_index_transcript(
    audio_path: str | None,
    show_name: str,
    stem: str,
    segments: list[dict] | None = None,
    source: str = "auto",
    output_dir: str | None = None,
) -> dict:
    """Build the transcript dict expected by vectorize_batch.

    If *segments* are provided directly (e.g. from version DB), wraps them.
    Otherwise resolves from the version DB (corrected > transcript fallback).
    Injects RSS metadata (title, pub_date, episode_number) when available.
    """
    from podcodex.core._utils import AudioPaths
    from podcodex.ingest.rss import load_episode_meta

    p = AudioPaths.from_audio(audio_path, output_dir=output_dir)

    if segments is None:
        segments, source = _resolve_source_segments(p, source)

    transcript: dict = {
        "meta": {"show": show_name, "episode": stem, "source": source},
        "segments": segments,
    }

    # Inject RSS metadata
    ep_meta = load_episode_meta(p.base.parent)
    if ep_meta:
        if ep_meta.title:
            transcript["meta"].setdefault("rss_title", ep_meta.title)
        if ep_meta.pub_date:
            transcript["meta"].setdefault("rss_pub_date", ep_meta.pub_date)
        if ep_meta.episode_number is not None:
            transcript["meta"].setdefault("episode_number", ep_meta.episode_number)
        if ep_meta.description:
            transcript["meta"].setdefault("rss_description", ep_meta.description)
        # Media pointers for the Discord bot (index-only): episode artwork, the
        # RSS enclosure to link, and the explicit YouTube video id so the bot
        # can build a timestamped watch link.
        if ep_meta.artwork_url:
            transcript["meta"].setdefault("rss_artwork_url", ep_meta.artwork_url)
        if ep_meta.audio_url:
            transcript["meta"].setdefault("rss_audio_url", ep_meta.audio_url)
        if ep_meta.youtube_id:
            transcript["meta"].setdefault("youtube_id", ep_meta.youtube_id)

    # Broadcast (airing) number: extracted from the episode title using the
    # show's configured regex, when set. Distinct from the per-season
    # episode_number. Absent for shows with no pattern.
    bnum = _extract_broadcast_number(p.show_dir, ep_meta.title if ep_meta else "")
    if bnum is not None:
        transcript["meta"].setdefault("broadcast_number", bnum)

    return transcript
