"""Build `.podcodex` archives.

Pure functions — no argparse, no prompts. CLI and API endpoints wrap these.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from podcodex.bundle.manifest import (
    MANIFEST_FILENAME,
    CollectionEntry,
    ExportResult,
    Manifest,
    Mode,
    ShowEntry,
    is_collection_entry,
    manifest_to_json,
    show_member_prefix,
)
from podcodex.core._utils import atomic_write
from podcodex.core.constants import AUDIO_EXTENSIONS, PARTIAL_DOWNLOAD_SUFFIXES
from podcodex.core.pipeline_db import DB_FILENAME
from podcodex.ingest.show import SHOW_META_FILENAME, show_display

if TYPE_CHECKING:
    from podcodex.rag.index_store import IndexStore

ProgressCallback = Callable[[str, float], None]

# Dotfiles that carry episode metadata (titles, dates, descriptions) and the
# per-show feed cache. Everything else starting with a dot stays out of the
# archive. Kept as literals rather than imported from ingest.rss so export
# does not pull the feed parser onto the bundle path.
_META_DOTFILES = frozenset({".feed_cache.json", ".episode_meta.json"})

# Episode fields in those files that can carry a private feed's token.
_URL_FIELDS = ("audio_url", "artwork_url")


def _sqlite_snapshot(path: Path) -> bytes | None:
    """A consistent copy of a SQLite file, taken through the backup API.

    Copying the raw file can catch a transaction halfway (the per-show
    ``pipeline.db`` runs with a rollback journal), which the importer would
    then open as a corrupt database. Read-only first; a read-only handle
    cannot roll back a hot journal left by a crash, so a second, read-write
    attempt lets SQLite recover it. Returns None when neither works: the
    file is then left out (the importer rebuilds step status from the files
    on disk) rather than shipped torn, without the journal that would fix it.
    """
    import sqlite3

    uri = path.resolve().as_uri()  # percent-encodes "#" and "?" in folder names
    for mode in ("ro", "rw"):
        try:
            src = sqlite3.connect(f"{uri}?mode={mode}", uri=True)
            try:
                dst = sqlite3.connect(":memory:")
                try:
                    src.backup(dst)
                    return dst.serialize()
                finally:
                    dst.close()
            finally:
                src.close()
        except sqlite3.Error as exc:
            last = exc
    logger.warning(f"Could not snapshot {path} ({last}); leaving it out of the bundle")
    return None


def _scrubbed_payload(rel: str, path: Path) -> bytes | None:
    """Archive bytes for a file that carries feed URLs, with secrets stripped.

    Returns None for files that carry none, which are added as they are. A
    private feed keeps its per-subscriber token in its URLs (feed, enclosure
    and image), so a bundle shared with someone else drops their query and
    userinfo unless the exporter opts in (``shareable_url``). The recipient
    pastes their own feed URL to refresh. The search index, which holds its
    own copy of the enclosure URLs, is left out of such a bundle altogether.

    Rewrites the known fields only (``rss_url`` and ``artwork_url`` in
    show.toml, ``audio_url`` and ``artwork_url`` in the meta files): guids
    are episode identity and links in descriptions are content. A file that
    does not parse falls back to stripping every URL in its text, so it
    fails closed.
    """
    import json

    from podcodex.ingest.rss import redact_urls, shareable_url

    def _strip(value):
        return shareable_url(value) if isinstance(value, str) else value

    name = Path(rel).name
    if rel == SHOW_META_FILENAME:
        from podcodex.ingest.show import load_show_meta, render_show_toml

        meta = load_show_meta(path.parent)
        if meta is not None:
            before = (meta.rss_url, meta.artwork_url)
            meta.rss_url, meta.artwork_url = map(_strip, before)
            if (meta.rss_url, meta.artwork_url) == before:
                return None  # nothing to hide: ship the file as written
            return render_show_toml(meta, meta.id).encode("utf-8")
    elif name in _META_DOTFILES:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, (list, dict)):
            for entry in data if isinstance(data, list) else [data]:
                if isinstance(entry, dict):
                    for key in _URL_FIELDS:
                        if key in entry:
                            entry[key] = _strip(entry[key])
            return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    else:
        return None
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    return redact_urls(text).encode("utf-8", errors="surrogateescape")


def _has_rss_feed(show_folder: Path) -> bool:
    """True when the show's episodes come from an RSS feed (YouTube and local
    shows carry no subscriber token)."""
    from podcodex.ingest.show import load_show_meta

    meta = load_show_meta(show_folder)
    return bool(meta and meta.rss_url)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _podcodex_version() -> str:
    try:
        return pkg_version("podcodex")
    except Exception:
        return "unknown"


def _show_collections(
    show_name: str,
    store: IndexStore,
    info: dict[str, dict] | None = None,
    show_id: str = "",
) -> list[CollectionEntry]:
    """LanceDB collections belonging to a show.

    Matched by id when the show has one, and by display name otherwise, so a
    renamed show still exports its own collections instead of nothing.

    ``info`` is the cached output of :meth:`IndexStore.get_all_collection_info`;
    pass it in when iterating over many shows to avoid one query per show.
    """
    if info is None:
        info = store.get_all_collection_info()
    mine = set(store.collections_for_show(show_id, show_label=show_name))
    out: list[CollectionEntry] = []
    for col, meta in sorted(info.items()):
        if col not in mine:
            continue
        out.append(
            CollectionEntry(
                name=col,
                model=meta["model"],
                chunker=meta["chunker"],
                dim=int(meta["dim"]),
                rows=store.count_rows(col),
            )
        )
    return out


def _collection_sidecars(collection: str, entries: list[Path]) -> list[Path]:
    """Pick on-disk entries belonging to ``collection`` from a pre-scanned list.

    Includes ``{col}.lance`` plus its sentinel files (``is_collection_entry``).
    """
    return [e for e in entries if is_collection_entry(e.name, collection)]


def _walk_show_files(folder: Path, *, with_audio: bool) -> list[tuple[str, Path]]:
    """Files in show folder as ``(relative_path_str, abs_path)`` pairs.

    Filters audio extensions when ``with_audio=False``. Skips ``__pycache__``
    and dotfile *directories* at any depth. Dotfiles themselves are skipped
    too, except the two episode-metadata carriers in ``_META_DOTFILES``:
    without them an imported show loses every title, date and description
    (and a feed-backed show loses the cache it refreshes from).
    """
    pairs: list[tuple[str, Path]] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder)
        if any(
            part == "__pycache__" or part.startswith(".") for part in rel.parts[:-1]
        ):
            continue
        if rel.name.startswith(".") and rel.name not in _META_DOTFILES:
            continue
        if path.suffix.lower() in PARTIAL_DOWNLOAD_SUFFIXES or rel.name.endswith(
            ("-journal", "-wal", "-shm")
        ):
            continue
        if not with_audio and path.suffix.lower() in AUDIO_EXTENSIONS:
            continue
        pairs.append((str(rel), path))
    return pairs


def _emit(progress: ProgressCallback | None, msg: str, frac: float) -> None:
    if progress:
        progress(msg, frac)


def _add_manifest(tf: tarfile.TarFile, manifest: Manifest) -> None:
    payload = manifest_to_json(manifest).encode("utf-8")
    info = tarfile.TarInfo(name=MANIFEST_FILENAME)
    info.size = len(payload)
    info.mtime = int(datetime.now(timezone.utc).timestamp())
    info.mode = 0o644
    tf.addfile(info, io.BytesIO(payload))


def _add_show(
    tf: tarfile.TarFile,
    folder_name: str,
    src: Path,
    *,
    with_audio: bool,
    include_feed_credentials: bool,
) -> None:
    arc_prefix = show_member_prefix(folder_name)
    for rel, abs_path in _walk_show_files(src, with_audio=with_audio):
        if rel == DB_FILENAME:
            payload: bytes | None = _sqlite_snapshot(abs_path)
            if payload is None:
                continue
        elif include_feed_credentials:
            payload = None
        else:
            payload = _scrubbed_payload(rel, abs_path)
        if payload is None:
            tf.add(abs_path, arcname=arc_prefix + rel)
            continue
        info = tf.gettarinfo(abs_path, arcname=arc_prefix + rel)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))


def _add_collection(tf: tarfile.TarFile, sidecars: list[Path]) -> None:
    """Add all on-disk entries for one collection under ``lancedb/`` in archive."""
    for sidecar in sidecars:
        if sidecar.is_dir():
            prefix = f"lancedb/{sidecar.name}/"
            for path in sorted(sidecar.rglob("*")):
                if path.is_file():
                    tf.add(path, arcname=prefix + str(path.relative_to(sidecar)))
        else:
            tf.add(sidecar, arcname=f"lancedb/{sidecar.name}")


def _build_archive(
    archive_path: Path,
    manifest: Manifest,
    show_dirs: dict[str, Path],
    collection_dirs: dict[str, list[Path]],
    *,
    with_audio: bool,
    progress: ProgressCallback | None,
    include_feed_credentials: bool = False,
) -> None:
    """Write `.podcodex` (tar.gz) atomically with manifest + payload."""
    total = 1 + len(show_dirs) + len(collection_dirs)
    done = 0

    def _bump(msg: str) -> None:
        nonlocal done
        done += 1
        _emit(progress, msg, done / total)

    def _build(tmp: Path) -> None:
        with tarfile.open(tmp, mode="w:gz") as tf:
            _add_manifest(tf, manifest)
            _bump("manifest")
            for folder_name, src in show_dirs.items():
                _add_show(
                    tf,
                    folder_name,
                    src,
                    with_audio=with_audio,
                    include_feed_credentials=include_feed_credentials,
                )
                _bump(f"show:{folder_name}")
            for col, sidecars in collection_dirs.items():
                _add_collection(tf, sidecars)
                _bump(f"collection:{col}")

    atomic_write(archive_path, _build)


def export_show(
    show_folder: Path,
    output_path: Path,
    *,
    with_audio: bool = False,
    index_only: bool = False,
    include_feed_credentials: bool = False,
    progress: ProgressCallback | None = None,
) -> ExportResult:
    """Export a single show as a `.podcodex` bundle.

    Args:
        show_folder: Source show folder (must contain ``show.toml``).
        output_path: Destination archive path.
        with_audio: Include audio files. Ignored when ``index_only=True``.
        index_only: Skip show folder content; export only LanceDB collections.
        include_feed_credentials: Keep the feed, enclosure and image URLs
            verbatim, and the search index with them. Off by default: a
            private feed's token lives in those URLs, and the index rows
            hold their own copy of the enclosure URLs, so a full bundle of
            an RSS show without credentials carries no collections (the
            recipient re-indexes).
        progress: Optional ``(message, fraction)`` callback.

    Raises:
        FileNotFoundError: ``show_folder`` does not exist.
        ValueError: index-only export with no collections.
    """
    show_folder = Path(show_folder).resolve()
    if not show_folder.is_dir():
        raise FileNotFoundError(f"show folder not found: {show_folder}")

    show_name = show_display(show_folder)
    folder_name = show_folder.name

    # Imported here, not at module scope: index_store pulls pyarrow and
    # numpy (~150 ms), and podcodex.bundle's package __init__ puts this
    # module on the API's startup import path via routes/shows.py.
    from podcodex.rag.index_store import get_index_store

    store = get_index_store()
    from podcodex.ingest.show import ensure_show_id

    export_show_id = ensure_show_id(show_folder)
    collections = _show_collections(show_name, store, show_id=export_show_id)

    mode = Mode.INDEX_ONLY if index_only else Mode.FULL
    if (
        mode == Mode.FULL
        and not include_feed_credentials
        and _has_rss_feed(show_folder)
    ):
        # Index rows carry the raw enclosure URLs, token and all, and cannot
        # be rewritten from here; the recipient re-indexes instead. Shows
        # without an RSS feed have no such token and keep their index.
        collections = []
    if mode == Mode.INDEX_ONLY and not collections:
        raise ValueError(f"show '{show_name}' has no LanceDB collections to export")

    audio_in_bundle = with_audio and not index_only

    manifest = Manifest(
        mode=mode,
        podcodex_version=_podcodex_version(),
        exported_at=_now_iso(),
        shows=[
            ShowEntry(
                id=export_show_id,
                name=show_name,
                folder=folder_name,
                audio_included=audio_in_bundle,
                collections=collections,
            )
        ],
    )
    show_dirs: dict[str, Path] = {} if index_only else {folder_name: show_folder}
    index_entries = list(store.path.iterdir())
    collection_dirs = {
        c.name: _collection_sidecars(c.name, index_entries) for c in collections
    }

    _build_archive(
        Path(output_path),
        manifest,
        show_dirs,
        collection_dirs,
        with_audio=audio_in_bundle,
        progress=progress,
        include_feed_credentials=include_feed_credentials,
    )

    size = Path(output_path).stat().st_size
    return ExportResult(
        output_path=str(output_path),
        size_bytes=size,
        mode=mode,
        shows_exported=1,
        collections_exported=len(collections),
        audio_included=audio_in_bundle,
    )


def export_index(
    show_folders: list[Path],
    output_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> ExportResult:
    """Export multiple shows as an index-only `.podcodex` bundle.

    Args:
        show_folders: List of show folder paths. Caller resolves
            registered names → paths and ``--all`` expansion.
        output_path: Destination archive path.
        progress: Optional ``(message, fraction)`` callback.

    Raises:
        ValueError: empty input or no collections found.
        FileNotFoundError: any folder missing.
    """
    if not show_folders:
        raise ValueError("no shows to export")

    # Imported here, not at module scope: index_store pulls pyarrow and
    # numpy (~150 ms), and podcodex.bundle's package __init__ puts this
    # module on the API's startup import path via routes/shows.py.
    from podcodex.rag.index_store import get_index_store

    store = get_index_store()
    info = store.get_all_collection_info()
    index_entries = list(store.path.iterdir())
    shows: list[ShowEntry] = []
    collection_dirs: dict[str, list[Path]] = {}

    for folder in show_folders:
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"show folder not found: {folder}")
        name = show_display(folder)
        from podcodex.ingest.show import ensure_show_id

        entry_show_id = ensure_show_id(folder)
        cols = _show_collections(name, store, info=info, show_id=entry_show_id)
        if not cols:
            logger.warning(f"show '{name}' has no collections — skipping")
            continue
        shows.append(
            ShowEntry(
                id=entry_show_id,
                name=name,
                folder=folder.name,
                audio_included=False,
                collections=cols,
            )
        )
        for c in cols:
            collection_dirs[c.name] = _collection_sidecars(c.name, index_entries)

    if not shows:
        raise ValueError("no collections to export across requested shows")

    manifest = Manifest(
        mode=Mode.INDEX_ONLY,
        podcodex_version=_podcodex_version(),
        exported_at=_now_iso(),
        shows=shows,
    )

    _build_archive(
        Path(output_path),
        manifest,
        show_dirs={},
        collection_dirs=collection_dirs,
        with_audio=False,
        progress=progress,
    )

    size = Path(output_path).stat().st_size
    return ExportResult(
        output_path=str(output_path),
        size_bytes=size,
        mode=Mode.INDEX_ONLY,
        shows_exported=len(shows),
        collections_exported=len(collection_dirs),
        audio_included=False,
    )
