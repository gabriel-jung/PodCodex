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
    manifest_to_json,
    show_member_prefix,
)
from podcodex.core._utils import atomic_write
from podcodex.core.constants import AUDIO_EXTENSIONS
from podcodex.ingest.show import show_display
from podcodex.rag.index_store import get_index_store

if TYPE_CHECKING:
    from podcodex.rag.index_store import IndexStore

ProgressCallback = Callable[[str, float], None]


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
) -> list[CollectionEntry]:
    """LanceDB collections registered to ``show_name`` (display name match).

    ``info`` is the cached output of :meth:`IndexStore.get_all_collection_info`;
    pass it in when iterating over many shows to avoid one query per show.
    """
    if info is None:
        info = store.get_all_collection_info()
    out: list[CollectionEntry] = []
    for col, meta in sorted(info.items()):
        if meta.get("show") != show_name:
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

    Includes ``{col}.lance`` plus any ``{col}.<index_name>`` sidecar. Trailing
    dot prevents prefix collisions with longer collection names.
    """
    return [
        e
        for e in entries
        if e.name == f"{collection}.lance" or e.name.startswith(collection + ".")
    ]


def _walk_show_files(folder: Path, *, with_audio: bool) -> list[tuple[str, Path]]:
    """Files in show folder as ``(relative_path_str, abs_path)`` pairs.

    Filters audio extensions when ``with_audio=False``. Skips ``__pycache__``
    and dotfile directories at any depth.
    """
    pairs: list[tuple[str, Path]] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if any(
            part == "__pycache__" or part.startswith(".")
            for part in path.relative_to(folder).parts
        ):
            continue
        if not with_audio and path.suffix.lower() in AUDIO_EXTENSIONS:
            continue
        pairs.append((str(path.relative_to(folder)), path))
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
    tf: tarfile.TarFile, folder_name: str, src: Path, *, with_audio: bool
) -> None:
    arc_prefix = show_member_prefix(folder_name)
    for rel, abs_path in _walk_show_files(src, with_audio=with_audio):
        tf.add(abs_path, arcname=arc_prefix + rel)


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
                _add_show(tf, folder_name, src, with_audio=with_audio)
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
    progress: ProgressCallback | None = None,
) -> ExportResult:
    """Export a single show as a `.podcodex` bundle.

    Args:
        show_folder: Source show folder (must contain ``show.toml``).
        output_path: Destination archive path.
        with_audio: Include audio files. Ignored when ``index_only=True``.
        index_only: Skip show folder content; export only LanceDB collections.
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

    store = get_index_store()
    collections = _show_collections(show_name, store)

    mode = Mode.INDEX_ONLY if index_only else Mode.FULL
    if mode == Mode.INDEX_ONLY and not collections:
        raise ValueError(f"show '{show_name}' has no LanceDB collections to export")

    audio_in_bundle = with_audio and not index_only

    manifest = Manifest(
        mode=mode,
        podcodex_version=_podcodex_version(),
        exported_at=_now_iso(),
        shows=[
            ShowEntry(
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
        cols = _show_collections(name, store, info=info)
        if not cols:
            logger.warning(f"show '{name}' has no collections — skipping")
            continue
        shows.append(
            ShowEntry(
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
